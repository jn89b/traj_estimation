#!/usr/bin/env python3
"""Interpolate low-rate GeoPose onto high-rate IMU time.

This node fuses two asynchronous streams:
1) ``/ap/geopose/filtered`` (lower rate pose in geodetic coordinates), and
2) ``/ap/imu/experimental/data`` (higher rate IMU).

Workflow:
- Buffer incoming GeoPose samples with timestamps.
- Convert GeoPose orientation from ENU/FLU to NED/FRD once at ingest.
- On each IMU message, interpolate the two surrounding GeoPose samples to the
    IMU timestamp (linear for lat/lon/alt, SLERP for quaternion).
- Publish the latest synchronized IMU + interpolated GeoPose at a controlled
    output rate.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import rclpy
from geographic_msgs.msg import GeoPoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from traj_estimation.rotation_utils import (
    geopose_enu_flu_to_ned_frd,
    slerp_quat,
)
from traj_estimation.msg import SyncedNavImu

@dataclass
class GeoSample:
    """Buffered geopose sample stored in interpolation-friendly form.

    The quaternion is stored in NED/FRD after conversion so interpolation uses
    the same frame that is later published.
    """

    stamp_ns: int
    lat: float
    lon: float
    alt: float
    quat: Tuple[float, float, float, float]  # NED/FRD, ROS field order x,y,z,w

class InterpolateNode(Node):
    """ROS 2 node that time-aligns GeoPose with IMU and republishes both.

    The node keeps a short history of GeoPose samples and computes an
    interpolated GeoPose for each IMU timestamp. Publishing is timer-driven so
    downstream consumers receive data at a predictable rate.
    """

    def __init__(self) -> None:
        """Initialize parameters, subscriptions, publishers, and timer.

        Input parameters (ROS params):
            imu_topic (str): IMU input topic.
            geopose_topic (str): Filtered GeoPose input topic.
            output_imu_topic (str): Republished synchronized IMU topic.
            output_geopose_topic (str): Interpolated GeoPose output topic.
            output_synced_topic (str): Combined nav+IMU synchronized output.
            output_frame_id (str): Frame id set on published GeoPose.
            publish_rate_hz (float): Controlled output publish rate.
            max_geo_buffer_sec (float): Max age for stored GeoPose samples.
            qos_depth (int): QoS queue depth for pubs/subs.
        """
        super().__init__('interpolate_node')

        self.declare_parameter('imu_topic', '/ap/imu/experimental/data')
        self.declare_parameter('geopose_topic', '/ap/geopose/filtered')
        self.declare_parameter(
            'output_imu_topic',
            '/ap/imu/experimental/data/synced',
        )
        self.declare_parameter(
            'output_geopose_topic',
            '/ap/geopose/filtered/interpolated_ned',
        )
        self.declare_parameter('output_synced_topic', '/ap/state/synced')
        self.declare_parameter('output_frame_id', 'base_link_ned')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('max_geo_buffer_sec', 5.0)
        self.declare_parameter('qos_depth', 100)

        imu_topic = self.get_parameter('imu_topic').value
        geopose_topic = self.get_parameter('geopose_topic').value

        self.output_imu_topic = self.get_parameter('output_imu_topic').value
        self.output_geopose_topic = (
            self.get_parameter('output_geopose_topic').value
        )
        self.output_synced_topic = (
            self.get_parameter('output_synced_topic').value
        )
        self.output_frame_id = self.get_parameter('output_frame_id').value
        self.publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        self.max_geo_buffer_sec = float(
            self.get_parameter('max_geo_buffer_sec').value
        )
        qos_depth = int(self.get_parameter('qos_depth').value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )

        self.geo_buffer: Deque[GeoSample] = deque()
        self.latest_imu_for_pub: Optional[Imu] = None
        self.latest_geo_for_pub: Optional[GeoPoseStamped] = None

        self.create_subscription(Imu, imu_topic, self.imu_cb, qos)
        self.create_subscription(
            GeoPoseStamped,
            geopose_topic,
            self.geopose_cb,
            qos,
        )

        self.imu_pub = self.create_publisher(
            Imu,
            self.output_imu_topic,
            qos,
        )
        self.geo_pub = self.create_publisher(
            GeoPoseStamped,
            self.output_geopose_topic,
            qos,
        )
        self.synced_pub = self.create_publisher(
            SyncedNavImu,
            self.output_synced_topic,
            qos,
        )

        timer_period = 1.0 / max(self.publish_rate_hz, 1e-3)
        self.publish_timer = self.create_timer(
            timer_period,
            self.publish_cb,
        )

        self.get_logger().info(
            f'Interpolating {geopose_topic} to IMU time from {imu_topic}; '
            f'converting GeoPose orientation to NED/FRD; publishing at '
            f'{self.publish_rate_hz:.2f} Hz to '
            f'{self.output_geopose_topic}, {self.output_imu_topic}, and '
            f'{self.output_synced_topic}'
        )

    def _stamp_to_ns(self, sec: int, nanosec: int) -> int:
        """Convert ROS 2 Time fields into integer nanoseconds.

        Args:
            sec: Seconds component of ROS time.
            nanosec: Nanoseconds component of ROS time.

        Returns:
            Timestamp expressed as total nanoseconds.
        """
        return int(sec) * 1_000_000_000 + int(nanosec)

    def geopose_cb(self, msg: GeoPoseStamped) -> None:
        """Store GeoPose sample in buffer after frame conversion.

        Orientation is converted from ENU/FLU to NED/FRD at ingest time. Old
        samples are dropped to keep memory bounded and interpolation local.

        Args:
            msg: Incoming filtered GeoPose sample.
        """
        stamp = msg.header.stamp
        stamp_ns = self._stamp_to_ns(stamp.sec, stamp.nanosec)

        if stamp_ns <= 0:
            stamp_ns = self.get_clock().now().nanoseconds

        quat_enu = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

        # Convert once here, before interpolation.
        quat_ned = geopose_enu_flu_to_ned_frd(quat_enu)

        self.geo_buffer.append(
            GeoSample(
                stamp_ns=stamp_ns,
                lat=msg.pose.position.latitude,
                lon=msg.pose.position.longitude,
                alt=msg.pose.position.altitude,
                quat=quat_ned,
            )
        )

        now_ns = self.get_clock().now().nanoseconds
        min_keep_ns = now_ns - int(
            self.max_geo_buffer_sec * 1_000_000_000
        )

        while (
            len(self.geo_buffer) > 2
            and self.geo_buffer[1].stamp_ns < min_keep_ns
        ):
            self.geo_buffer.popleft()

    def _interpolate_geo(self, target_ns: int) -> Optional[GeoPoseStamped]:
        """Interpolate buffered GeoPose to ``target_ns``.

        If ``target_ns`` lies outside the buffer, the nearest endpoint sample is
        used. Inside the buffer, position is linearly interpolated and attitude
        uses spherical linear interpolation (SLERP).

        Args:
            target_ns: Timestamp to interpolate to, in nanoseconds.

        Returns:
            Interpolated GeoPoseStamped at ``target_ns`` if data is available,
            otherwise ``None``.
        """
        if not self.geo_buffer:
            return None

        if len(self.geo_buffer) == 1:
            return self._build_geo_msg(target_ns, self.geo_buffer[0])

        samples = self.geo_buffer

        if target_ns <= samples[0].stamp_ns:
            return self._build_geo_msg(target_ns, samples[0])

        if target_ns >= samples[-1].stamp_ns:
            return self._build_geo_msg(target_ns, samples[-1])

        for i in range(len(samples) - 1):
            a = samples[i]
            b = samples[i + 1]

            if a.stamp_ns <= target_ns <= b.stamp_ns:
                dt = b.stamp_ns - a.stamp_ns

                if dt <= 0:
                    return self._build_geo_msg(target_ns, a)

                t = (target_ns - a.stamp_ns) / float(dt)

                lat = a.lat + t * (b.lat - a.lat)
                lon = a.lon + t * (b.lon - a.lon)
                alt = a.alt + t * (b.alt - a.alt)
                q_ned = slerp_quat(a.quat, b.quat, t)

                out = GeoPoseStamped()
                out.header.stamp.sec = int(target_ns // 1_000_000_000)
                out.header.stamp.nanosec = int(target_ns % 1_000_000_000)
                out.header.frame_id = self.output_frame_id

                out.pose.position.latitude = lat
                out.pose.position.longitude = lon
                out.pose.position.altitude = alt

                out.pose.orientation.x = q_ned[0]
                out.pose.orientation.y = q_ned[1]
                out.pose.orientation.z = q_ned[2]
                out.pose.orientation.w = q_ned[3]

                return out

        return None

    def _build_geo_msg(
        self,
        target_ns: int,
        sample: GeoSample,
    ) -> GeoPoseStamped:
        """Build output GeoPoseStamped from one buffered sample.

        Args:
            target_ns: Timestamp to assign to output message.
            sample: GeoSample values copied into output fields.

        Returns:
            GeoPoseStamped populated from ``sample`` and ``target_ns``.
        """
        out = GeoPoseStamped()

        out.header.stamp.sec = int(target_ns // 1_000_000_000)
        out.header.stamp.nanosec = int(target_ns % 1_000_000_000)
        out.header.frame_id = self.output_frame_id

        out.pose.position.latitude = sample.lat
        out.pose.position.longitude = sample.lon
        out.pose.position.altitude = sample.alt

        out.pose.orientation.x = sample.quat[0]
        out.pose.orientation.y = sample.quat[1]
        out.pose.orientation.z = sample.quat[2]
        out.pose.orientation.w = sample.quat[3]

        return out

    def imu_cb(self, msg: Imu) -> None:
        """Process IMU sample and compute time-matched GeoPose.

        This callback does not publish directly. It updates the latest aligned
        pair, which is published by ``publish_cb`` at the configured rate.

        Args:
            msg: Incoming IMU sample used as interpolation time reference.
        """
        stamp = msg.header.stamp
        target_ns = self._stamp_to_ns(stamp.sec, stamp.nanosec)

        if target_ns <= 0:
            target_ns = self.get_clock().now().nanoseconds

        interp_geo = self._interpolate_geo(target_ns)

        if interp_geo is None:
            return

        self.latest_imu_for_pub = msg
        self.latest_geo_for_pub = interp_geo

    def publish_cb(self) -> None:
        """Publish most recent synchronized IMU and interpolated GeoPose.

        This method publishes only when both cached messages are available.
        """
        if (
            self.latest_imu_for_pub is None
            or self.latest_geo_for_pub is None
        ):
            return

        self.imu_pub.publish(self.latest_imu_for_pub)
        self.geo_pub.publish(self.latest_geo_for_pub)

        synced = SyncedNavImu()
        synced.header = self.latest_geo_for_pub.header
        synced.latitude = self.latest_geo_for_pub.pose.position.latitude
        synced.longitude = self.latest_geo_for_pub.pose.position.longitude
        synced.altitude = self.latest_geo_for_pub.pose.position.altitude
        synced.imu = self.latest_imu_for_pub
        self.synced_pub.publish(synced)


def main(args=None) -> None:
    """Entrypoint for the interpolation node process.

    Args:
        args: Optional ROS command-line arguments.
    """
    rclpy.init(args=args)
    node = InterpolateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()