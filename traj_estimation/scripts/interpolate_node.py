#!/usr/bin/env python3
"""Stream a fixed-rate Inspyre-compatible navigation + IMU topic.

This node is the live equivalent of the batch path:

    Pose -> Pose.r_e_eb -> linear/CubicSpline(ECEF) -> quaternion.slerp
         -> ecef_to_ned -> Pose

Unlike a timer that republishes the latest cached message, this node creates a
fixed output-time grid (150 Hz by default). For every grid timestamp it:

1. Interpolates position in ECEF using either ``np.interp`` or a cached,
   local four-knot ``CubicSpline``.
2. SLERPs the GeoPose attitude with ``quaternion.slerp``.
3. Linearly interpolates gyro and accelerometer values from bracketing IMUs.
4. Publishes the interpolated Inspyre attitude ``q_nb`` in the output IMU
   orientation field, using standard ROS ``x,y,z,w`` storage.
5. Publishes exactly one GeoPose, IMU, and SyncedNavImu message.

True interpolation requires future endpoints. ``linear`` position mode is
usually delayed by roughly one GeoPose period plus one IMU period (about 40 ms
at 30 Hz GeoPose / 200 Hz IMU). ``cubic`` uses a four-knot local window and
usually has about one additional GeoPose period of delay (about 70 ms total).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple, TypeVar

import numpy as np
import quaternion
from scipy.interpolate import CubicSpline
import rclpy
from geographic_msgs.msg import GeoPoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu

# Same Inspyre APIs used by the original batch interpolation function.
from traj_estimation.inspyre.states import Pose
from traj_estimation.inspyre.earth import ecef_to_ned

from traj_estimation.rotation_utils import geopose_enu_flu_to_ned_frd
from traj_estimation_msgs.msg import SyncedNavImu


NSEC_PER_SEC = 1_000_000_000
T = TypeVar("T")


@dataclass(frozen=True)
class GeoSample:
    """One filtered GeoPose stored through the Inspyre Pose API."""

    stamp_ns: int
    time_s: float
    pose: Pose
    r_e_eb: np.ndarray
    q_nb: quaternion.quaternion


@dataclass(frozen=True)
class ImuSample:
    """One raw IMU message with its source timestamp."""

    stamp_ns: int
    msg: Imu


@dataclass(frozen=True)
class CubicEcefSegment:
    """Cached local cubic position spline for one completed GeoPose interval.

    The spline uses four neighboring GeoPose samples ``g0, g1, g2, g3`` and is
    evaluated only over the interior interval ``[g1, g2]``.  A segment is built
    once when the newest GPS/GeoPose point makes that interval complete, then
    reused for every 150 Hz output timestamp in that interval.
    """

    start_ns: int
    end_ns: int
    reference_time_s: float
    spline_x: CubicSpline
    spline_y: CubicSpline
    spline_z: CubicSpline


class InterpolateNode(Node):
    """Resample filtered GeoPose and IMU onto one fixed-rate timeline."""

    def __init__(self) -> None:
        super().__init__("interpolate_node")

        self.declare_parameter("imu_topic", "/ap/imu/experimental/data")
        self.declare_parameter("geopose_topic", "/ap/geopose/filtered")
        self.declare_parameter(
            "output_imu_topic",
            "/ap/imu/experimental/data/resampled",
        )
        self.declare_parameter(
            "output_geopose_topic",
            "/ap/geopose/filtered/interpolated_ned",
        )
        self.declare_parameter(
            "output_synced_topic",
            "/ap/state/synced",
        )
        # ``GeoPoseStamped`` contains global LLA plus a local-navigation
        # attitude.  This label tells downstream consumers that the published
        # attitude is q_nb (body FRD -> local NED), not ROS ENU/FLU.
        self.declare_parameter("output_frame_id", "ned")
        # IMU vectors remain in the physical ArduPilot body frame: FRD.
        self.declare_parameter("output_imu_frame_id", "base_link_frd")
        self.declare_parameter("output_rate_hz", 150.0)
        self.declare_parameter(
            "position_interpolation",
            "linear",
        )  # "linear" or "cubic"
        self.declare_parameter("max_buffer_sec", 2.0)
        self.declare_parameter("qos_depth", 300)

        imu_topic = str(self.get_parameter("imu_topic").value)
        geopose_topic = str(self.get_parameter("geopose_topic").value)
        self.output_imu_topic = str(
            self.get_parameter("output_imu_topic").value
        )
        self.output_geopose_topic = str(
            self.get_parameter("output_geopose_topic").value
        )
        self.output_synced_topic = str(
            self.get_parameter("output_synced_topic").value
        )
        self.output_frame_id = str(
            self.get_parameter("output_frame_id").value
        )
        self.output_imu_frame_id = str(
            self.get_parameter("output_imu_frame_id").value
        )
        self.output_rate_hz = float(
            self.get_parameter("output_rate_hz").value
        )
        self.position_interpolation = str(
            self.get_parameter("position_interpolation").value
        ).strip().lower()
        self.max_buffer_sec = float(
            self.get_parameter("max_buffer_sec").value
        )
        qos_depth = int(self.get_parameter("qos_depth").value)

        if self.output_rate_hz <= 0.0:
            raise ValueError("output_rate_hz must be > 0")
        if self.position_interpolation not in {"linear", "cubic"}:
            raise ValueError(
                "position_interpolation must be 'linear' or 'cubic'"
            )
        if self.max_buffer_sec <= 0.0:
            raise ValueError("max_buffer_sec must be > 0")

        self.period_ns = NSEC_PER_SEC / self.output_rate_hz

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )

        self.geo_buffer: Deque[GeoSample] = deque()
        self.imu_buffer: Deque[ImuSample] = deque()
        self.last_geo_stamp_ns: Optional[int] = None
        self.last_imu_stamp_ns: Optional[int] = None

        # Cubic mode caches one spline for each completed interior GPS interval.
        # This cache is updated only when a new GeoPose sample arrives.
        self.cubic_segments: dict[Tuple[int, int], CubicEcefSegment] = {}

        # The grid starts only after both streams have enough samples for the
        # selected position interpolation method.
        self.grid_anchor_ns: Optional[int] = None
        self.grid_index = 0

        self.create_subscription(Imu, imu_topic, self.imu_cb, qos)
        self.create_subscription(
            GeoPoseStamped,
            geopose_topic,
            self.geopose_cb,
            qos,
        )

        self.imu_pub = self.create_publisher(Imu, self.output_imu_topic, qos)
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

        self._validate_ned_conversion()

        self.get_logger().info(
            f"Fixed-rate Inspyre NED/FRD interpolation enabled at "
            f"{self.output_rate_hz:.3f} Hz; "
            f"position_mode={self.position_interpolation}: "
            f"{geopose_topic} + {imu_topic} -> {self.output_synced_topic}"
        )

    @staticmethod
    def _stamp_to_ns(sec: int, nanosec: int) -> int:
        return int(sec) * NSEC_PER_SEC + int(nanosec)

    @staticmethod
    def _set_stamp(header, stamp_ns: int) -> None:
        header.stamp.sec = int(stamp_ns // NSEC_PER_SEC)
        header.stamp.nanosec = int(stamp_ns % NSEC_PER_SEC)

    @staticmethod
    def _xyzw_to_quaternion(
        xyzw: Tuple[float, float, float, float],
    ) -> quaternion.quaternion:
        """Convert ROS XYZW storage to numpy-quaternion WXYZ storage."""
        x, y, z, w = xyzw
        q = quaternion.quaternion(float(w), float(x), float(y), float(z))
        if abs(q) < 1e-12:
            raise ValueError("zero-norm quaternion")
        return q.normalized()

    @staticmethod
    def _quaternion_to_xyzw(
        q: quaternion.quaternion,
    ) -> Tuple[float, float, float, float]:
        q = q.normalized()
        return float(q.x), float(q.y), float(q.z), float(q.w)

    def _validate_ned_conversion(self) -> None:
        """Fail fast if the ENU/FLU -> NED/FRD helper has wrong semantics.

        ArduPilot emits a level vehicle pointing north on ``/ap/geopose/filtered``
        as the ROS ENU/FLU quaternion [x, y, z, w] =
        [0, 0, sqrt(1/2), sqrt(1/2)].  In Inspyre's required q_nb convention
        (body FRD -> navigation NED), the same physical attitude is identity.
        """
        half_sqrt_2 = float(np.sqrt(0.5))
        q_xyzw = geopose_enu_flu_to_ned_frd(
            (0.0, 0.0, half_sqrt_2, half_sqrt_2)
        )
        q_nb = self._xyzw_to_quaternion(q_xyzw)
        identity = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        # q and -q represent the same attitude.
        if abs(float(np.dot(quaternion.as_float_array(q_nb),
                            quaternion.as_float_array(identity)))) < 1.0 - 1e-6:
            raise RuntimeError(
                "geopose_enu_flu_to_ned_frd does not produce q_nb identity "
                "for level/north input; do not run with mixed frame semantics."
            )

    def _attach_ned_attitude(
        self,
        imu: Imu,
        q_nb: quaternion.quaternion,
    ) -> None:
        """Put Inspyre's q_nb in output IMU orientation as standard ROS XYZW.

        ``q_nb`` rotates a body-FRD vector into the local NED navigation frame.
        Acceleration and gyro fields remain body-FRD, which is the convention
        expected by Inspyre's inertial-navigation routines.
        """
        qx, qy, qz, qw = self._quaternion_to_xyzw(q_nb)
        imu.header.frame_id = self.output_imu_frame_id
        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw
        # GeoPose does not carry an attitude covariance.  ROS uses an all-zero
        # covariance to mean "unknown covariance" (not zero uncertainty).
        imu.orientation_covariance = [0.0] * 9

    def _geopose_to_sample(self, msg: GeoPoseStamped) -> Optional[GeoSample]:
        """Construct the same Inspyre Pose representation used offline."""
        stamp_ns = self._stamp_to_ns(
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
        )
        if stamp_ns <= 0:
            self.get_logger().warn("Dropping GeoPose with zero timestamp.")
            return None

        lla_deg = np.array(
            [
                msg.pose.position.latitude,
                msg.pose.position.longitude,
                msg.pose.position.altitude,
            ],
            dtype=float,
        )

        # Keep the existing node's conversion before using Inspyre Pose/q_nb.
        q_enu_flu_xyzw = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        q_ned_frd_xyzw = geopose_enu_flu_to_ned_frd(q_enu_flu_xyzw)

        try:
            q_nb = self._xyzw_to_quaternion(q_ned_frd_xyzw)
        except ValueError as exc:
            self.get_logger().warn(f"Dropping GeoPose: {exc}")
            return None

        time_s = stamp_ns / float(NSEC_PER_SEC)
        pose = Pose(lla_deg, q_nb, time_s, radians=False)

        return GeoSample(
            stamp_ns=stamp_ns,
            time_s=time_s,
            pose=pose,
            r_e_eb=np.asarray(pose.r_e_eb, dtype=float),
            q_nb=pose.q_nb.normalized(),
        )

    def _target_stamp_ns(self) -> int:
        """Return the next fixed-rate output timestamp without drift."""
        assert self.grid_anchor_ns is not None
        return int(round(self.grid_anchor_ns + self.grid_index * self.period_ns))

    def _initialize_grid_if_ready(self) -> bool:
        """Anchor the fixed-rate grid once both source streams are usable.

        Linear interpolation needs two GeoPose samples.  Local cubic mode needs
        four samples and begins at the second knot, where the first completed
        interior interval ``[g1, g2]`` can eventually be evaluated.
        """
        if self.grid_anchor_ns is not None:
            return True

        required_geo = 4 if self.position_interpolation == "cubic" else 2
        if len(self.geo_buffer) < required_geo or len(self.imu_buffer) < 2:
            return False

        geo_start = (
            self.geo_buffer[1]
            if self.position_interpolation == "cubic"
            else self.geo_buffer[0]
        )
        self.grid_anchor_ns = max(
            geo_start.stamp_ns,
            self.imu_buffer[0].stamp_ns,
        )
        self.grid_index = 0
        self.get_logger().info(
            "Output grid anchored at "
            f"{self.grid_anchor_ns / NSEC_PER_SEC:.9f} s "
            f"({self.position_interpolation} position mode)."
        )
        return True

    @staticmethod
    def _bracket(
        samples: Deque[T],
        target_ns: int,
    ) -> Optional[Tuple[T, T]]:
        """Return source samples that bracket ``target_ns``.

        The deque must contain objects with a ``stamp_ns`` attribute and be
        sorted in ascending timestamp order. Samples already older than the
        current target are discarded, but the left endpoint is retained.
        """
        while len(samples) >= 2 and samples[1].stamp_ns <= target_ns:
            samples.popleft()

        if not samples:
            return None
        if target_ns < samples[0].stamp_ns:
            return None

        if len(samples) == 1:
            if target_ns == samples[0].stamp_ns:
                return samples[0], samples[0]
            return None

        return samples[0], samples[1]

    @staticmethod
    def _fraction(a_ns: int, b_ns: int, target_ns: int) -> float:
        """Return interpolation fraction, including the exact endpoint case."""
        if b_ns <= a_ns:
            return 0.0
        return float(np.clip(
            (target_ns - a_ns) / float(b_ns - a_ns),
            0.0,
            1.0,
        ))

    def _find_geo_pair(
        self,
        target_ns: int,
    ) -> Optional[Tuple[GeoSample, GeoSample]]:
        """Return the two GeoPose samples that bracket ``target_ns``.

        This lookup does not discard GPS history because cubic mode needs a
        neighbor on either side of the active interpolation interval.
        """
        if len(self.geo_buffer) < 2:
            return None

        samples = self.geo_buffer
        if (
            target_ns < samples[0].stamp_ns
            or target_ns > samples[-1].stamp_ns
        ):
            return None

        for a, b in zip(samples, list(samples)[1:]):
            if a.stamp_ns <= target_ns <= b.stamp_ns:
                return a, b
        return None

    def _find_cubic_window(
        self,
        target_ns: int,
    ) -> Optional[Tuple[GeoSample, GeoSample, GeoSample, GeoSample]]:
        """Return ``g0,g1,g2,g3`` with target in the interior [g1, g2]."""
        if len(self.geo_buffer) < 4:
            return None

        samples = list(self.geo_buffer)
        for index in range(1, len(samples) - 2):
            g0 = samples[index - 1]
            g1 = samples[index]
            g2 = samples[index + 1]
            g3 = samples[index + 2]
            if g1.stamp_ns <= target_ns <= g2.stamp_ns:
                return g0, g1, g2, g3
        return None

    def _get_or_create_cubic_segment(
        self,
        g0: GeoSample,
        g1: GeoSample,
        g2: GeoSample,
        g3: GeoSample,
    ) -> CubicEcefSegment:
        """Create one cached four-knot local ECEF CubicSpline when needed."""
        key = (g1.stamp_ns, g2.stamp_ns)
        cached = self.cubic_segments.get(key)
        if cached is not None:
            return cached

        reference_time_s = g1.time_s
        knot_times_s = np.array(
            [g0.time_s, g1.time_s, g2.time_s, g3.time_s],
            dtype=float,
        ) - reference_time_s

        if not np.all(np.diff(knot_times_s) > 0.0):
            raise ValueError("GeoPose timestamps must be strictly increasing")

        xyz = np.vstack(
            [g0.r_e_eb, g1.r_e_eb, g2.r_e_eb, g3.r_e_eb]
        )

        # Default ``not-a-knot`` boundary conditions match CubicSpline(...) in
        # the original offline Inspyre interpolation function.
        segment = CubicEcefSegment(
            start_ns=g1.stamp_ns,
            end_ns=g2.stamp_ns,
            reference_time_s=reference_time_s,
            spline_x=CubicSpline(knot_times_s, xyz[:, 0]),
            spline_y=CubicSpline(knot_times_s, xyz[:, 1]),
            spline_z=CubicSpline(knot_times_s, xyz[:, 2]),
        )
        self.cubic_segments[key] = segment
        return segment

    def _interpolate_pose(self, target_ns: int) -> Optional[Pose]:
        """Interpolate NED/FRD Pose with selected ECEF position method.

        ``linear`` uses the two bracketing GeoPose samples.  ``cubic`` uses a
        cached four-knot local CubicSpline, but only over its middle interval.
        Attitude remains SLERP between the same two interval endpoints in both
        modes, which is intentionally low-cost and stable in real time.
        """
        target_s = target_ns / float(NSEC_PER_SEC)

        if self.position_interpolation == "linear":
            pair = self._find_geo_pair(target_ns)
            if pair is None:
                return None
            a, b = pair
            if a.stamp_ns == b.stamp_ns:
                return Pose(
                    np.array([a.pose.lat, a.pose.lon, a.pose.alt]),
                    a.q_nb,
                    target_s,
                    radians=True,
                )

            time_pair = np.array([a.time_s, b.time_s], dtype=float)
            x = np.interp(target_s, time_pair, [a.r_e_eb[0], b.r_e_eb[0]])
            y = np.interp(target_s, time_pair, [a.r_e_eb[1], b.r_e_eb[1]])
            z = np.interp(target_s, time_pair, [a.r_e_eb[2], b.r_e_eb[2]])
        else:
            window = self._find_cubic_window(target_ns)
            if window is None:
                return None
            g0, a, b, g3 = window
            segment = self._get_or_create_cubic_segment(g0, a, b, g3)
            local_t = target_s - segment.reference_time_s
            x = float(segment.spline_x(local_t))
            y = float(segment.spline_y(local_t))
            z = float(segment.spline_z(local_t))

        q_nb = quaternion.slerp(
            a.q_nb,
            b.q_nb,
            a.time_s,
            b.time_s,
            target_s,
        ).normalized()

        # Inspyre retains this historical function name; it returns LLA radians.
        lla_rad = ecef_to_ned(np.array([x, y, z], dtype=float))
        return Pose(lla_rad, q_nb, target_s, radians=True)

    def _interpolate_imu(
        self,
        a: ImuSample,
        b: ImuSample,
        target_ns: int,
    ) -> Imu:
        """Resample the IMU values onto the same output timestamp.

        Gyro and accelerometer values are linearly interpolated in ArduPilot's
        body-FRD axes.  Attitude is injected later from the synchronized
        Inspyre q_nb pose, so this method intentionally ignores the raw IMU
        orientation encoding.
        """
        u = self._fraction(a.stamp_ns, b.stamp_ns, target_ns)
        a_msg = a.msg
        b_msg = b.msg

        out = Imu()
        self._set_stamp(out.header, target_ns)
        out.header.frame_id = self.output_imu_frame_id

        gyro_a = np.array(
            [
                a_msg.angular_velocity.x,
                a_msg.angular_velocity.y,
                a_msg.angular_velocity.z,
            ],
            dtype=float,
        )
        gyro_b = np.array(
            [
                b_msg.angular_velocity.x,
                b_msg.angular_velocity.y,
                b_msg.angular_velocity.z,
            ],
            dtype=float,
        )
        accel_a = np.array(
            [
                a_msg.linear_acceleration.x,
                a_msg.linear_acceleration.y,
                a_msg.linear_acceleration.z,
            ],
            dtype=float,
        )
        accel_b = np.array(
            [
                b_msg.linear_acceleration.x,
                b_msg.linear_acceleration.y,
                b_msg.linear_acceleration.z,
            ],
            dtype=float,
        )

        gyro = gyro_a + u * (gyro_b - gyro_a)
        accel = accel_a + u * (accel_b - accel_a)

        out.angular_velocity.x = float(gyro[0])
        out.angular_velocity.y = float(gyro[1])
        out.angular_velocity.z = float(gyro[2])
        out.linear_acceleration.x = float(accel[0])
        out.linear_acceleration.y = float(accel[1])
        out.linear_acceleration.z = float(accel[2])

        # Do not propagate /ap/imu/experimental/data.orientation here.
        # ArduPilot publishes q_bn (NED -> body FRD) in q1,q2,q3,q4 order
        # stored in ROS fields x,y,z,w.  The output instead receives q_nb from
        # the interpolated Inspyre Pose in _attach_ned_attitude().

        # The model does not consume covariance, but retain source metadata.
        out.angular_velocity_covariance = a_msg.angular_velocity_covariance
        out.linear_acceleration_covariance = a_msg.linear_acceleration_covariance
        return out

    def _pose_to_geopose_msg(
        self,
        pose: Pose,
        stamp_ns: int,
    ) -> GeoPoseStamped:
        """Serialize the interpolated pose with standard-XYZW q_nb attitude.

        LLA stays WGS-84 geodetic.  Only the attitude convention is local NED
        navigation with FRD body axes, as required by Inspyre.
        """
        qx, qy, qz, qw = self._quaternion_to_xyzw(pose.q_nb)

        out = GeoPoseStamped()
        self._set_stamp(out.header, stamp_ns)
        out.header.frame_id = self.output_frame_id
        out.pose.position.latitude = float(np.rad2deg(pose.lat))
        out.pose.position.longitude = float(np.rad2deg(pose.lon))
        out.pose.position.altitude = float(pose.alt)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        return out

    def _publish_pair(self, imu: Imu, geo: GeoPoseStamped) -> None:
        """Publish one exact-rate NED/FRD time-aligned output sample."""
        self.imu_pub.publish(imu)
        self.geo_pub.publish(geo)

        synced = SyncedNavImu()
        synced.header = geo.header
        synced.latitude = geo.pose.position.latitude
        synced.longitude = geo.pose.position.longitude
        synced.altitude = geo.pose.position.altitude
        synced.imu = imu
        self.synced_pub.publish(synced)

    def _advance_grid_past_unrecoverable_gap(self) -> bool:
        """Skip timestamps permanently older than the remaining source buffers."""
        if self.grid_anchor_ns is None or not self.imu_buffer:
            return False

        required_geo = 4 if self.position_interpolation == "cubic" else 2
        if len(self.geo_buffer) < required_geo:
            return False

        # Cubic evaluation cannot begin before the second GeoPose knot because
        # it needs g0 before and g3 after the active [g1, g2] interval.
        geo_earliest_ns = (
            self.geo_buffer[1].stamp_ns
            if self.position_interpolation == "cubic"
            else self.geo_buffer[0].stamp_ns
        )

        target_ns = self._target_stamp_ns()
        earliest_possible_ns = max(
            geo_earliest_ns,
            self.imu_buffer[0].stamp_ns,
        )
        if target_ns >= earliest_possible_ns:
            return False

        old_index = self.grid_index
        self.grid_index = int(np.ceil(
            (earliest_possible_ns - self.grid_anchor_ns) / self.period_ns
        ))
        self.grid_index = max(self.grid_index, old_index)
        return self.grid_index != old_index

    def _process_ready_outputs(self) -> None:
        """Publish all fixed-grid outputs currently bracketed by both streams."""
        if not self._initialize_grid_if_ready():
            return

        while True:
            if self._advance_grid_past_unrecoverable_gap():
                # A PredictionNode should reset its sequence after this gap.
                continue

            target_ns = self._target_stamp_ns()
            imu_pair = self._bracket(self.imu_buffer, target_ns)

            # The next target still needs a future IMU endpoint.
            if imu_pair is None:
                return

            try:
                interp_pose = self._interpolate_pose(target_ns)
                # The selected GeoPose method does not yet have enough future
                # knots: wait rather than extrapolating or retimestamping stale
                # GPS data.
                if interp_pose is None:
                    return
                interp_imu = self._interpolate_imu(
                    imu_pair[0],
                    imu_pair[1],
                    target_ns,
                )
                self._attach_ned_attitude(interp_imu, interp_pose.q_nb)
            except (ValueError, FloatingPointError) as exc:
                self.get_logger().warn(
                    f"Skipping output at {target_ns}: {exc}"
                )
                self.grid_index += 1
                continue

            interp_geo = self._pose_to_geopose_msg(interp_pose, target_ns)
            self._publish_pair(interp_imu, interp_geo)
            self.grid_index += 1

    @staticmethod
    def _trim_buffer(
        buffer: Deque[T],
        newest_ns: int,
        max_age_ns: int,
        minimum_samples: int = 2,
    ) -> None:
        """Bound storage while retaining enough source samples for the mode."""
        min_keep_ns = newest_ns - max_age_ns
        while (
            len(buffer) > minimum_samples
            and buffer[1].stamp_ns < min_keep_ns
        ):
            buffer.popleft()

    def imu_cb(self, msg: Imu) -> None:
        """Buffer an IMU sample and emit any newly bracketed grid timestamps."""
        stamp_ns = self._stamp_to_ns(
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
        )
        if stamp_ns <= 0:
            self.get_logger().warn("Dropping IMU with zero timestamp.")
            return
        if (
            self.last_imu_stamp_ns is not None
            and stamp_ns <= self.last_imu_stamp_ns
        ):
            self.get_logger().warn("Dropping non-monotonic IMU timestamp.")
            return

        self.last_imu_stamp_ns = stamp_ns
        self.imu_buffer.append(ImuSample(stamp_ns=stamp_ns, msg=msg))
        self._trim_buffer(
            self.imu_buffer,
            stamp_ns,
            int(self.max_buffer_sec * NSEC_PER_SEC),
        )
        self._process_ready_outputs()

    def geopose_cb(self, msg: GeoPoseStamped) -> None:
        """Buffer a filtered GeoPose and emit any newly completed outputs."""
        sample = self._geopose_to_sample(msg)
        if sample is None:
            return
        if (
            self.last_geo_stamp_ns is not None
            and sample.stamp_ns <= self.last_geo_stamp_ns
        ):
            self.get_logger().warn("Dropping non-monotonic GeoPose timestamp.")
            return

        self.last_geo_stamp_ns = sample.stamp_ns
        self.geo_buffer.append(sample)

        # Cache only the newest completed local cubic interval.  This is the
        # requested "keep the previous N points and update with the newest GPS"
        # behavior: each new GPS point enables one new [g1, g2] spline segment.
        if self.position_interpolation == "cubic" and len(self.geo_buffer) >= 4:
            g0, g1, g2, g3 = list(self.geo_buffer)[-4:]
            try:
                self._get_or_create_cubic_segment(g0, g1, g2, g3)
            except ValueError as exc:
                self.get_logger().warn(f"Skipping cubic cache update: {exc}")

        self._trim_buffer(
            self.geo_buffer,
            sample.stamp_ns,
            int(self.max_buffer_sec * NSEC_PER_SEC),
            minimum_samples=(
                4 if self.position_interpolation == "cubic" else 2
            ),
        )

        # Prune spline cache entries that cannot be used with retained GPS data.
        if self.geo_buffer:
            keep_from_ns = self.geo_buffer[0].stamp_ns
            self.cubic_segments = {
                key: segment
                for key, segment in self.cubic_segments.items()
                if segment.end_ns >= keep_from_ns
            }

        self._process_ready_outputs()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InterpolateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
