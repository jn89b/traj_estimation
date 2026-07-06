#!/usr/bin/env python3
"""Consume synchronized navigation + IMU samples for prediction.

This node subscribes to one fused topic that already contains aligned
latitude/longitude/altitude and IMU data. Downstream prediction logic can use
this callback directly without having to synchronize multiple subscriptions.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from traj_estimation.msg import SyncedNavImu


class PredictionNode(Node):
    """Subscriber node for synchronized state estimation input."""

    def __init__(self) -> None:
        """Initialize subscriber and runtime parameters.

        Input parameters (ROS params):
            synced_topic (str): Topic containing synchronized nav and IMU.
            qos_depth (int): QoS queue depth used for the subscription.
        """
        super().__init__('prediction_node')

        self.declare_parameter('synced_topic', '/ap/state/synced')
        # Change the depth to what we need for the network input
        self.declare_parameter('qos_depth', 100)

        synced_topic = self.get_parameter('synced_topic').value
        qos_depth = int(self.get_parameter('qos_depth').value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )

        self.latest_synced: SyncedNavImu | None = None

        self.create_subscription(
            SyncedNavImu,
            synced_topic,
            self.synced_cb,
            qos,
        )

        self.get_logger().info(
            f'Listening for synchronized nav+IMU on {synced_topic}'
        )

    def synced_cb(self, msg: SyncedNavImu) -> None:
        """Handle synchronized message used by prediction logic.

        Args:
            msg: Time-aligned nav + IMU message from interpolate node.
        """
        self.latest_synced = msg


def main(args=None) -> None:
    """Entrypoint for the prediction node process.

    Args:
        args: Optional ROS command-line arguments.
    """
    rclpy.init(args=args)
    node = PredictionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
