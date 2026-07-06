#!/usr/bin/env python3
"""Consume synchronized navigation + IMU samples and run model inference.

This node subscribes to a fused topic that already contains time-aligned
latitude/longitude/altitude and IMU data, converts each sample into a feature
vector, feeds a rolling window into the correction network, and publishes the
predicted correction output.
"""

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray
import torch
from traj_estimation.corrector_simple import (
    make_gru_correction_model,
    make_lstm_correction_model,
)
from traj_estimation_msgs.msg import SyncedNavImu


class PredictionNode(Node):
    """Run ML inference on synchronized navigation and IMU samples."""

    def __init__(self) -> None:
        """Initialize model, subscription, and correction publisher.

        Input parameters (ROS params):
            synced_topic (str): Topic containing synchronized nav and IMU.
            qos_depth (int): QoS queue depth used for pub/sub.
            sequence_length (int): Number of timesteps per inference window.
            model_type (str): Recurrent model type, ``lstm`` or ``gru``.
            model_checkpoint (str): Optional checkpoint path for model weights.
            device (str): Inference device, e.g. ``cpu`` or ``cuda``.
            correction_topic (str): Topic for model correction outputs.
        """
        super().__init__('prediction_node')

        self.declare_parameter('synced_topic', '/ap/state/synced')
        self.declare_parameter('qos_depth', 100)
        self.declare_parameter('sequence_length', 30)
        self.declare_parameter('model_type', 'lstm')
        self.declare_parameter('model_checkpoint', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('correction_topic', '/ap/state/correction')

        synced_topic = str(self.get_parameter('synced_topic').value)
        qos_depth = int(self.get_parameter('qos_depth').value)
        self.sequence_length = int(self.get_parameter('sequence_length').value)
        model_type = str(self.get_parameter('model_type').value).lower()
        model_checkpoint = str(self.get_parameter('model_checkpoint').value)
        requested_device = str(self.get_parameter('device').value)
        correction_topic = str(self.get_parameter('correction_topic').value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )

        if self.sequence_length < 1:
            raise ValueError('sequence_length must be >= 1')

        self.latest_synced: Optional[SyncedNavImu] = None
        self.feature_buffer: Deque[List[float]] = deque(maxlen=self.sequence_length)

        if requested_device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn(
                'CUDA requested but unavailable. Falling back to CPU.'
            )
            requested_device = 'cpu'
        self.device = torch.device(requested_device)

        if model_type == 'gru':
            self.model = make_gru_correction_model(input_dim=13)
        else:
            self.model = make_lstm_correction_model(input_dim=13)

        if model_checkpoint:
            ckpt_path = Path(model_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f'model_checkpoint not found: {ckpt_path}'
                )
            state = torch.load(str(ckpt_path), map_location=self.device)
            # Support both raw state_dict and wrapped checkpoints.
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.correction_pub = self.create_publisher(
            Float64MultiArray,
            correction_topic,
            qos,
        )

        self.create_subscription(
            SyncedNavImu,
            synced_topic,
            self.synced_cb,
            qos,
        )

        self.get_logger().info(
            f'Listening on {synced_topic}; model={model_type}, '
            f'seq_len={self.sequence_length}, device={self.device.type}, '
            f'publishing corrections to {correction_topic}'
        )

    def _message_to_feature(self, msg: SyncedNavImu) -> List[float]:
        """Convert SyncedNavImu into one model feature vector.

        Args:
            msg: Time-aligned nav + IMU message.

        Returns:
            Feature vector ordered as:
            ``[lat, lon, alt, qx, qy, qz, qw, gx, gy, gz, ax, ay, az]``.
        """
        imu = msg.imu
        return [
            msg.latitude,
            msg.longitude,
            msg.altitude,
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w,
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
        ]

    def _run_inference(self) -> Optional[List[float]]:
        """Run one forward pass if the sequence buffer is full.

        Returns:
            A list of 6 correction values, or ``None`` if not enough data.
        """
        if len(self.feature_buffer) < self.sequence_length:
            return None

        sequence = torch.tensor(
            [list(self.feature_buffer)],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            correction = self.model(sequence)
        return correction[0].detach().cpu().tolist()

    def synced_cb(self, msg: SyncedNavImu) -> None:
        """Handle synchronized message and trigger inference.

        Args:
            msg: Time-aligned nav + IMU message from interpolate node.
        """
        self.latest_synced = msg
        # The feature buffer is a rolling window of the last N samples, where N is
        # the sequence length, the dequeu automatically discards the oldest sample when a new one is added.
        self.feature_buffer.append(self._message_to_feature(msg))

        correction = self._run_inference()
        if correction is None:
            return

        out = Float64MultiArray()
        out.data = correction
        self.correction_pub.publish(out)


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
