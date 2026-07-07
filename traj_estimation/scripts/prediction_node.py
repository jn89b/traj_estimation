#!/usr/bin/env python3
"""Run recurrent correction-model inference on a fixed-rate synced topic.

Optionally publishes the exact rolling feature window sent to the GRU/LSTM.

Published input window layout:
    shape = [sequence_length, 13]

Feature order per timestep:
    [lat, lon, alt, qx, qy, qz, qw, gx, gy, gz, ax, ay, az]
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import (
    Float32MultiArray,
    Float64MultiArray,
    MultiArrayDimension,
)
import torch

from traj_estimation.corrector_simple import (
    make_gru_correction_model,
    make_lstm_correction_model,
)
from traj_estimation_msgs.msg import SyncedNavImu


NSEC_PER_SEC = 1_000_000_000
FEATURE_DIM = 13


class PredictionNode(Node):
    """Run model inference over a rolling fixed-rate nav/IMU window."""

    def __init__(self) -> None:
        super().__init__("prediction_node")

        # Input/output topics.
        self.declare_parameter("synced_topic", "/ap/state/synced")
        self.declare_parameter("correction_topic", "/ap/state/correction")

        # Model configuration.
        self.declare_parameter("sequence_length", 30)
        self.declare_parameter("model_type", "lstm")
        self.declare_parameter("model_checkpoint", "")
        self.declare_parameter("device", "cuda")

        # Timing/QoS configuration.
        self.declare_parameter("qos_depth", 300)
        self.declare_parameter("expected_input_rate_hz", 150.0)
        self.declare_parameter("max_timing_error_fraction", 0.20)

        # Optional debug output for the actual inference window.
        self.declare_parameter("publish_input_window", True)
        self.declare_parameter(
            "input_window_topic",
            "/ap/state/model_input_window",
        )
        self.declare_parameter("window_publish_every_n", 1)

        synced_topic = str(self.get_parameter("synced_topic").value)
        correction_topic = str(self.get_parameter("correction_topic").value)

        self.sequence_length = int(
            self.get_parameter("sequence_length").value
        )
        model_type = str(self.get_parameter("model_type").value).lower()
        model_checkpoint = str(
            self.get_parameter("model_checkpoint").value
        )
        requested_device = str(self.get_parameter("device").value)

        qos_depth = int(self.get_parameter("qos_depth").value)
        self.expected_input_rate_hz = float(
            self.get_parameter("expected_input_rate_hz").value
        )
        self.max_timing_error_fraction = float(
            self.get_parameter("max_timing_error_fraction").value
        )

        self.publish_input_window = bool(
            self.get_parameter("publish_input_window").value
        )
        input_window_topic = str(
            self.get_parameter("input_window_topic").value
        )
        self.window_publish_every_n = int(
            self.get_parameter("window_publish_every_n").value
        )

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")

        if self.expected_input_rate_hz <= 0.0:
            raise ValueError("expected_input_rate_hz must be > 0")

        if not 0.0 < self.max_timing_error_fraction < 1.0:
            raise ValueError("max_timing_error_fraction must be in (0, 1)")

        if self.window_publish_every_n < 1:
            raise ValueError("window_publish_every_n must be >= 1")

        self.expected_dt_s = 1.0 / self.expected_input_rate_hz
        self.last_stamp_ns: Optional[int] = None
        self.inference_count = 0

        # Every entry is one 13-value feature row.
        self.feature_buffer: Deque[List[float]] = deque(
            maxlen=self.sequence_length
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )

        if requested_device == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn(
                "CUDA requested but unavailable. Falling back to CPU."
            )
            requested_device = "cpu"

        self.device = torch.device(requested_device)

        if model_type == "gru":
            self.model = make_gru_correction_model(input_dim=FEATURE_DIM)
        elif model_type == "lstm":
            self.model = make_lstm_correction_model(input_dim=FEATURE_DIM)
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")

        if model_checkpoint:
            ckpt_path = Path(model_checkpoint)

            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"model_checkpoint not found: {ckpt_path}"
                )

            state = torch.load(
                str(ckpt_path),
                map_location=self.device,
            )

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.correction_pub = self.create_publisher(
            Float64MultiArray,
            correction_topic,
            qos,
        )

        self.input_window_pub = None
        if self.publish_input_window:
            self.input_window_pub = self.create_publisher(
                Float32MultiArray,
                input_window_topic,
                qos,
            )

        self.create_subscription(
            SyncedNavImu,
            synced_topic,
            self.synced_cb,
            qos,
        )

        window_duration_s = (
            self.sequence_length / self.expected_input_rate_hz
        )

        self.get_logger().info(
            f"Listening on {synced_topic}; "
            f"model={model_type}, "
            f"window={self.sequence_length} samples "
            f"({window_duration_s:.3f} s), "
            f"expected_rate={self.expected_input_rate_hz:.3f} Hz, "
            f"device={self.device.type}, "
            f"publishing corrections to {correction_topic}"
        )

        if self.publish_input_window:
            self.get_logger().info(
                f"Publishing model input windows to {input_window_topic} "
                f"every {self.window_publish_every_n} inference(s)."
            )

    @staticmethod
    def _stamp_to_ns(msg: SyncedNavImu) -> int:
        """Convert ROS header timestamp to integer nanoseconds."""
        return (
            int(msg.header.stamp.sec) * NSEC_PER_SEC
            + int(msg.header.stamp.nanosec)
        )

    def _message_to_feature(self, msg: SyncedNavImu) -> List[float]:
        """Build the 13-element feature vector used by the trained model.

        Feature order:
        [lat, lon, alt, qx, qy, qz, qw, gx, gy, gz, ax, ay, az]
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

    def _timing_is_valid(self, stamp_ns: int) -> bool:
        """Clear the sequence buffer if input timing is discontinuous."""
        if stamp_ns <= 0:
            self.get_logger().warn(
                "Dropping synchronized sample with zero timestamp."
            )
            return False

        if self.last_stamp_ns is None:
            self.last_stamp_ns = stamp_ns
            return True

        dt_s = (stamp_ns - self.last_stamp_ns) / float(NSEC_PER_SEC)
        self.last_stamp_ns = stamp_ns

        max_error_s = (
            self.expected_dt_s * self.max_timing_error_fraction
        )

        if dt_s <= 0.0 or abs(dt_s - self.expected_dt_s) > max_error_s:
            self.feature_buffer.clear()

            self.get_logger().warn(
                f"Input timing discontinuity: dt={dt_s:.6f} s, "
                f"expected={self.expected_dt_s:.6f} s. "
                "Cleared model sequence buffer."
            )

        return True

    def _publish_input_window(self) -> None:
        """Publish the exact [sequence_length, 13] input sent to the model."""
        if self.input_window_pub is None:
            return

        if len(self.feature_buffer) < self.sequence_length:
            return

        self.inference_count += 1

        if self.inference_count % self.window_publish_every_n != 0:
            return

        window_msg = Float32MultiArray()

        # The flattened data is row-major:
        # first 13 values = oldest feature sample
        # last 13 values = newest feature sample
        flattened_window = [
            value
            for feature_row in self.feature_buffer
            for value in feature_row
        ]

        window_msg.layout.dim = [
            MultiArrayDimension(
                label="time",
                size=self.sequence_length,
                stride=self.sequence_length * FEATURE_DIM,
            ),
            MultiArrayDimension(
                label="feature",
                size=FEATURE_DIM,
                stride=FEATURE_DIM,
            ),
        ]

        window_msg.layout.data_offset = 0
        window_msg.data = flattened_window

        self.input_window_pub.publish(window_msg)

    def _run_inference(self) -> Optional[List[float]]:
        """Run one forward pass after the rolling input window is full."""
        if len(self.feature_buffer) < self.sequence_length:
            return None

        # This is the exact [1, sequence_length, 13] tensor.
        sequence = torch.tensor(
            [list(self.feature_buffer)],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            correction = self.model(sequence)

        return correction[0].detach().cpu().tolist()

    def synced_cb(self, msg: SyncedNavImu) -> None:
        """Append a sample, publish the model-input window, then infer."""
        stamp_ns = self._stamp_to_ns(msg)

        if not self._timing_is_valid(stamp_ns):
            return

        self.feature_buffer.append(self._message_to_feature(msg))

        if len(self.feature_buffer) < self.sequence_length:
            return

        # Publish the same rolling values about to go into the model.
        self._publish_input_window()

        correction = self._run_inference()
        if correction is None:
            return

        correction_msg = Float64MultiArray()
        correction_msg.data = correction
        self.correction_pub.publish(correction_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PredictionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()