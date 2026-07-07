#!/usr/bin/env python3
"""
Low-jitter real-time logger for synchronized navigation/IMU, RNN correction,
and optional RNN input-window debug data.

Design:
- ROS callbacks only copy message values and enqueue records.
- The optional model-input window log contains the exact flattened
  Float32MultiArray supplied to the recurrent model.
- A background writer thread batches JSONL writes.
- File buffers flush periodically.
- fsync runs periodically for crash/power-loss protection.
- SIGINT and SIGTERM trigger graceful queue draining and a final fsync.

A hard power loss or SIGKILL can still lose up to roughly fsync_interval_s
of the newest data.
"""

import json
import math
import os
import queue
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray, Float64MultiArray

from traj_estimation_msgs.msg import SyncedNavImu


class RealtimeLoggerNode(Node):
    """Subscribe to telemetry topics and write durable JSONL logs."""

    def __init__(self) -> None:
        super().__init__("realtime_logger_node")

        self.declare_parameter("synced_topic", "/ap/state/synced")
        self.declare_parameter("correction_topic", "/ap/state/correction")

        # Produced by PredictionNode when publish_input_window:=true.
        # This is the exact flattened [sequence_length, 13] input tensor
        # supplied to the GRU/LSTM immediately before inference.
        self.declare_parameter(
            "model_input_window_topic",
            "/ap/state/model_input_window",
        )
        self.declare_parameter("log_model_input_window", True)

        self.declare_parameter("output_dir", "/workspace/flight_logs")
        self.declare_parameter("run_name", "")

        # At 150 Hz per topic, 10,000 records provides substantial burst room.
        self.declare_parameter("qos_depth", 1000)
        self.declare_parameter("queue_size", 10000)
        self.declare_parameter("batch_size", 256)

        # Good balance for ~150 Hz logging.
        self.declare_parameter("flush_interval_s", 0.25)
        self.declare_parameter("fsync_interval_s", 1.0)
        self.declare_parameter("shutdown_drain_timeout_s", 10.0)
        self.declare_parameter("stats_period_s", 5.0)

        self.synced_topic = str(self.get_parameter("synced_topic").value)
        self.correction_topic = str(
            self.get_parameter("correction_topic").value
        )
        self.model_input_window_topic = str(
            self.get_parameter("model_input_window_topic").value
        )
        self.log_model_input_window = bool(
            self.get_parameter("log_model_input_window").value
        )

        output_dir = Path(str(self.get_parameter("output_dir").value))
        run_name = str(self.get_parameter("run_name").value)

        qos_depth = int(self.get_parameter("qos_depth").value)
        queue_size = int(self.get_parameter("queue_size").value)
        self.batch_size = int(self.get_parameter("batch_size").value)

        self.flush_interval_s = float(
            self.get_parameter("flush_interval_s").value
        )
        self.fsync_interval_s = float(
            self.get_parameter("fsync_interval_s").value
        )
        self.shutdown_drain_timeout_s = float(
            self.get_parameter("shutdown_drain_timeout_s").value
        )
        stats_period_s = float(self.get_parameter("stats_period_s").value)

        if not run_name:
            run_name = datetime.now().strftime("flight_%Y%m%d_%H%M%S")

        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.flush_interval_s <= 0.0:
            raise ValueError("flush_interval_s must be > 0")
        if self.fsync_interval_s < 0.0:
            raise ValueError("fsync_interval_s must be >= 0")

        self.log_dir = output_dir / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.log_dir / "telemetry.jsonl"
        self.log_file = self.log_path.open(
            "a",
            encoding="utf-8",
            buffering=1024 * 1024,
        )

        self.log_queue: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=queue_size
        )
        self.stop_event = threading.Event()
        self.accepting_records = True

        self.enqueued_count = 0
        self.written_count = 0
        self.dropped_count = 0
        self.write_error_count = 0
        self.last_write_error: Optional[str] = None
        self.last_drop_warning_time = 0.0

        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="telemetry_jsonl_writer",
            daemon=False,
        )
        self.writer_thread.start()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )

        self.create_subscription(
            SyncedNavImu,
            self.synced_topic,
            self.synced_cb,
            qos,
        )

        self.create_subscription(
            Float64MultiArray,
            self.correction_topic,
            self.correction_cb,
            qos,
        )

        if self.log_model_input_window:
            self.create_subscription(
                Float32MultiArray,
                self.model_input_window_topic,
                self.model_input_window_cb,
                qos,
            )

        self.create_timer(stats_period_s, self._report_stats)

        logged_topics = [
            self.synced_topic,
            self.correction_topic,
        ]
        if self.log_model_input_window:
            logged_topics.append(self.model_input_window_topic)

        self.get_logger().info(
            "Logging topics:\n"
            + "".join(f"  {topic}\n" for topic in logged_topics)
            + f"to:\n  {self.log_path}\n"
            + f"queue_size={queue_size}, batch_size={self.batch_size}, "
            + f"flush_interval_s={self.flush_interval_s}, "
            + f"fsync_interval_s={self.fsync_interval_s}"
        )

    @staticmethod
    def _finite_float_or_none(value: Any) -> Optional[float]:
        """Convert numeric values to finite floats; invalid values become null."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        return number if math.isfinite(number) else None

    @staticmethod
    def _header_stamp_dict(message: Any) -> Optional[Dict[str, int]]:
        """Safely extract a ROS Header timestamp when one is available."""
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)

        if stamp is None:
            return None

        sec = int(getattr(stamp, "sec", 0))
        nanosec = int(getattr(stamp, "nanosec", 0))

        return {
            "sec": sec,
            "nanosec": nanosec,
            "ns": sec * 1_000_000_000 + nanosec,
        }

    def _enqueue(self, topic: str, payload: Dict[str, Any]) -> None:
        """Put a complete record into the writer queue without blocking ROS."""
        if not self.accepting_records:
            return

        record = {
            "topic": topic,
            "receipt_time_utc": datetime.now(timezone.utc).isoformat(),
            "receipt_unix_ns": time.time_ns(),
            "receipt_ros_ns": int(self.get_clock().now().nanoseconds),
            "data": payload,
        }

        try:
            self.log_queue.put_nowait(record)
            self.enqueued_count += 1
        except queue.Full:
            self.dropped_count += 1

            now = time.monotonic()
            if now - self.last_drop_warning_time >= 1.0:
                self.last_drop_warning_time = now
                self.get_logger().warn(
                    "Logger queue is full. Dropping records. "
                    f"dropped={self.dropped_count}, "
                    f"queue_size={self.log_queue.maxsize}"
                )

    def synced_cb(self, msg: SyncedNavImu) -> None:
        """Copy synchronized navigation/IMU values and enqueue them."""
        imu = msg.imu

        self._enqueue(
            self.synced_topic,
            {
                "message_stamp": self._header_stamp_dict(msg),
                "imu_stamp": self._header_stamp_dict(imu),
                "latitude": self._finite_float_or_none(msg.latitude),
                "longitude": self._finite_float_or_none(msg.longitude),
                "altitude": self._finite_float_or_none(msg.altitude),
                "orientation": {
                    "x": self._finite_float_or_none(imu.orientation.x),
                    "y": self._finite_float_or_none(imu.orientation.y),
                    "z": self._finite_float_or_none(imu.orientation.z),
                    "w": self._finite_float_or_none(imu.orientation.w),
                },
                "angular_velocity": {
                    "x": self._finite_float_or_none(imu.angular_velocity.x),
                    "y": self._finite_float_or_none(imu.angular_velocity.y),
                    "z": self._finite_float_or_none(imu.angular_velocity.z),
                },
                "linear_acceleration": {
                    "x": self._finite_float_or_none(
                        imu.linear_acceleration.x
                    ),
                    "y": self._finite_float_or_none(
                        imu.linear_acceleration.y
                    ),
                    "z": self._finite_float_or_none(
                        imu.linear_acceleration.z
                    ),
                },
            },
        )

    def correction_cb(self, msg: Float64MultiArray) -> None:
        """Copy RNN correction output and enqueue it."""
        self._enqueue(
            self.correction_topic,
            {
                "values": [
                    self._finite_float_or_none(value)
                    for value in msg.data
                ],
            },
        )

    @staticmethod
    def _multiarray_layout_dict(
        msg: Float32MultiArray,
    ) -> Dict[str, Any]:
        """Copy Float32MultiArray layout metadata into JSON-safe values."""
        return {
            "data_offset": int(msg.layout.data_offset),
            "dimensions": [
                {
                    "label": str(dim.label),
                    "size": int(dim.size),
                    "stride": int(dim.stride),
                }
                for dim in msg.layout.dim
            ],
        }

    @staticmethod
    def _multiarray_shape(
        msg: Float32MultiArray,
    ) -> list[int]:
        """Return the declared shape, such as [30, 13]."""
        return [int(dim.size) for dim in msg.layout.dim]

    def model_input_window_cb(self, msg: Float32MultiArray) -> None:
        """Copy the exact recurrent-model input window and enqueue it.

        ``values`` is flattened in the same row-major order supplied by
        PredictionNode:

            values[0:13]   -> oldest timestep
            values[-13:]   -> newest timestep

        The corresponding tensor passed to the model is conceptually:
        ``[batch=1, time=sequence_length, feature=13]``.  The ROS message
        carries the [time, feature] portion, while the batch dimension is
        implicit and always one.
        """
        self._enqueue(
            self.model_input_window_topic,
            {
                "layout": self._multiarray_layout_dict(msg),
                "shape": self._multiarray_shape(msg),
                "feature_order": [
                    "latitude",
                    "longitude",
                    "altitude",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                    "gx",
                    "gy",
                    "gz",
                    "ax",
                    "ay",
                    "az",
                ],
                "values": [
                    self._finite_float_or_none(value)
                    for value in msg.data
                ],
            },
        )

    def _write_batch(self, batch: list[Dict[str, Any]]) -> None:
        """Serialize and write records from the background writer thread."""
        if not batch:
            return

        lines = "".join(
            json.dumps(
                record,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for record in batch
        )

        self.log_file.write(lines)
        self.written_count += len(batch)

    def _flush_and_sync(self, force_sync: bool) -> None:
        """Flush Python/OS buffers and optionally force data to disk."""
        self.log_file.flush()

        if force_sync:
            os.fsync(self.log_file.fileno())

    def _writer_loop(self) -> None:
        """Continuously batch queued records and write them to JSONL."""
        batch: list[Dict[str, Any]] = []

        last_flush_time = time.monotonic()
        last_fsync_time = last_flush_time

        try:
            while not self.stop_event.is_set() or not self.log_queue.empty():
                try:
                    record = self.log_queue.get(timeout=0.05)
                    batch.append(record)

                    while len(batch) < self.batch_size:
                        try:
                            batch.append(self.log_queue.get_nowait())
                        except queue.Empty:
                            break

                except queue.Empty:
                    pass

                now = time.monotonic()

                flush_due = (
                    len(batch) >= self.batch_size
                    or now - last_flush_time >= self.flush_interval_s
                    or (
                        self.stop_event.is_set()
                        and self.log_queue.empty()
                        and len(batch) > 0
                    )
                )

                if flush_due and batch:
                    try:
                        self._write_batch(batch)
                        batch.clear()
                        self._flush_and_sync(force_sync=False)
                        last_flush_time = now
                    except OSError as exc:
                        self.write_error_count += len(batch)
                        self.last_write_error = str(exc)
                        batch.clear()

                fsync_due = (
                    self.fsync_interval_s > 0.0
                    and now - last_fsync_time >= self.fsync_interval_s
                )

                if fsync_due:
                    try:
                        if batch:
                            self._write_batch(batch)
                            batch.clear()

                        self._flush_and_sync(force_sync=True)
                        last_flush_time = now
                        last_fsync_time = now
                    except OSError as exc:
                        self.last_write_error = str(exc)
                        self.write_error_count += 1

        finally:
            # Graceful shutdown: write every record still in memory, then fsync.
            try:
                while True:
                    try:
                        batch.append(self.log_queue.get_nowait())
                    except queue.Empty:
                        break

                    if len(batch) >= self.batch_size:
                        self._write_batch(batch)
                        batch.clear()

                if batch:
                    self._write_batch(batch)

                self._flush_and_sync(force_sync=True)

            except OSError as exc:
                self.last_write_error = str(exc)
                self.write_error_count += 1

    def _report_stats(self) -> None:
        """Periodically report logger health."""
        queue_depth = self.log_queue.qsize()

        self.get_logger().info(
            "Logger status: "
            f"queued={queue_depth}, "
            f"enqueued={self.enqueued_count}, "
            f"written={self.written_count}, "
            f"dropped={self.dropped_count}, "
            f"write_errors={self.write_error_count}"
        )

        if self.last_write_error is not None:
            self.get_logger().error(
                f"Most recent logger write error: {self.last_write_error}"
            )

    def destroy_node(self) -> bool:
        """Stop callbacks, drain queued data, and close the log file."""
        self.accepting_records = False
        self.stop_event.set()

        if hasattr(self, "writer_thread") and self.writer_thread.is_alive():
            self.writer_thread.join(
                timeout=self.shutdown_drain_timeout_s
            )

            if self.writer_thread.is_alive():
                self.get_logger().error(
                    "Logger writer did not finish before shutdown timeout. "
                    "Some queued records may not have been written."
                )

        if hasattr(self, "log_file") and not self.log_file.closed:
            try:
                self.log_file.flush()
                os.fsync(self.log_file.fileno())
                self.log_file.close()
            except OSError as exc:
                self.get_logger().error(
                    f"Could not close log file cleanly: {exc}"
                )

        return super().destroy_node()


def _shutdown_signal_handler(signum, _frame) -> None:
    """Convert Ctrl+C or Docker SIGTERM into graceful Python shutdown."""
    raise KeyboardInterrupt


def main(args=None) -> None:
    signal.signal(signal.SIGINT, _shutdown_signal_handler)
    signal.signal(signal.SIGTERM, _shutdown_signal_handler)

    rclpy.init(args=args)
    node = RealtimeLoggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()