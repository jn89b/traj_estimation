#!/usr/bin/env python3
"""Launch one interpolator plus multiple recurrent inference/logging pipelines.

Topology
--------
interpolate_node.py
    └── /ap/state/synced
          ├── prediction_<model_id>
          │     ├── /ap/state/models/<model_id>/correction
          │     └── /ap/state/models/<model_id>/input_window
          └── logger_<model_id>
                └── <log_root>/<run_name>/<model_id>/telemetry.jsonl

Each prediction node has its own sequence length and checkpoint.  Each logger
receives the same synchronized source telemetry but only the correction and
input-window topics for its matching model.

Before use, edit MODEL_CONFIGS below so each checkpoint path and window length
match the model you trained.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


PACKAGE_NAME = "traj_estimation"

# ---------------------------------------------------------------------------
# EDIT THIS LIST TO ADD/REMOVE MODEL PIPELINES.
#
# - model_id must be unique. It is used in ROS node/topic names and the log
#   directory name, so use letters, numbers, and underscores only.
# - sequence_length must match the sequence/window length used in training.
# - model_checkpoint must point to the matching .pt/.pth checkpoint available
#   inside the same environment/container that launches ROS 2.
# - window_publish_every_n controls debug-window logging only. Inference still
#   runs on every synced message once its buffer is full.
# ---------------------------------------------------------------------------
MODEL_CONFIGS: List[Dict[str, Any]] = [
    {
        "model_id": "lstm_w30",
        "model_type": "lstm",
        "sequence_length": 30,
        "model_checkpoint": "", #"/workspace/models/lstm_w30.pt",
        "window_publish_every_n": 1,
    },
    {
        "model_id": "lstm_w60",
        "model_type": "lstm",
        "sequence_length": 60,
        "model_checkpoint": "", #"/workspace/models/lstm_w60.pt",
        "window_publish_every_n": 1,
    },
]


def _validate_model_configs() -> None:
    """Fail early when the static model configuration is malformed."""
    seen_model_ids = set()

    for config in MODEL_CONFIGS:
        required_keys = {
            "model_id",
            "model_type",
            "sequence_length",
            "model_checkpoint",
        }
        missing = required_keys.difference(config)
        if missing:
            raise ValueError(
                f"Model config is missing required key(s): {sorted(missing)}"
            )

        model_id = str(config["model_id"])
        if not model_id:
            raise ValueError("model_id cannot be empty")
        if model_id in seen_model_ids:
            raise ValueError(f"Duplicate model_id: {model_id}")
        seen_model_ids.add(model_id)

        model_type = str(config["model_type"]).lower()
        if model_type not in {"lstm", "gru"}:
            raise ValueError(
                f"model_type for {model_id!r} must be 'lstm' or 'gru'"
            )

        if int(config["sequence_length"]) < 1:
            raise ValueError(
                f"sequence_length for {model_id!r} must be >= 1"
            )

        if int(config.get("window_publish_every_n", 1)) < 1:
            raise ValueError(
                f"window_publish_every_n for {model_id!r} must be >= 1"
            )


def generate_launch_description() -> LaunchDescription:
    _validate_model_configs()

    default_run_name = datetime.now().strftime("inference_%Y%m%d_%H%M%S")

    # Shared launch-time configuration.
    synced_topic = LaunchConfiguration("synced_topic")
    device = LaunchConfiguration("device")
    expected_input_rate_hz = LaunchConfiguration("expected_input_rate_hz")
    max_timing_error_fraction = LaunchConfiguration(
        "max_timing_error_fraction"
    )
    prediction_qos_depth = LaunchConfiguration("prediction_qos_depth")
    logger_qos_depth = LaunchConfiguration("logger_qos_depth")
    logger_queue_size = LaunchConfiguration("logger_queue_size")
    logger_batch_size = LaunchConfiguration("logger_batch_size")
    log_root = LaunchConfiguration("log_root")
    run_name = LaunchConfiguration("run_name")

    actions = [
        DeclareLaunchArgument(
            "start_interpolator",
            default_value="true",
            description=(
                "Start interpolate_node.py. Set false when another process "
                "already publishes the synced topic."
            ),
        ),
        DeclareLaunchArgument(
            "synced_topic",
            default_value="/ap/state/synced",
            description="Shared SyncedNavImu topic consumed by all models.",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Torch device for all prediction nodes: cuda or cpu.",
        ),
        DeclareLaunchArgument(
            "expected_input_rate_hz",
            default_value="150.0",
            description="Expected fixed rate of the synchronized input topic.",
        ),
        DeclareLaunchArgument(
            "max_timing_error_fraction",
            default_value="0.20",
            description="Allowed fractional timing deviation before reset.",
        ),
        DeclareLaunchArgument(
            "prediction_qos_depth",
            default_value="300",
            description="QoS history depth for each prediction node.",
        ),
        DeclareLaunchArgument(
            "logger_qos_depth",
            default_value="1000",
            description="QoS history depth for each logger node.",
        ),
        DeclareLaunchArgument(
            "logger_queue_size",
            default_value="10000",
            description="In-memory JSONL queue size for each logger.",
        ),
        DeclareLaunchArgument(
            "logger_batch_size",
            default_value="256",
            description="JSONL records written per logger batch.",
        ),
        DeclareLaunchArgument(
            "log_root",
            default_value="/workspace/flight_logs",
            description="Parent directory for all model-specific log folders.",
        ),
        DeclareLaunchArgument(
            "run_name",
            default_value=default_run_name,
            description="Unique parent folder name for this launch run.",
        ),
        # These defaults match the filenames in your scripts/ directory.
        # Override them at launch only if your installed ROS 2 executable names
        # omit the .py suffix (for example: prediction_node).
        DeclareLaunchArgument(
            "interpolator_executable",
            default_value="interpolate_node.py",
            description="Installed ROS 2 executable for the interpolator.",
        ),
        DeclareLaunchArgument(
            "prediction_executable",
            default_value="prediction_node.py",
            description="Installed ROS 2 executable for each model runner.",
        ),
        DeclareLaunchArgument(
            "logger_executable",
            default_value="logger_node.py",
            description="Installed ROS 2 executable for each JSONL logger.",
        ),
        LogInfo(
            msg=(
                "Starting one interpolator and "
                f"{len(MODEL_CONFIGS)} model/logger pipeline(s)."
            )
        ),
    ]

    # This assumes your interpolate node's default output is /ap/state/synced.
    # If your interpolate_node.py uses a different parameter name for that
    # output topic, add that parameter here using the name it declares.
    actions.append(
        Node(
            package=PACKAGE_NAME,
            executable=LaunchConfiguration("interpolator_executable"),
            name="interpolate_node",
            output="screen",
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration("start_interpolator")),
        )
    )

    for config in MODEL_CONFIGS:
        model_id = str(config["model_id"])
        model_type = str(config["model_type"]).lower()
        sequence_length = int(config["sequence_length"])
        model_checkpoint = str(config["model_checkpoint"])
        window_publish_every_n = int(
            config.get("window_publish_every_n", 15)
        )

        # These are intentionally different for every model. This prevents
        # corrections/windows from different checkpoints being mixed together.
        correction_topic = (
            f"/ap/state/models/{model_id}/correction"
        )
        input_window_topic = (
            f"/ap/state/models/{model_id}/input_window"
        )

        # LoggerNode appends its run_name to output_dir. With the values below,
        # each pipeline writes to:
        #   <log_root>/<launch_run_name>/<model_id>/telemetry.jsonl
        model_log_parent = PathJoinSubstitution([log_root, run_name])

        actions.append(
            Node(
                package=PACKAGE_NAME,
                executable=LaunchConfiguration("prediction_executable"),
                name=f"prediction_{model_id}",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "synced_topic": synced_topic,
                        "qos_depth": prediction_qos_depth,
                        "sequence_length": sequence_length,
                        "model_type": model_type,
                        "model_checkpoint": model_checkpoint,
                        "device": device,
                        "correction_topic": correction_topic,
                        "expected_input_rate_hz": expected_input_rate_hz,
                        "max_timing_error_fraction": (
                            max_timing_error_fraction
                        ),
                        "publish_input_window": True,
                        "input_window_topic": input_window_topic,
                        "window_publish_every_n": window_publish_every_n,
                    }
                ],
            )
        )

        actions.append(
            Node(
                package=PACKAGE_NAME,
                executable=LaunchConfiguration("logger_executable"),
                name=f"logger_{model_id}",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        # Shared telemetry source used by this model.
                        "synced_topic": synced_topic,
                        # Only this model's outputs are logged by this logger.
                        "correction_topic": correction_topic,
                        "model_input_window_topic": input_window_topic,
                        "log_model_input_window": True,
                        # Directory structure: log_root/run_name/model_id/...
                        "output_dir": model_log_parent,
                        "run_name": model_id,
                        "qos_depth": logger_qos_depth,
                        "queue_size": logger_queue_size,
                        "batch_size": logger_batch_size,
                    }
                ],
            )
        )

    return LaunchDescription(actions)
