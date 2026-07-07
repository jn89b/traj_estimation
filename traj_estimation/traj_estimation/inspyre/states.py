"""Provides the various state classes for managing the state of the flight body.

Sandia Proprietary
"""

from collections.abc import Iterable

import numpy as np
import quaternion

from traj_estimation.inspyre import earth, rotations
from traj_estimation.inspyre.transformations import quat_to_rpy, rpy_to_quat


class StateIndices:
    """Positions of the various attributes within the state vector.

    Notes
    -----
    This class is effectively a custom rolled enum to handle slices.
    """

    POSITION = slice(0, 3, 1)
    VELOCITY = slice(3, 6, 1)
    ATTITUDE = slice(6, 9, 1)
    ACCEL_SCALE = slice(9, 12, 1)
    ACCEL_BIAS = slice(12, 15, 1)
    GYRO_SCALE = slice(15, 18, 1)
    GYRO_BIAS = slice(18, 21, 1)
    POSITION_X = 0
    POSITION_Y = 1
    POSITION_Z = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    VELOCITY_Z = 5
    ATTITUDE_ROLL = 6
    ATTITUDE_PITCH = 7
    ATTITUDE_YAW = 8

    @classmethod
    def __getitem__(cls, item):
        """Get the value corresponding to the name."""
        return cls.__dict__[item]


class VehicleState:
    """State variables for the flight body.

    Parameters
    ----------
    position : np.ndarray
        Initial position in the ECEF frame
    velocity : np.ndarray
        Initial velocity in the ECEF frame
    attitude : np.ndarray | quaternion.quaternion
        Initial attitude of the vehicle (roll, pitch, yaw)
    time : float, optional
        Current time, default is 0
    accel_scale_factor : Iterable[float], optional
        accel = scale_factor * meas + bias
    accel_bias : Iterable[float], optional
        accel = scale_factor * meas + bias
    gyro_scale_factor : Iterable[float], optional
        gyro = scale_factor * meas + bias
    gyro_bias : Iterable[float], optional
        gyro = scale_factor * meas + bias

    Notes
    -----
    State vector and corresponding EKF matrices are indexed as follows:
        position in [:3]
        velocity in [3:6]
        rotation in [6:9]
        accel scale factor in [9:12]
        accel bias in [12:15]
        gyro scale factor in [15:18]
        gyro bias in [18:]
    """

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        attitude: np.ndarray | quaternion.quaternion,
        time: float = 0.0,
        accel_scale_factor: Iterable[float] = (0.0, 0.0, 0.0),
        accel_bias: Iterable[float] = (0.0, 0.0, 0.0),
        gyro_scale_factor: Iterable[float] = (0.0, 0.0, 0.0),
        gyro_bias: Iterable[float] = (0.0, 0.0, 0.0),
    ):
        self.r_e_eb = position
        self.v_e_eb = velocity
        try:
            q_nb = rpy_to_quat(attitude)
            lat, lon, _ = earth.ecef_to_ned(self.r_e_eb)
            q_en = quaternion.from_rotation_matrix(
                rotations.local_nav_to_ecef(lat, lon)
            )
            self.q_eb = q_en * q_nb  # Order matters here
        except IndexError:
            self.q_eb = attitude.normalized()
        self.time = time
        self.accel_scale_factor = np.asarray(accel_scale_factor)
        self.accel_bias = np.asarray(accel_bias)
        self.gyro_scale_factor = np.asarray(gyro_scale_factor)
        self.gyro_bias = np.asarray(gyro_bias)

    @property
    def q_nb(self) -> quaternion.quaternion:
        """Compute the body -> local navigation rotation quaternion."""
        # Convert quaternion to be relative to local navigation plane
        lat, lon, _ = self.position
        q_ne = quaternion.from_rotation_matrix(rotations.ecef_to_local_nav(lat, lon))
        return q_ne * self.q_eb

    @property
    def rpy(self) -> np.ndarray:
        """Compute the Roll, Pitch, and Yaw of the body in radians."""
        q_nb = self.q_nb
        return quat_to_rpy(q_nb)

    @property
    def position(self) -> np.ndarray:
        """Compute the lat, lon, and altitude of the body."""
        return earth.ecef_to_ned(self.r_e_eb)


class EKFState:
    """State variables for the Kalman Filter.

    Parameters
    ----------
    vehicle_state : VehicleState
        The state describing the flight body and its IMU
    state_covariance : np.ndarray
        State covariance matrix
    """

    def __init__(self, vehicle_state: VehicleState, state_covariance: np.ndarray):
        self.state = vehicle_state
        self.state_covariance = state_covariance


class Pose:
    """Position and attitude of a vehicle.

    Parameters
    ----------
    lla : np.ndarray | list[float]
        Lat, lon, altitude - Provided in degrees
    attitude : np.ndarray | quaternion.quaternion
        Either roll, pitch, yaw or equivalent quaternion
    time : float
        Current time in seconds
    radians : bool
        Whether lat/lon is passed as radians (True) or degrees (False, default)

    Notes
    -----
    The attitude quaternion is assumed to be in a local navigation frame
    """

    def __init__(
        self,
        lla: np.ndarray | list[float],
        attitude: np.ndarray | quaternion.quaternion,
        time: float,
        radians: bool = False,
    ):
        if radians:
            self.lat = lla[0]
            self.lon = lla[1]
        else:
            self.lat = np.deg2rad(lla[0])
            self.lon = np.deg2rad(lla[1])

        self.alt = lla[2]
        try:
            q_nb = rpy_to_quat(attitude)
        except IndexError:
            q_nb = attitude.normalized()
        q_en = quaternion.from_rotation_matrix(
            rotations.local_nav_to_ecef(self.lat, self.lon)
        )
        self.q_eb = q_en * q_nb  # Order matters here
        self.time = time

    @property
    def q_nb(self) -> quaternion.quaternion:
        """Compute the body -> local navigation rotation quaternion."""
        q_ne = quaternion.from_rotation_matrix(
            rotations.ecef_to_local_nav(self.lat, self.lon)
        )
        return q_ne * self.q_eb

    @property
    def r_e_eb(self) -> np.ndarray:
        """Compute the vehicle's location in ECEF frame."""
        return earth.ned_to_ecef(self.lat, self.lon, self.alt)

    @classmethod
    def new(cls, ahrs):
        """Convert a row of ahrs data into a Pose."""
        time = ahrs[1] / 1e6
        alt, lat, lon = ahrs[5:8]
        q_nb = quaternion.as_quat_array(ahrs[8:12])
        return cls(np.array([lat, lon, alt]), q_nb, time)
