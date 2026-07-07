"""Provide various coordinate frame and rotation transformations.

Sandia Proprietary
"""

import numpy as np
import quaternion


def rpy_to_quat(rpy: np.ndarray) -> quaternion.quaternion:
    """Convert roll, pitch, and yaw vector to the corresponding quaternion.

    Parameters
    ----------
    rpy : np.ndarray
        [roll, pitch, yaw] in radians

    Returns
    -------
    quaternion.quaternion :
        Quaternion corresponding to q_eb (orientation of the body relative to the ECEF frame)
    """
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]
    w = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    x = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    y = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    z = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    return quaternion.quaternion(w, x, y, z)


def quat_to_rpy(quat: quaternion.quaternion) -> np.ndarray:
    """Convert quaternion to the corresponding roll, pitch, yaw vector in radians.

    Parameters
    ----------
    quat : quaternion.quaternion
        Quaternion representing current attitude (orientation of body relative to ECEF frame)

    Returns
    -------
    np.ndarray :
        Roll, pitch, yaw in radians
    """
    w = quat.w
    x = quat.x
    y = quat.y
    z = quat.z
    roll = np.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.asin(2 * (w * y - z * x))
    yaw = np.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([roll, pitch, yaw])


def skew_symmetric(rotation_vec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to a skew symmetric matrix.

    Parameters
    ----------
    rotation_vec : np.ndarray
        Rotation vector to be converted, length 3

    Returns
    -------
    np.ndarray :
        3x3 skew symmetric matrix
    """
    return np.array(
        [
            [0, -rotation_vec[2], rotation_vec[1]],
            [rotation_vec[2], 0, -rotation_vec[0]],
            [-rotation_vec[1], rotation_vec[0], 0],
        ]
    )
