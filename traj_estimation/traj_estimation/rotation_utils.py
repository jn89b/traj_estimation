from typing import Tuple

import numpy as np
# import casadi as ca
import math

def rot2d(psi):
    return np.array([[np.cos(psi), -np.sin(psi)],
                     [np.sin(psi), np.cos(psi)]])


def rot2d_casadi(psi):
    return ca.vertcat(
        ca.horzcat(ca.cos(psi), -ca.sin(psi)),
        ca.horzcat(ca.sin(psi), ca.cos(psi))
    )

#rotation 3d
def rot3d(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_x, np.dot(R_y, R_z))
    return R

def rot3d_casadi(roll, pitch, yaw):
    R_x = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
        ca.horzcat(0, ca.sin(roll), ca.cos(roll))
    )

    R_y = ca.vertcat(
        ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch))
    )

    R_z = ca.vertcat(
        ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
        ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
        ca.horzcat(0, 0, 1)
    )

    R = ca.mtimes(R_z, ca.mtimes(R_y, R_x))
    return R


def euler_from_quaternion(x:float, y:float, z:float, w:float) -> tuple:
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


def vector_euler_from_quaternion(x:np.ndarray, y:np.ndarray, z:np.ndarray, w:np.ndarray):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


def get_quaternion_from_euler(roll:float, pitch:float, yaw:float) -> list:
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """

  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

  return [qx, qy, qz, qw]


def convertENUToNED(x_enu:float, y_enu:float, z_enu:float) -> list:
    """converts from ENU to NED"""
    ned =  np.zeros(3, dtype=np.float64)
    ned[0] = y_enu
    ned[1] = x_enu
    ned[2] = -z_enu
    return ned

def convertNEDToENU(x_ned:float, y_ned:float, z_ned:float) -> list:
    """converts from NED to ENU"""
    #create 3,1 array
    enu = np.zeros(3, dtype=np.float64)
    enu[0] = y_ned
    enu[1] = x_ned
    enu[2] = -z_ned
    return enu

def convertNEDtoENUVector(x_ned:np.ndarray, y_ned:np.ndarray, z_ned:np.ndarray) -> list:
    """converts from NED to ENU"""
    enu_x = y_ned
    enu_y = x_ned
    enu_z = -z_ned
    return enu_x, enu_y, enu_z

def convertENUToNEDVector(x_enu:np.ndarray, y_enu:np.ndarray, z_enu:np.ndarray) -> list:
    """converts from ENU to NED"""
    ned_x = y_enu
    ned_y = x_enu
    ned_z = -z_enu
    return ned_x, ned_y, ned_z  

def normalize_quat(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length quaternion")
    return (x / norm, y / norm, z / norm, w / norm)

def geopose_enu_flu_to_ned_frd(
    q_enu_flu: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Convert ArduPilot DDS /ap/geopose/filtered orientation back to
    ArduPilot-native NED world frame / FRD body frame.

    Input and output use ROS Quaternion field order:
        (x, y, z, w)
    """
    x_enu, y_enu, z_enu, w_enu = normalize_quat(q_enu_flu)

    s = math.sqrt(0.5)

    # Exact inverse of AP_DDS_Client.cpp GeoPose conversion.
    q_ned_frd = (
        s * (x_enu + y_enu),  # x
        s * (x_enu - y_enu),  # y
        s * (w_enu - z_enu),  # z
        s * (w_enu + z_enu),  # w
    )

    return normalize_quat(q_ned_frd)


def slerp_quat(
    q0: Tuple[float, float, float, float],
    q1: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    x0, y0, z0, w0 = normalize_quat(q0)
    x1, y1, z1, w1 = normalize_quat(q1)

    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1

    if dot < 0.0:
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1
        dot = -dot

    if dot > 0.9995:
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)
        w = w0 + t * (w1 - w0)
        return normalize_quat((x, y, z, w))

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)

    if abs(sin_theta_0) < 1e-12:
        return (x0, y0, z0, w0)

    theta = theta_0 * t
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    x = s0 * x0 + s1 * x1
    y = s0 * y0 + s1 * y1
    z = s0 * z0 + s1 * z1
    w = s0 * w0 + s1 * w1

    return normalize_quat((x, y, z, w))
