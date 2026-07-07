"""Provide a variety of rotation matrices.

Sandia Proprietary
"""

import numpy as np

from traj_estimation.inspyre.earth import w_ie


def ecef_to_local_nav(lat, lon):
    """Compute the rotation matrix from ECEF to local navigation (C_e^n).

    Parameters
    ----------
    lat : float
        Latitude in radians
    lon : float
        Longitude in radians

    Returns
    -------
    np.ndarray :
        Rotation matrix, C_e^n

    Notes
    -----
    See eq. 2.150

    Local navigation frame is NED at the current location.
    """
    return np.array(
        [
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [-np.sin(lon), np.cos(lon), 0],
            [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)],
        ]
    )


def local_nav_to_ecef(lat, lon):
    """Compute the rotation matrix from local navigation to ECEF (C_n^e).

    Parameters
    ----------
    lat : float
        Latitude in radians
    lon : float
        Longitude in radians

    Returns
    -------
    np.ndarray :
        Rotation matrix, C_n^e

    Notes
    -----
    See eq. 2.150

    """
    return ecef_to_local_nav(lat, lon).T


def ecef_to_eci(dt):
    """Compute the rotation matrix from ECEF to ECI (C_e^i).

    Parameters
    ----------
    lat : float
        Latitude in radians
    lon : float
        Longitude in radians

    Returns
    -------
    np.ndarray :
        Rotation matrix, C_e^i

    Notes
    -----
    See eq. 2.145
    """
    return np.array(
        [
            [np.cos(w_ie * dt), -np.sin(w_ie * dt), 0],
            [np.sin(w_ie * dt), np.cos(w_ie * dt), 0],
            [0, 0, 1],
        ]
    )


def eci_to_ecef(dt):
    """Compute the rotation matrix from ECI to ECEF (C_i^e).

    Parameters
    ----------
    lat : float
        Latitude in radians
    lon : float
        Longitude in radians

    Returns
    -------
    np.ndarray :
        Rotation matrix, C_i^e

    Notes
    -----
    See eq. 2.145
    """
    return ecef_to_eci(dt).T


def local_tangent_to_ecef(lat0, lon0):
    """Compute the rotation matrix from local tangent-plane (NED) to ECEF (C_l^e).

    Parameters
    ----------
    lat0 : float
        Latitude at origin in radians
    lon0 : float
        Longitude at origin in radians

    Returns
    -------
    np.ndarray :
        Rotation matrix, C_l^e

    Notes
    -----
    See eq. 2.158
    """
    return local_nav_to_ecef(lat0, lon0)


def ecef_to_local_tangent(lat0, lon0):
    """Compute the rotation matrix from ECEF to local tangent-plane (NED) (C_e^l).

    Parameters
    ----------
    lat0 : float
        Latitude at origin in radians
    lon0 : float
        Longitude at origin in radians

    Returns
    -------
    np.ndarray :
        Rotation matrix, C_e^l

    Notes
    -----
    See eq. 2.158
    """
    return ecef_to_local_nav(lat0, lon0)
