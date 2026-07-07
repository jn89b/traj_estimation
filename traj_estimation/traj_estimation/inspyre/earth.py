"""Various functions and parameters related to the Earth.

Sandia Proprietary
"""

import numpy as np

w_ie = 7.2921151467e-5  # rad/s
w_ie_squared = 5.317494331273156252089e-9
R_0 = 6378137  # WGS84 Equatorial radius in meters
R_p = 6356752.31425  # WGS74 Polar radius in meters
e = 0.0818191908425  # WGS84 eccentricity
f = 1 / 298.257223563  # flattening
mu = 3.986004418e14  # WGS84 gravitational constant
j2 = 0.108262999e-2  # Second degree zonal harmonic

w_e_ie = np.array([0, 0, w_ie])
Omega_e_ie = np.array([[0, -w_ie, 0], [w_ie, 0, 0], [0, 0, 0]])


def gravity(r_e_eb: np.ndarray) -> np.ndarray:
    """Compute the gravity components at the specified location.

    Parameters
    ----------
    r_e_eb : np.ndarray
        (x, y, z) location of the body resolved in ECEF frame

    Returns
    -------
    np.ndarray :
        (x, y, z) gravity components in ECEF frame

    Notes
    -----
    Eqn. 2.142 from Groves
    """
    r = np.linalg.norm(r_e_eb)
    sub1 = 1.5 * j2 * (R_0 / r) * (R_0 / r)
    sub2 = 5.0 * r_e_eb[2] * r_e_eb[2] / (r * r)
    sub3 = -mu / (r * r * r)
    sub4 = sub3 * (1.0 - sub1 * (sub2 - 1.0))
    return np.array(
        [
            r_e_eb[0] * sub4,
            r_e_eb[1] * sub4,
            r_e_eb[2] * sub3 * (1.0 - sub1 * (sub2 - 3.0)),
        ]
    )


def gravity_gradient(r_e_eb: np.ndarray) -> np.ndarray:
    """Compute the partial derivative of gravity at the specified location.

    Parameters
    ----------
    r_e_eb : np.ndarray
        (x, y, z) location of the body resolved in the ECEF frame

    Returns
    -------
    np.ndarray :
        (dG/dx, dG/dy, dG/dz) partial derivative components

    Notes
    -----
    Compute gradient via finite difference
    """
    # Finite difference with +/- 50 meters
    d = 50
    dx = np.array([d, 0, 0])
    dy = np.array([0, d, 0])
    dz = np.array([0, 0, d])

    dg_dx = (gravity(r_e_eb + dx) - gravity(r_e_eb - dx)) / (2 * d)
    dg_dy = (gravity(r_e_eb + dy) - gravity(r_e_eb - dy)) / (2 * d)
    dg_dz = (gravity(r_e_eb + dz) - gravity(r_e_eb - dz)) / (2 * d)
    return np.array([dg_dx, dg_dy, dg_dz])


def centrifugal(r_e_eb: np.ndarray) -> np.ndarray:
    """Compute the centrigual acceleration components at the specified location.

    Parameters
    ----------
    r_e_eb : np.ndarray
        (x, y, z) location of the body resolved in ECEF frame

    Returns
    -------
    np.ndarray :
        (x, y, z) centrifugal acceleration components in ECEF frame

    Notes
    -----
    See Eqns. 2.81, 5.33, and 5.34 from Groves
    """
    return np.array([-w_ie_squared * r_e_eb[0], -w_ie_squared * r_e_eb[1], 0.0])


def coriolis(v_e_eb: np.ndarray) -> np.ndarray:
    """Compute the coriolis effect components at the specified location.

    Parameters
    ----------
    v_e_eb : np.ndarray
        (v_x, v_y, v_z) velocity of the body resolved in ECEF frame

    Returns
    -------
    np.ndarray :
        (x, y, z) coriolis effect components in ECEF frame

    Notes
    -----
    See Eqns. 2.81, 5.33, and 5.34 from Groves
    """
    return np.array([-2.0 * w_ie * v_e_eb[1], 2.0 * w_ie * v_e_eb[0], 0.0])


def effective_radius(lat):
    """Compute the effective Earth radius at the given latitude.

    Parameters
    ----------
    lat : float
        latitude in radians.

    Notes
    -----
    Eqn. 2.106 from Groves
    """
    return R_0 / np.sqrt(1 - e**2 * np.sin(lat) ** 2)


def ecef_to_ned(r_e_eb: np.ndarray) -> np.ndarray:
    """Convert ECEF frame to lat, lon, alt.

    Parameters
    ----------
    r_e_eb : np.ndarray
        [x, y, z] coordinates in ECEF frame

    Returns
    -------
    np.ndarray :
        [lat, lon, alt]

    Notes
    -----
    See appendix C of Groves for Borkowski closed-form exact solution
    """
    x, y, z = r_e_eb
    k1 = np.sqrt(1 - e**2) * np.abs(z)
    k2 = e**2 * R_0
    beta = np.sqrt(x**2 + y**2)
    E = (k1 - k2) / beta  # noqa: N806
    F = (k1 + k2) / beta  # noqa: N806
    P = 4 / 3 * (E * F + 1)  # noqa: N806
    Q = 2 * (E**2 - F**2)  # noqa: N806
    D = P**3 + Q**2  # noqa: N806
    V = (np.sqrt(D) - Q) ** (1 / 3) - (np.sqrt(D) + Q) ** (1 / 3)  # noqa: N806
    G = (np.sqrt(E**2 + V) + E) / 2  # noqa: N806
    T = np.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G  # noqa: N806
    L_b = np.sign(z) * np.arctan((1 - T**2) / (2 * T * np.sqrt(1 - e**2)))  # noqa: N806
    h_b = (beta - R_0 * T) * np.cos(L_b) + (
        z - np.sign(z) * R_0 * np.sqrt(1 - e**2)
    ) * np.sin(L_b)
    lambda_b = np.arctan2(y, x)

    return np.array([L_b, lambda_b, h_b])


def ned_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """Convert lat, lon, alt to ECEF frame.

    Parameters
    ----------
    lat : float
        latitude in radians
    lon : float
        longitude in radians
    alt : float
        altitude in meters

    Returns
    -------
    np.ndarray :
        x, y, z coordinates in ECEF frame

    Notes
    -----
    Eqn. 2.112 from Groves
    """
    Re = effective_radius(lat)  # noqa: N806
    x = (Re + alt) * np.cos(lat) * np.cos(lon)
    y = (Re + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - e**2) * Re + alt) * np.sin(lat)
    return np.array([x, y, z])


def ned_to_local(
    reference_ned: np.ndarray, local_coordinate: np.ndarray, ned: np.ndarray
) -> np.ndarray:
    """Convert lat, lon, alt to local frame (north, east, down in meters).

    Parameters
    ----------
    reference_ned : np.ndarray
        A lat, lon, alt measurement with a known local coordinate, can be 0, 0, 0 or other coordinate
    local_coordinate : np.ndarray
        The local coordinate corresponding to the reference_ned location
    ned : np.ndarray
        The lat, lon, alt measurement to be converted into local coordinates

    Returns
    -------
    np.ndarray :
        North, East, down relative to the (0, 0, 0) local coordinate in meters

    Notes
    -----
    Latitude and longitude numbers should be in radians.
    """
    Re = effective_radius(ned[0])  # noqa: N806
    diff = reference_ned - ned
    north = np.sin(-diff[0]) * R_p
    east = np.sin(-diff[1]) * Re
    down = diff[2]
    delta = np.array((north, east, down))
    return local_coordinate + delta
