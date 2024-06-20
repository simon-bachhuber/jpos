from typing import Optional
import warnings

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from scipy.signal import butter
from scipy.signal import sosfiltfilt


def _constraint(acc1, gyr1, gyrdot1, r1, v1, a1, acc2, gyr2, gyrdot2, r2, v2, a2):
    def gamma(gyr, gyrdot, r, v, a):
        return (
            jnp.cross(gyr, jnp.cross(gyr, r))
            + jnp.cross(gyrdot, r)
            + 2 * jnp.cross(gyr, v)
            + a
        )

    return jnp.linalg.norm(
        acc1 - gamma(gyr1, gyrdot1, r1, v1, a1), axis=-1
    ) - jnp.linalg.norm(acc2 - gamma(gyr2, gyrdot2, r2, v2, a2), axis=-1)


def _lpf(x, hz, cutoff):
    return np.stack(
        [
            sosfiltfilt(butter(4, cutoff, output="sos", fs=hz), x[:, i])
            for i in range(x.shape[-1])
        ]
    ).T


def _dot(x, hz: float, lpf_freq: float | None = 10.0):
    if lpf_freq is not None:
        x = _lpf(x, hz, lpf_freq)
    xdot = (x[2:] - x[:-2]) / (2 * (1 / hz))
    return jnp.vstack((xdot[0][None], xdot, xdot[-1][None]))


def _poly_predict(params, x):

    def poly_forward(params, x):
        order = params.size - 2
        return params[1:] @ jnp.power(x - params[0], jnp.arange(0, order + 1))

    width = int(params.size / 3)
    r2 = []
    for i in range(3):
        r2.append(poly_forward(params[width * i : width * (i + 1)], x))  # noqa: E203
    r2 = jnp.array(r2)
    return r2


def solve(
    acc1: np.ndarray,
    gyr1: np.ndarray,
    acc2: np.ndarray,
    gyr2: np.ndarray,
    phi: np.ndarray,
    hz: float,
    order: int = 2,
    verbose: bool = False,
    n_grid_points: int = 50,
    seed: Optional[int] = None,
):
    """Joint Translation Estimation. Estimates the vector from joint center to IMU 1
    and IMU2 where the IMU1 is rigidly attached and the IMU2 is nonrigidly attached.

    Args:
        acc1 (np.ndarray): Nx3, m/s**2, with or without gravity
        gyr1 (np.ndarray): Nx3, rad/s
        acc2 (np.ndarray): Nx3, m/s**2 with or without gravity
        gyr2 (np.ndarray): Nx3, rad/s
        phi (np.ndarray): Joint angle over time in radians, unused if `order`=0
        hz (float): Sampling rate, Hz
        order (int, optional): Order of the polynomial of joint translation of second
            IMU. Defaults to 2.
        verbose (bool, optional): Print information to stdout. Defaults to False.
        n_grid_points (int, optional): Number of grid points in the `joint-to-imu2-grid`
            return value. Defaults to 50.
        seed (int, optional): Seed used for initilization of the optimization. By
            default this function is non-deterministic. Fixing the seed makes this
            function deterministic.

    Returns:
        tuple: Array (3,), Array (Nx3), dict
        which are joint-to-imu1-vector, joint-to-imu2-timeseries, and infos dictionary
        where both joint-to-imu vectors are given in the respective local sensor
        coordinate system in meters.
    """
    if seed is not None:
        np.random.seed(seed)

    T = acc1.shape[0]
    for arr in [acc1, gyr1, acc2, gyr2]:
        assert arr.shape == (
            T,
            3,
        ), (
            "All IMU data must be given as a Nx3 array with consistent number of "
            + f"samples `N` but found {arr.shape}"
        )

    for arr in [gyr1, gyr2, phi]:
        max_val = np.max(np.abs(arr))
        if max_val > 10:
            warnings.warn(
                f"Found very large gyroscope or phi value of {max_val}. Are you sure "
                "you have provided Gyroscope and `phi` value in radians?"
            )

    phi = phi[None] if phi.ndim == 0 else phi
    phi = phi[:, None] if phi.ndim == 1 else phi
    assert phi.shape[0] == T, f"{phi.shape[0]} != {T}"

    acc1 = _lpf(acc1, hz, 25)
    acc2 = _lpf(acc2, hz, 25)
    phi = _lpf(phi, hz, 25)

    gyrdot1 = _dot(gyr1, hz)
    gyrdot2 = _dot(gyr2, hz)
    phidot = _dot(phi, hz)
    phidotdot = _dot(phidot, hz)
    # convert from (N, 1) -> (N,)
    phi, phidot, phidotdot = phi[:, 0], phidot[:, 0], phidotdot[:, 0]

    def residual(x, acc1, gyr1, gyrdot1, acc2, gyr2, gyrdot2, phi, phidot, phidotdot):
        r1 = x[:3]
        v1 = jnp.zeros_like(gyr1)
        a1 = jnp.zeros_like(v1)
        params = x[3:]
        r2 = _poly_predict(params, phi)
        drdphi = jax.jacfwd(lambda phi: _poly_predict(params, phi))
        v2 = drdphi(phi) * phidot
        a2 = jax.jacfwd(drdphi)(phi) * phidot**2 + drdphi(phi) * phidotdot
        return _constraint(
            acc1, gyr1, gyrdot1, r1, v1, a1, acc2, gyr2, gyrdot2, r2, v2, a2
        )

    initial_params = np.random.normal(size=(3 + 3 * (order + 2),)) * 0.2

    def mean_squared_residual(x):
        e = jax.vmap(residual, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
            x,
            acc1,
            gyr1,
            gyrdot1,
            acc2,
            gyr2,
            gyrdot2,
            phi,
            phidot,
            phidotdot,
        )
        return jnp.mean(e**2)

    res = jaxopt.GradientDescent(mean_squared_residual, maxiter=500).run(initial_params)
    params, final_residual = res.params, mean_squared_residual(res.params)

    if verbose:
        print(f"Final residual={final_residual} m/s**2")

    phi_grid = jnp.linspace(
        jnp.min(phi), jnp.max(phi), endpoint=True, num=n_grid_points
    )
    r2_grid = jax.vmap(_poly_predict, in_axes=(None, 0))(params[3:], phi_grid)
    r1 = params[:3]
    r2 = jax.vmap(_poly_predict, in_axes=(None, 0))(params[3:], phi)
    return (
        r1,
        r2,
        {
            "phi_grid_rad": phi_grid,
            "r2_grid_m": r2_grid,
            "final residual m/s**2": final_residual,
            "polynomial_coefs": params[3:],
            "jaxopt_results": res,
        },
    )
