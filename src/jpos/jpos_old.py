from functools import partial

import jax
import jax.numpy as jnp
import jaxopt
from scipy.signal import butter
from scipy.signal import sosfiltfilt


def constraint(acc1, gyr1, gyrdot1, r1, v1, a1, acc2, gyr2, gyrdot2, r2, v2, a2):
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


def lpf(x, hz, cutoff):
    return np.stack(
        [
            sosfiltfilt(butter(4, cutoff, output="sos", fs=hz), x[:, i])
            for i in range(x.shape[-1])
        ]
    ).T


def dot(x, dt: float, lpf_freq: float | None = 10.0):
    if lpf_freq is not None:
        x = lpf(x, 1 / dt, lpf_freq)
    xdot = (x[2:] - x[:-2]) / (2 * dt)
    return jnp.vstack((xdot[0][None], xdot, xdot[-1][None]))


def poly_forward(params, x):
    order = params.size - 2
    return params[1:] @ jnp.power(x - params[0], jnp.arange(0, order + 1))


def poly_predict(params, x):
    width = int(params.size / 3)
    r2 = []
    for i in range(3):
        r2.append(poly_forward(params[width * i : width * (i + 1)], x))
    r2 = jnp.array(r2)
    return r2


def solve(
    acc1,
    gyr1,
    acc2,
    gyr2,
    hz: float,
    phi=None,
    max_iters: int = 100,
    eps=1e-5,
    order: int = 2,
    solve_seel=False,
    verbose: bool = False,
    num: bool = False,
    minus_a: bool = False,
):
    gyrdot1 = dot(gyr1, 1 / hz)
    gyrdot2 = dot(gyr2, 1 / hz)

    acc1 = lpf(acc1, hz, 25)
    acc2 = lpf(acc2, hz, 25)

    def error(x, acc1, gyr1, gyrdot1, acc2, gyr2, gyrdot2, _phi, _phidot, t):
        r1 = x[:3]
        v1 = jnp.zeros_like(gyr1)
        a1 = jnp.zeros_like(v1)
        params = x[3:]

        phi3 = jax.lax.dynamic_slice_in_dim(_phi, t + 1, 3)
        phidot3 = jax.lax.dynamic_slice_in_dim(_phidot, t + 1, 3)

        if phi is not None:
            r2 = poly_predict(params, phi3[1])
            if num:
                v2 = (
                    jax.vmap(jax.jacfwd(lambda phi: poly_predict(params, phi)))(phi3)
                    * phidot3
                )
                a2 = dot(v2, 1 / hz, None)[1]
                v2 = v2[1]
            else:
                v2_of_phi = (
                    lambda phi: jax.jacfwd(lambda phi: poly_predict(params, phi))(phi)
                    * phidot3[1]
                )
                v2 = v2_of_phi(phi3[1])
                a2 = jax.jacfwd(v2_of_phi)(phi3[1]) * phidot3[1]
        else:
            r2 = x[3:]
            v2 = jnp.zeros_like(gyr2)
            a2 = jnp.zeros_like(v2)

        if minus_a:
            a2 = -a2

        # Debug statements
        print("t =", t)
        print("r1 =", r1)
        print("r2 =", r2)
        print("v1 =", v1)
        print("v2 =", v2)
        print("a1 =", a1)
        print("a2 =", a2)
        print("minus_a =", minus_a)

        return constraint(
            acc1, gyr1, gyrdot1, r1, v1, a1, acc2, gyr2, gyrdot2, r2, v2, a2
        )

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, None, None, 0))
    def vmap_jacobian(x, acc1, gyr1, gyrdot1, acc2, gyr2, gyrdot2, _phi, _phidot, ts):
        @jax.grad
        def gradient(x):
            return error(x, acc1, gyr1, gyrdot1, acc2, gyr2, gyrdot2, _phi, _phidot, ts)

        return gradient(x)

    vmap_error = jax.vmap(error, in_axes=(None, 0, 0, 0, 0, 0, 0, None, None, 0))

    if phi is not None:
        _phi = phi
        assert _phi.ndim == 1
        _phi = jnp.vstack((_phi[0][None, None], _phi[:, None], _phi[-1][None, None]))[
            :, 0
        ]
        x = np.random.normal(size=(3 + 3 * (order + 2),)) * 0.2
    else:
        _phi = jnp.zeros((acc1.shape[0] + 2,))
        x = np.random.normal(size=(6,)) * 0.2

    _phidot = dot(_phi[:, None], 1 / hz)[:, 0]
    ts = jnp.arange(len(acc1))

    def solve1(x):
        @jax.jit
        def step(x):
            dedx = vmap_jacobian(
                x, acc1, gyr1, gyrdot1, acc2, gyr2, gyrdot2, _phi, _phidot, ts
            )
            e = vmap_error(
                x, acc1, gyr1, gyrdot1, acc2, gyr2, gyrdot2, _phi, _phidot, ts
            )
            return x - jnp.linalg.pinv(dedx) @ e, e

        for _ in range(max_iters):
            next_x, e = step(x)
            delta = jnp.linalg.norm(next_x - x)
            if verbose:
                print(delta, np.mean(e**2))
            if delta < eps:
                break
            x = next_x
        return x, np.mean(e**2)

    def solve2(x):
        def f(x):
            e = vmap_error(
                x, acc1, gyr1, gyrdot1, acc2, gyr2, gyrdot2, _phi, _phidot, ts
            )
            return jnp.mean(e**2)

        res = jaxopt.GradientDescent(f, maxiter=max_iters).run(x)
        return res.params, f(res.params)

    solve = solve1 if solve_seel else solve2

    x, e = solve(x)
    if verbose:
        print(f"Final residual={e} meter")
    return x, e
