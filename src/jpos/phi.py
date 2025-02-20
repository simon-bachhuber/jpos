import warnings

import numpy as np


def estimate_phi(
    acc1: np.ndarray,
    gyr1: np.ndarray,
    acc2: np.ndarray,
    gyr2: np.ndarray,
    hz: float,
    backend: str = "qmt",
):
    for arr in [gyr1, gyr2]:
        max_val = np.max(np.abs(arr))
        if max_val > 10:
            warnings.warn(
                f"Found very large gyroscope value of {max_val}. Are you sure "
                "you have provided Gyroscope values in radians?"
            )

    assert backend in ["qmt", "ring"], f"Backend `{backend}` not available"
    T, dt = len(acc1), 1 / hz

    try:
        import qmt

        # use ollson to compute the joint axis
        axis = qmt.jointAxisEstHingeOlsson(
            acc1,
            acc2,
            gyr1,
            gyr2,
            estSettings=dict(quiet=True),
        )[0][:, 0]
        assert axis.shape == (3,)
    except ImportError:
        axis = np.zeros((3,))

    if backend == "qmt":
        import qmt

        # from 1 to eps
        q1 = qmt.oriEstVQF(gyr1, acc1, params=dict(Ts=dt))
        q1 = qmt.qmult(q1, qmt.qinv())
        # from 2 to eps
        q2 = qmt.oriEstVQF(gyr2, acc2, params=dict(Ts=dt))
        q2c = qmt.headingCorrection(
            gyr1,
            gyr2,
            q1,
            q2,
            np.arange(T * dt, step=dt),
            axis,
            None,
            estSettings=dict(constraint="1d_corr"),
        )[0]
        # compute from 2 to 1
        qrel = qmt.qmult(qmt.qinv(q1), q2c)
    else:
        import ring

        X = np.zeros((T, 2, 9))
        X[:, 0, :3] = acc1
        X[:, 0, 3:6] = gyr1
        X[:, 1, :3] = acc2
        X[:, 1, 3:6] = gyr2
        X[:, 1, 6:9] = axis
        qs, _ = ring.RING([-1, 0], dt).apply(X=X)
        qrel = qs[:, 1]

    return qmt.quatAngle(qrel)
