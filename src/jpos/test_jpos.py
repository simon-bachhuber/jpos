import numpy as np

from jpos import solve


def test_quickstart_example():
    draw = lambda: np.clip(np.random.normal(size=(500, 3)), -10, 10)

    r1, r2, infos = solve(draw(), draw(), draw(), draw(), draw()[:, 0], 60, seed=1)
    assert r2.shape == (500, 3)
    r1_2, r2_2, infos = solve(
        draw(), draw(), draw(), draw(), draw()[:, 0:1], 60, seed=1
    )
    np.testing.assert_array_equal(r2, r2_2)


def test_phi_is_None():
    draw = lambda: np.clip(np.random.normal(size=(500, 3)), -10, 10)
    a1, a2, g1, g2, phi = draw(), draw(), draw(), draw(), draw()[:, 0]

    r1, r2, infos = solve(a1, g1, a2, g2, phi, 60, seed=1, order=0)
    r1_2, r2_2, infos = solve(a1, g1, a2, g2, None, 60, seed=1, order=0)
    np.testing.assert_array_equal(r2, r2_2)
