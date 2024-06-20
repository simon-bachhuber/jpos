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
