import numpy as np

from jpos import solve


def test_quickstart_example():
    draw = lambda: np.clip(np.random.normal(size=(500, 3)), -10, 10)

    _ = solve(draw(), draw(), draw(), draw(), draw()[:, 0], 60)
    _ = solve(draw(), draw(), draw(), draw(), draw()[:, 0:1], 60)
