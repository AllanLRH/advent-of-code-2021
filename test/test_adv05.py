import pytest
import numpy as np
import textwrap
from src.adv05 import make_simple_map


@pytest.fixture()
def data() -> np.ndarray:
    return np.asarray(
        [
            [0, 9, 5, 9],
            [8, 0, 0, 8],
            [9, 4, 3, 4],
            [2, 2, 2, 1],
            [7, 0, 7, 4],
            [6, 4, 2, 0],
            [0, 9, 2, 9],
            [3, 4, 1, 4],
            [0, 0, 8, 8],
            [5, 5, 8, 2],
        ],
        dtype=int,
    )


def test_draw_simple_map(data):
    expected = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 2, 1, 1, 1, 2, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    seafloor = make_simple_map(data)
    assert (seafloor == expected).all()


def test_simple_map_draw():
    coords = np.asarray([[0, 0, 0, 4], [4, 2, 0, 2]])
    seafloor = make_simple_map(coords, map_shape=(6, 6))
    expected = np.asarray(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [2, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    assert (seafloor == expected).all()
