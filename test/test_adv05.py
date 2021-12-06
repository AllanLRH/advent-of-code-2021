import pytest
import numpy as np
import textwrap
from src.adv05 import (
    make_simple_map,
    is_vertical_or_horizontal,
    is_diagonal,
    align_coordinates,
)


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


def test_simple_map_draw_straight():
    coords = np.asarray([[0, 5, 4, 5], [1, 5, 3, 5], [3, 1, 3, 4]])
    seafloor = make_simple_map(coords, map_shape=(6, 6))
    expected = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 2, 2, 2, 1, 0],
        ],
        dtype=int,
    )
    assert seafloor.sum() > 0
    assert (seafloor == expected).all()


def test_align_coordinates():
    data = np.asarray([1, 10, 3, 4])
    aligned = align_coordinates(data)
    expected = np.asarray([1, 4, 3, 10])
    assert (aligned == expected).all()


def test_is_vertical_or_horizontal():
    data = np.asarray([[0, 1, 0, 1], [0, 1, 0, 2], [0, 1, 2, 1], [1, 2, 3, 4]])
    h_or_v = [is_vertical_or_horizontal(coords) for coords in data]
    assert h_or_v == [True, True, True, False]


def test_is_diagonal():
    data = np.asarray([[0, 0, 3, 3], [3, 3, 0, 0], [4, 4, 2, 6], [0, 0, 3, 4]])
    is_diag = [is_diagonal(coors) for coors in data]
    assert is_diag == [True, True, True, False]


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
    straight_lines = np.asarray([row for row in data if is_vertical_or_horizontal(row)])
    seafloor = make_simple_map(straight_lines)
    assert (seafloor == expected).all()
