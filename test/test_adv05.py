import pytest
import numpy as np
from src.adv05 import SeaFloor, get_map_shape_from_data


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
    seafloor = SeaFloor(map_shape=(6, 6))
    for row in coords:
        seafloor.draw_route(row)
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
    assert seafloor.data.sum() > 0
    try:
        np.testing.assert_equal(seafloor.data, expected)
    except AssertionError:
        print(seafloor.data)
        raise


def test_is_vertical_or_horizontal():
    data = np.asarray([[0, 1, 0, 1], [0, 1, 0, 2], [0, 1, 2, 1], [1, 2, 3, 4]])
    seafloor = SeaFloor((1, 1))
    h_or_v = [seafloor.is_vertical_or_horizontal(coords) for coords in data]
    assert h_or_v == [True, True, True, False]


def test_is_diagonal():
    data = np.asarray([[0, 0, 3, 3], [3, 3, 0, 0], [4, 4, 2, 6], [0, 0, 3, 4]])
    seafloor = SeaFloor((1, 1))
    is_diag = [seafloor.is_diagonal(coords) for coords in data]
    assert is_diag == [True, True, True, False]


# @pytest.mark.skip("Not implemented yet")
def test_draw_straight_map(data):
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
    map_shape = get_map_shape_from_data(data)
    seafloor = SeaFloor(map_shape)
    straight_lines = np.asarray(
        [row for row in data if seafloor.is_vertical_or_horizontal(row)]
    )
    for row in straight_lines:
        seafloor.draw_route(row)
    np.testing.assert_equal(seafloor.data, expected)


# @pytest.mark.skip("Not implemented yet")
def test_draw_diagonal_map(data):
    expected = np.asarray(
        [
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 2, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 0, 2, 0, 0, 0],
            [0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    seafloor = SeaFloor(expected.shape)
    diagonals = np.asarray(
        [row for row in data if seafloor.is_diagonal(row)],
        dtype=int,
    )
    print(seafloor.data)
    for row in diagonals:
        seafloor.draw_route(row)
        print(row)
        print(seafloor.data)
    try:
        np.testing.assert_equal(seafloor.data, expected)
    except AssertionError:
        # print(seafloor)
        # print(seafloor.data - expected)
        raise


# @pytest.mark.skip("Not implemented yet")
def test_draw_complex_map(data):
    expected = np.asarray(
        [
            [1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 2, 0, 2, 0, 0],
            [0, 1, 1, 2, 3, 1, 3, 2, 1, 1],
            [0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [2, 2, 2, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    seafloor = SeaFloor(expected.shape)
    for row in data:
        seafloor.draw_route(row)
    try:
        np.testing.assert_equal(seafloor.data, expected)
    except AssertionError:
        print(seafloor)
        print(seafloor.data - expected)
        raise
