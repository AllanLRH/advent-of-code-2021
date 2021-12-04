from src.adv03 import bit_criteria, get_rate
import pytest
import numpy as np


@pytest.fixture()
def data() -> np.ndarray:
    data = np.asarray(
        [
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
        ],
        dtype=int,
    )
    return data


def test_bit_criteria(data: np.ndarray) -> None:
    mask_oxygen = bit_criteria(data[:, 0], "oxygen")
    mask_co2 = bit_criteria(data[:, 0], "co2")
    expected_mask = np.asarray([0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
    assert (mask_oxygen == expected_mask).all()


def test_get_rate_oxygen(data):
    rate = get_rate(data, "oxygen")
    expected_rate = np.asarray([1, 0, 1, 1, 1], dtype=int)
    assert (rate == expected_rate).all()


def test_get_rate_co2(data):
    rate = get_rate(data, "co2")
    expected_rate = np.asarray([0, 1, 0, 1, 0], dtype=int)
    assert (rate == expected_rate).all()
