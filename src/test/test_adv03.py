from code.adv03 import bit_criteria
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
    assert (mask_co2 == ~expected_mask).all()
    assert (mask_oxygen == expected_mask).all()
