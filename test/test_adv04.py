import pytest
import numpy as np
from src.adv04 import check_boards_for_winners, run_game


@pytest.fixture()
def drawn_numbers() -> np.ndarray:
    # fmt: off
    numbers = np.asarray(
        [
            7, 4, 9, 5, 11, 17, 23, 2, 0, 14, 21,
            24, 10, 16, 13, 6, 15, 25, 12, 22, 18,
            20, 8, 19, 3, 26, 1
        ],
        dtype=int
    )
    # fmt: on
    return numbers


@pytest.fixture()
def boards() -> np.ndarray:
    # fmt: off
    boards = np.asarray([
                [[22, 13, 17, 11,  0],
                [ 8,  2, 23,  4, 24],
                [21,  9, 14, 16,  7],
                [ 6, 10,  3, 18,  5],
                [ 1, 12, 20, 15, 19]],

                [[ 3, 15,  0,  2, 22],
                [ 9, 18, 13, 17,  5],
                [19,  8,  7, 25, 23],
                [20, 11, 10, 24,  4],
                [14, 21, 16, 12,  6]],

                [[14, 21, 17, 24,  4],
                [10, 16, 15,  9, 19],
                [18,  8, 23, 26, 20],
                [22, 11, 13,  6,  5],
                [ 2,  0, 12,  3,  7]]
                ]
            )
    # fmt: on
    return boards


def test_check_boards_for_solutions():
    observed = np.asarray(
        [
            [
                [False, False, False, False, False],
                [False, False, False, False, True],
                [False, False, True, False, False],
                [True, False, False, False, False],
                [True, False, False, False, False],
            ],
            [
                [True, True, True, True, True],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            [
                [True, False, True, False, False],
                [False, False, True, False, False],
                [False, False, True, False, False],
                [True, True, True, False, False],
                [False, False, True, False, False],
            ],
        ]
    )
    wining_boards = sorted(check_boards_for_winners(observed))
    assert wining_boards == [1, 2]


def test_run_game(boards, drawn_numbers):
    winning_score = run_game(boards, drawn_numbers)
    assert int(winning_score) == 4512
