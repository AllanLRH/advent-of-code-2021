from pathlib import Path
import re
import numpy as np


def read_data(pth: Path) -> tuple[np.ndarray, np.ndarray]:
    def parse_board(lines: list[str]) -> np.ndarray:
        assert len(lines) == 5, f"Invalid board, {len(lines)=}"
        board = [[int(el) for el in re.split(r" +", line.strip())] for line in lines]
        return board

    with pth.open() as fid:
        raw = fid.readlines()
    draws = [int(el) for el in raw[0].strip().split(",")]
    boards = list()
    for i in range(2, len(raw), 6):
        board = parse_board(raw[i : i + 5])
        boards.append(board)
    return draws, np.asarray(boards)


def check_boards_for_winners(marked_numbers: np.ndarray) -> np.ndarray:
    full_cols = marked_numbers.all(axis=1)
    full_rows = marked_numbers.all(axis=2)
    full_either = full_cols | full_rows
    if full_either.any():
        board_indices, _ = np.nonzero(full_either)
        return np.unique(board_indices)
    return np.array([])


def run_game(boards: np.ndarray, draws: np.ndarray) -> np.ndarray:
    marked_numbers = np.zeros_like(boards, dtype=bool)
    scores = list()
    for number in draws:
        mask = boards == number
        marked_numbers |= mask
        winning_boards = check_boards_for_winners(marked_numbers)
        if winning_boards.any():
            # print(f"{winning_boards=}, {number=}")
            for winner in winning_boards:
                winboard = boards[winner]
                winmarked = marked_numbers[winner]
                score = winboard[~winmarked].sum() * number
                scores.append(score)
                # print(f"{winboard=}")
                # print(f"{winmarked=}")
                # print(f"{winboard[~winmarked].sum()=}")
                # print(f"{winboard[~winmarked]=}")
            return np.asarray(scores)
    return np.array([])


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv04.txt"
    draws, boards = read_data(data_path)
    # print(draws, boards)

    # First star
    answer = run_game(boards, draws)
    print(f"{answer=}")
