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


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv04.txt"
    draws, boards = read_data(data_path)
    # print(draws, boards)

    # First star
