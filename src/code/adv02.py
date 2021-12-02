from pathlib import Path
import numpy as np


def read_data(pth: Path) -> list[tuple[str, int]]:
    with pth.open() as fid:
        return [
            (direction, int(magnitude))
            for (direction, magnitude) in map(str.split, fid)
        ]


def process_line_first(direction: str, magnitude: int) -> np.ndarray:
    aim = np.zeros(2, dtype=int)
    if direction == "forward":
        aim[0] = magnitude
    elif direction == "down":
        aim[1] = magnitude
    elif direction == "up":
        aim[1] = -magnitude
    return aim


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv02.txt"
    data = read_data(data_path)

    processed_1 = np.asarray([process_line_first(*tup) for tup in data]).squeeze()
    final_position = processed_1.sum(axis=0)
    print(f"{final_position=}")
    print(f"{final_position.prod()=}")
