from pathlib import Path
import numpy as np


def read_data(pth: Path) -> list[tuple[str, int]]:
    with pth.open() as fid:
        return [
            (direction, int(magnitude))
            for (direction, magnitude) in map(str.split, fid)
        ]


def process_line_first(direction: str, magnitude: int) -> np.ndarray:
    displacement = np.zeros(2, dtype=int)
    if direction == "forward":
        displacement[0] = magnitude
    elif direction == "down":
        displacement[1] = magnitude
    elif direction == "up":
        displacement[1] = -magnitude
    else:
        raise ValueError(f"Invalid command {direction=}")
    return displacement


def resolve_path_second(data: list[tuple[str, int]]) -> tuple[int, int, int]:
    aim, hor, dep = 0, 0, 0
    for direction, v in data:
        if direction == "down":
            aim += v
        elif direction == "up":
            aim -= v
        elif direction == "forward":
            hor += v
            dep += aim * v
        else:
            raise ValueError(f"Invalid command {direction=}")
    return (aim, hor, dep)


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv02.txt"
    data = read_data(data_path)

    processed_1 = np.asarray([process_line_first(*tup) for tup in data]).squeeze()
    final_position_first = processed_1.sum(axis=0)
    print(f"{final_position_first=}")
    print(f"{final_position_first.prod()=}")

    final_position_second = resolve_path_second(data)
    answer = final_position_second[1] * final_position_second[2]
    print(f"{final_position_second=}")
    print(f"{answer=}")
