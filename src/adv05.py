# %% imports
from pathlib import Path
import numpy as np
import io
from typing import Optional

# %% functions


def read_data(pth: Path) -> np.ndarray:
    """
    # Returns an ndarray in the format r1, c1, r2, c2.
    Returns an ndarray in the format c1, r1, c2, r2.
    """
    with pth.open() as fid:
        raw = io.StringIO(fid.read().replace(" -> ", ","))
    data = np.loadtxt(raw, delimiter=",", dtype=int)
    # return data[:, [1, 0, 3, 2]]
    return data


def get_map_shape_from_data(data: np.ndarray) -> tuple[int, int]:
    x = data[:, [0, 2]].max() + 1
    y = data[:, [1, 3]].max() + 1
    return (x, y)


class SeaFloor:
    def __init__(self, map_shape: tuple[int, int]) -> None:
        self.data = np.zeros(map_shape, dtype=int)

    @staticmethod
    def is_vertical_or_horizontal(coords: np.ndarray) -> bool:
        r1, c1, r2, c2 = coords
        return (r1 == r2) | (c1 == c2)

    @staticmethod
    def is_diagonal(coords: np.ndarray) -> bool:
        r1, c1, r2, c2 = coords
        delta_y, delta_x = (c2 - c1), (r2 - r1)
        return abs(delta_y) == abs(delta_x)

    def _draw_straight(self, coords) -> None:
        r1, c1, r2, c2 = coords
        # pointing up or down doesn't matter
        if r1 == r2:
            a, b = sorted((c1, c2))
            idx = np.arange(a, b + 1)
            self.data[idx, r1] += 1
        else:
            a, b = sorted((r1, r2))
            idx = np.arange(a, b + 1)
            self.data[c1, idx] += 1

    def _draw_diagonal(self, coords) -> None:
        r1, c1, r2, c2 = coords
        # print(f"{r1=}, {c1=}, {r2=}, {c2=}")
        if r1 < r2:
            if c1 < c2:  # ↘
                # print("↘")
                for d in range(r2 - r1 + 1):
                    a, b = c1 + d, r1 + d
                    # print(f"{a=}, {b=}")
                    self.data[a, b] += 1
            else:  # ↗
                # print("↗")
                for d in range(r2 - r1 + 1):
                    a, b = c1 - d, r1 + d
                    # print(f"{a=}, {b=}")
                    self.data[a, b] += 1
        else:  # r1 > r2
            if c1 < c2:  # ↙
                # print("↙")
                for d in range(r1 - r2 + 1):
                    a, b = c1 + d, r1 - d
                    # print(f"{a=}, {b=}")
                    self.data[a, b] += 1
            else:  # ↖
                # print("↖")
                for d in range(r1 - r2 + 1):
                    a, b = c1 - d, r1 - d
                    # print(f"{a=}, {b=}")
                    self.data[a, b] += 1

    def draw_route(self, coords: np.ndarray) -> None:
        if SeaFloor.is_diagonal(coords):
            self._draw_diagonal(coords)
        elif SeaFloor.is_vertical_or_horizontal(coords):
            self._draw_straight(coords)
        else:
            raise ValueError(f"Not a straigh line or a diagonal {coords=}")

    def __repr__(self) -> str:
        return repr(self.data)


def align_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Swap coordinates to make everything point the same way
    """

    def swap_if_wrong_direction(a, b):
        if a > b:
            return b, a
        return a, b

    r1, c1, r2, c2 = coords
    r1, r2 = swap_if_wrong_direction(r1, r2)
    c1, c2 = swap_if_wrong_direction(c1, c2)
    return np.asarray([r1, c1, r2, c2])


def is_diagonal(coords: np.ndarray) -> bool:
    r1, c1, r2, c2 = coords
    delta_y, delta_x = (c2 - c1), (r2 - r1)
    return abs(delta_y) == abs(delta_x)


# %% script_part
if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv05.txt"
    data = read_data(data_path)

    map_shape = get_map_shape_from_data(data)

    # First star
    seafloor = SeaFloor(map_shape)
    straight_lines = np.asarray(
        [row for row in data if seafloor.is_vertical_or_horizontal(row)]
    )
    for row in straight_lines:
        seafloor.draw_route(row)
    answer = (seafloor.data > 1).sum()
    print(f"{answer=}")

    # Second star
    seafloor = SeaFloor(map_shape)
    for row in data:
        seafloor.draw_route(row)
    answer = (seafloor.data > 1).sum()
    print(f"{answer=}")
