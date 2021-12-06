# %% imports
from pathlib import Path
import numpy as np
import io
from typing import Optional

# %% functions


def read_data(pth: Path) -> np.ndarray:
    """
    Returns an ndarray in the format c1, r1, c2, r2.
    """
    with pth.open() as fid:
        raw = io.StringIO(fid.read().replace(" -> ", ","))
    data = np.loadtxt(raw, delimiter=",", dtype=int)
    # return data[:, [0, 2, 1, 3]]
    return data


def is_vertical_or_horizontal(coords: np.ndarray) -> bool:
    x1, y1, x2, y2 = coords
    return (x1 == x2) | (y1 == y2)


def align_coordinates(coords: np.ndarray) -> np.ndarray:
    def swap_if_wrong_direction(a, b):
        if a > b:
            return b, a
        return a, b

    x1, y1, x2, y2 = coords
    x1, x2 = swap_if_wrong_direction(x1, x2)
    y1, y2 = swap_if_wrong_direction(y1, y2)
    return np.asarray([x1, y1, x2, y2])


def is_diagonal(coords: np.ndarray) -> bool:
    x1, y1, x2, y2 = coords
    delta_y, delta_x = (y2 - y1), (x2 - x1)
    return abs(delta_y) == abs(delta_x)


def make_simple_map(
    data: np.ndarray, map_shape: Optional[tuple[int, int]] = None
) -> np.ndarray:
    map_shape = (
        (data[:, :2].max() + 1, data[:, 2:].max() + 1)
        if map_shape is None
        else map_shape
    )
    seafloor = np.zeros(map_shape, dtype=int)
    for row in data:
        c1, r1, c2, r2 = align_coordinates(row)
        r_idx = np.arange(r1, r2 + 1, dtype=int)
        c_idx = np.arange(c1, c2 + 1, dtype=int)
        seafloor[r_idx, c_idx] += 1
    return seafloor


# %% script_part
if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv05.txt"
    data = read_data(data_path)

    # First star
    straight_lines = np.asarray([row for row in data if is_vertical_or_horizontal(row)])
    seafloor = make_simple_map(straight_lines)
    answer = (seafloor > 1).sum()
    print(f"{answer=}")

# %%
