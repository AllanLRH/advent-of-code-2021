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


def is_vertical_or_horizontal(data: np.ndarray) -> np.ndarray:
    same_x = data[:, 0] == data[:, 1]
    same_y = data[:, 2] == data[:, 3]
    same_mask = same_x | same_y
    return same_mask


def make_simple_map(
    data: np.ndarray, map_shape: Optional[tuple[int, int]] = None
) -> np.ndarray:
    map_shape = (
        (data[:, :2].max() + 1, data[:, 2:].max() + 1)
        if map_shape is None
        else map_shape
    )
    seafloor = np.zeros(map_shape, dtype=int)
    for c1, r1, c2, r2 in data:
        if c1 == c2 or r1 == r2:
            cc1, cc2 = (c1, c2 + 1) if c1 < c2 else (c2, c1 + 1)
            rr1, rr2 = (r1, r2 + 1) if r1 < r2 else (r2, r1 + 1)
            ind_c, ind_r = np.arange(cc1, cc2), np.arange(rr1, rr2)
            seafloor[ind_r, ind_c] += 1
            # print(f"{(cc1, cc2, rr1, rr2)=}")
        else:
            continue
            # print(f"Skipping {(c1, r1)}, {c2, r2}")
    return seafloor


# %% script_part
if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv05.txt"
    data = read_data(data_path)

    # First star
    seafloor = make_simple_map(data)
    answer = (seafloor > 1).sum()
    print(f"{answer=}")

# %%
