from pathlib import Path
import numpy as np


def read_data(pth: Path) -> np.ndarray:
    with pth.open() as fid:
        return np.asarray([int(el) for el in fid.readlines()])


def count_increasing(data: np.ndarray) -> int:
    return (np.diff(data, n=1) > 0).sum()


def apply_window(data: list[int], window_size: int) -> int:
    return np.convolve(data, np.ones(window_size), mode="valid")


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv01.txt"
    data = read_data(data_path)

    n_incr = count_increasing(data)
    print(f"{n_incr} increasing observations")

    windowed_data = apply_window(data, 3)
    n_incr_windowed = count_increasing(windowed_data)
    print(f"{n_incr_windowed} increasing observations for windowed data")
