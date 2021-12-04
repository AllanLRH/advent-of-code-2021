from pathlib import Path

import numpy as np
from scipy.stats import mode


def read_data(pth: Path) -> np.ndarray:
    with pth.open() as fid:
        return np.asarray([[int(char) for char in line.strip()] for line in fid])


def bit_criteria(col: np.ndarray, gas: str) -> np.ndarray:
    assert col.squeeze().ndim == 1, f"Not a vector, {col.shape=}, {col=}"
    # Output sorted accodring to the bit variable
    bit, cnt = np.unique(col, return_counts=True)
    if len(bit) == 1:
        col_mode = int(bit.squeeze())
    else:
        col_mode = bit[1] if cnt[1] >= cnt[0] else bit[0]
    if gas == "oxygen":
        return col == col_mode
    return col != col_mode


def get_rate(data: np.ndarray, gas: str, col_idx: int = 0) -> int:
    # print(col_idx, data, sep="\n", end="\n\n")
    if gas not in ("oxygen", "co2"):
        raise ValueError(f"Invalid gas {gas=}")
    if data.squeeze().ndim == 1:
        return data.squeeze()
    mask = bit_criteria(data[:, col_idx], gas)
    return get_rate(data[mask, :], gas, col_idx + 1)


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv03.txt"
    data = read_data(data_path)

    # First star
    mode_data = mode(data, axis=0)[0].squeeze()
    print(f"{mode_data=}")
    gamma_rate = int("".join([str(el) for el in mode_data]), base=2)
    epsilon_rate = int("".join([str(el ^ 1) for el in mode_data]), base=2)
    power_rate = gamma_rate * epsilon_rate
    print(f"{epsilon_rate=}")
    print(f"{gamma_rate=}")
    print(f"{power_rate=}")

    # Second star
    rate_co2_binary = get_rate(data, "co2")
    rate_oxygen_binary = get_rate(data, "oxygen")
    print(f"{rate_co2_binary=}, {rate_oxygen_binary=}")
    rate_co2 = int("".join(rate_co2_binary.astype(str)), 2)
    rate_oxygen = int("".join(rate_oxygen_binary.astype(str)), 2)
    print(f"{rate_co2=}, {rate_oxygen=}")
    print(f"Answer: {rate_co2 * rate_oxygen}")
