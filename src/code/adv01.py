from pathlib import Path


def read_data(pth: Path) -> list[int]:
    with pth.open() as fid:
        return [int(el) for el in fid.readlines()]


def count_increasing(data: list[int]) -> int:
    increasing_count = 0
    i = data[0]
    for j in data[1:]:
        if j > i:
            increasing_count += 1
        i = j
    return increasing_count


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "adv01.txt"
    data = read_data(data_path)
    n_incr = count_increasing(data)
    print(f"{n_incr} increasing observations")
