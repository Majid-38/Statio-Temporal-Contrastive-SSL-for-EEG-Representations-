import os
import json
import numpy as np

def converted_paths(csv_path: str):
    folder = os.path.dirname(csv_path)
    return (
        os.path.join(folder, "eeg_clean.f32"),
        os.path.join(folder, "eeg_clean.meta.json"),
    )

def open_memmap(csv_path: str):
    bin_path, meta_path = converted_paths(csv_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    rows, cols = int(meta["rows"]), int(meta["cols"])
    return np.memmap(bin_path, dtype=np.float32, mode="r", shape=(rows, cols))

def sample_random_window_mm(mm, win_len: int):
    T = mm.shape[0]
    if T <= win_len:
        x = np.asarray(mm[:], dtype=np.float32)
        pad = win_len - T
        return np.pad(x, ((0,pad),(0,0)), mode="edge"), 0
    s = np.random.randint(0, T - win_len + 1)
    return np.asarray(mm[s:s+win_len], dtype=np.float32), int(s)

def sample_adjacent_window_mm(mm, start: int, win_len: int, step: int):
    T = mm.shape[0]
    if T <= win_len:
        x = np.asarray(mm[:], dtype=np.float32)
        pad = win_len - T
        return np.pad(x, ((0,pad),(0,0)), mode="edge")
    s2 = min(start + step, T - win_len)
    return np.asarray(mm[s2:s2+win_len], dtype=np.float32)
