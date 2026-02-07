import os
import json
import argparse
import yaml
import gc

import numpy as np
import pandas as pd

from src.windowing import converted_paths

def convert_csv_to_f32(csv_path: str, usecols: list, chunksize: int = 12000) -> None:
    bin_path, meta_path = converted_paths(csv_path)
    if os.path.exists(bin_path) and os.path.exists(meta_path):
        return

    rows_total = 0
    C = len(usecols)

    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False)
    with open(bin_path, "wb") as f:
        for chunk in reader:
            arr = chunk.to_numpy(dtype=np.float32, copy=False)
            if arr.ndim != 2 or arr.shape[1] != C:
                arr = np.asarray(arr, dtype=np.float32).reshape(-1, C)
            f.write(arr.tobytes(order="C"))
            rows_total += arr.shape[0]
            del chunk, arr
            gc.collect()

    with open(meta_path, "w") as g:
        json.dump({"rows": int(rows_total), "cols": int(C), "channels": list(usecols)}, g)

def main(cfg: dict) -> None:
    out_dir = cfg["paths"]["out_dir"]
    run_name = cfg["paths"]["run_name"]

    train_tbl = pd.read_csv(os.path.join(out_dir, f"{run_name}_train_tbl.csv"))
    test_tbl  = pd.read_csv(os.path.join(out_dir, f"{run_name}_test_tbl.csv"))
    with open(os.path.join(out_dir, f"{run_name}_eeg_cols.json"), "r") as f:
        eeg_cols = json.load(f)

    all_paths = pd.concat([train_tbl["eeg_path"], test_tbl["eeg_path"]], axis=0).tolist()
    print("➡️ Total files to cache:", len(all_paths))

    for i, p in enumerate(all_paths, 1):
        convert_csv_to_f32(p, eeg_cols, chunksize=12000)
        if i % 5 == 0 or i == len(all_paths):
            print(f"✅ cached {i}/{len(all_paths)}", flush=True)
            gc.collect()

    print("✅ memmaps created beside each eeg_clean.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
