import numpy as np
import pandas as pd
import torch
import gc

from .windowing import open_memmap, sample_random_window_mm
from .train_ssl import scale_np

@torch.no_grad()
def encode_recording(enc, csv_path, mean, std, win_len, n_windows, device):
    mm = open_memmap(csv_path)
    zs = []
    for _ in range(n_windows):
        w, _ = sample_random_window_mm(mm, win_len)
        w = scale_np(w, mean, std).astype(np.float32)
        x = torch.from_numpy(w[None, ...]).to(device)
        z = enc(x).detach().cpu().numpy()[0]
        zs.append(z)
    del mm
    gc.collect()
    return np.mean(np.stack(zs), axis=0)

@torch.no_grad()
def build_subject_tables(enc, tbl: pd.DataFrame, mean, std, fs: int, win_seconds: int, n_windows: int, device: str):
    win_len = int(fs * win_seconds)
    rows_ss = []
    for _, r in tbl.iterrows():
        z = encode_recording(enc, r["eeg_path"], mean, std, win_len, n_windows, device)
        rows_ss.append({
            "subject": r["subject"],
            "sid": int(r["sid"]),
            "stage": str(r["stage"]),
            "label": int(r["label"]),
            **{f"e{k}": float(z[k]) for k in range(z.shape[0])},
        })
    ss = pd.DataFrame(rows_ss)

    feat_cols = [c for c in ss.columns if c.startswith("e")]
    subj = ss.groupby(["subject", "sid", "label"], as_index=False)[feat_cols].mean()
    return subj, ss
