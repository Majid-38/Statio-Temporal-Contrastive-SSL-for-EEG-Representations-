"""
Window-level probe:
- embeds multiple random windows per recording
- trains a simple classifier at window level (optional)
- saves confusion matrix + UMAP before/after probe

This is useful for qualitative figures (window geometry), but your paper’s main
metric should remain subject-level (averaged embeddings).
"""
import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import umap

from src.model import Encoder
from src.windowing import open_memmap, sample_random_window_mm
from src.train_ssl import scale_np

def embed_windows(enc, tbl, mean, std, fs, win_seconds, n_wins_per_record, device):
    win_len = int(fs * win_seconds)
    X, y = [], []
    for i, r in tbl.reset_index(drop=True).iterrows():
        mm = open_memmap(r["eeg_path"])
        for _ in range(n_wins_per_record):
            w, _ = sample_random_window_mm(mm, win_len)
            w = scale_np(w, mean, std).astype(np.float32)
            x = torch.from_numpy(w[None, ...]).to(device)
            z = enc(x).detach().cpu().numpy()[0]
            X.append(z)
            y.append(int(r["label"]))
        if (i + 1) % 10 == 0:
            print(f"  embedded {i+1}/{len(tbl)} records")
    return np.asarray(X), np.asarray(y)

def save_umap(X, y, out_path, title):
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.25, metric="cosine", random_state=0)
    XY = reducer.fit_transform(X)
    plt.figure(figsize=(5.0, 4.2))
    plt.scatter(XY[y==0,0], XY[y==0,1], s=12, alpha=0.8, label="Low")
    plt.scatter(XY[y==1,0], XY[y==1,1], s=12, alpha=0.8, label="High")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main(cfg):
    out_dir = cfg["paths"]["out_dir"]
    run_name = cfg["paths"]["run_name"]
    train_tbl = pd.read_csv(os.path.join(out_dir, f"{run_name}_train_tbl.csv"))
    test_tbl  = pd.read_csv(os.path.join(out_dir, f"{run_name}_test_tbl.csv"))
    with open(os.path.join(out_dir, f"{run_name}_eeg_cols.json"), "r") as f:
        eeg_cols = json.load(f)

    ckpt_path = os.path.join(out_dir, f"{run_name}_ssl_epoch{cfg['train']['epochs']}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    enc = Encoder(
        in_ch=len(eeg_cols),
        d_model=mcfg["d_model"],
        gru_hidden=mcfg["gru_hidden"],
        attn_heads=mcfg["attn_heads"],
        emb_dim=mcfg["emb_dim"],
        dropout=mcfg["dropout"],
    ).to(device)
    enc.load_state_dict(ckpt["enc"], strict=True)
    enc.eval()

    mean = ckpt["mean"]; std = ckpt["std"]
    fs = cfg["windowing"]["fs"]
    win_seconds = cfg["windowing"]["win_seconds"]
    n_wins = 6  # can set 4, 6, 10 depending on figure quality

    print("➡️ Embedding windows (train)...")
    Xtr, ytr = embed_windows(enc, train_tbl, mean, std, fs, win_seconds, n_wins, device)
    print("➡️ Embedding windows (test)...")
    Xte, yte = embed_windows(enc, test_tbl, mean, std, fs, win_seconds, n_wins, device)

    # UMAP before probe
    save_umap(Xte, yte, os.path.join(out_dir, f"{run_name}_umap_windows_preprobe.png"),
              "Window embeddings (encoder only)")

    # LogReg probe
    sc = StandardScaler()
    Xtr2 = sc.fit_transform(Xtr)
    Xte2 = sc.transform(Xte)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(Xtr2, ytr)
    yhat = clf.predict(Xte2)
    yprob = clf.predict_proba(Xte2)[:, 1]

    # UMAP colored by predicted class (post probe)
    save_umap(Xte, yhat, os.path.join(out_dir, f"{run_name}_umap_windows_postprobe.png"),
              "Window embeddings (colored by LogReg prediction)")

    # Confusion
    cm_fig = os.path.join(out_dir, f"{run_name}_cm_window_probe.png")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay.from_predictions(yte, yhat, display_labels=["Low","High"], ax=ax, values_format="d")
    ax.set_title("Window-level probe (LogReg)")
    fig.tight_layout()
    fig.savefig(cm_fig, dpi=300)
    plt.close(fig)

    print("✅ Saved:")
    print(" -", cm_fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
