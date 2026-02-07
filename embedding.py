import os
import argparse
import yaml
import pandas as pd
import torch

from src.model import Encoder
from src.embed import build_subject_tables

def main(cfg: dict) -> None:
    out_dir = cfg["paths"]["out_dir"]
    run_name = cfg["paths"]["run_name"]

    train_tbl = pd.read_csv(os.path.join(out_dir, f"{run_name}_train_tbl.csv"))
    test_tbl  = pd.read_csv(os.path.join(out_dir, f"{run_name}_test_tbl.csv"))

    ckpt_path = os.path.join(out_dir, f"{run_name}_ssl_epoch{cfg['train']['epochs']}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ device:", device)

    mcfg = cfg["model"]
    enc = Encoder(
        in_ch=len(cfg.get("eeg_cols_override", [])) or 9,  # overwritten below if ckpt has channels
        d_model=mcfg["d_model"],
        gru_hidden=mcfg["gru_hidden"],
        attn_heads=mcfg["attn_heads"],
        emb_dim=mcfg["emb_dim"],
        dropout=mcfg["dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc.load_state_dict(ckpt["enc"], strict=True)
    enc.eval()

    mean = ckpt["mean"]
    std  = ckpt["std"]

    fs = cfg["windowing"]["fs"]
    win_seconds = cfg["windowing"]["win_seconds"]

    train_subj, train_ss = build_subject_tables(enc, train_tbl, mean, std, fs, win_seconds, n_windows=4, device=str(device))
    test_subj,  test_ss  = build_subject_tables(enc, test_tbl,  mean, std, fs, win_seconds, n_windows=4, device=str(device))

    train_subj.to_csv(os.path.join(out_dir, f"{run_name}_train_subject_mean.csv"), index=False)
    test_subj.to_csv(os.path.join(out_dir, f"{run_name}_test_subject_mean.csv"), index=False)
    train_ss.to_csv(os.path.join(out_dir, f"{run_name}_train_subject_stage_mean.csv"), index=False)
    test_ss.to_csv(os.path.join(out_dir, f"{run_name}_test_subject_stage_mean.csv"), index=False)

    print("✅ Saved embeddings to:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
