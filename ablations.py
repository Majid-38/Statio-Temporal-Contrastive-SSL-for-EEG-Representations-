"""
Ablations:
We reproduce the same architecture, but toggle:
- aug_on
- stage_on
- temp_on (temporal consistency loss + temporal adjacency pairing)
and export embeddings + probe metrics.

This script assumes memmaps and train/test tables already exist.
"""
import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from src.model import Encoder, Projector
from src.train_ssl import train_castcssl
from src.embed import build_subject_tables
from src.probe import probe_subject_logreg, stage_knn_emergence

def run_one(cfg, ab, train_tbl, test_tbl, eeg_cols, stage_order):
    out_dir = cfg["paths"]["out_dir"]
    base_run = cfg["paths"]["run_name"]
    ab_id = ab["id"]
    run_name = f"{base_run}_{ab_id}"

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
    proj = Projector(emb_dim=mcfg["emb_dim"], proj_dim=mcfg["proj_dim"]).to(device)

    tcfg = cfg["train"]
    wcfg = cfg["windowing"]

    # augmentation toggle
    aug_cfg = cfg["augment"].copy()
    if not ab.get("aug_on", True):
        aug_cfg = dict(jitter_sigma=0.0, scale_std=0.0, time_mask_frac=0.0, channel_drop_prob=0.0)

    # stage/temp toggles
    stage_on = bool(ab.get("stage_on", True))
    temp_on  = bool(ab.get("temp_on", True))

    train_cfg = {
        "seed": tcfg["seed"],
        "fs": wcfg["fs"],
        "win_seconds": wcfg["win_seconds"],
        "stride_seconds": wcfg["stride_seconds"],
        "epochs": tcfg["epochs"],
        "steps_per_epoch": tcfg["steps_per_epoch"],
        "batch_size": tcfg["batch_size"],
        "temperature": tcfg["temperature"],
        "lambda_temporal": (tcfg["lambda_temporal"] if temp_on else 0.0),
        "p_stage_positive": (cfg["pair_sampling"]["p_stage_positive"] if stage_on else 0.0),
        "lr": tcfg["lr"],
        "weight_decay": tcfg["weight_decay"],
        "aug": aug_cfg,
        # NOTE: stage_on/temp_on are handled inside train_ssl sampling
        "stage_on": stage_on,
        "temp_on": temp_on,
    }

    # train with toggles
    # We store stage_on/temp_on in cfg and the trainer uses them when sampling pairs
    from src import train_ssl as _tssl
    _tssl.GLOBAL_ABLATION_FLAGS = {"stage_on": stage_on, "temp_on": temp_on}
    mean, std = train_castcssl(enc, proj, train_tbl, stage_order, train_cfg, out_dir, run_name)

    # extract embeddings
    train_subj, train_ss = build_subject_tables(enc, train_tbl, mean, std, wcfg["fs"], wcfg["win_seconds"], n_windows=4, device=str(device))
    test_subj,  test_ss  = build_subject_tables(enc, test_tbl,  mean, std, wcfg["fs"], wcfg["win_seconds"], n_windows=4, device=str(device))

    train_subj.to_csv(os.path.join(out_dir, f"{run_name}_train_subject_mean.csv"), index=False)
    test_subj.to_csv(os.path.join(out_dir, f"{run_name}_test_subject_mean.csv"), index=False)
    train_ss.to_csv(os.path.join(out_dir, f"{run_name}_train_subject_stage_mean.csv"), index=False)
    test_ss.to_csv(os.path.join(out_dir, f"{run_name}_test_subject_stage_mean.csv"), index=False)

    # evaluate
    knn = stage_knn_emergence(train_ss, test_ss, k=5)
    res = probe_subject_logreg(train_subj, test_subj)
    return {"id": ab_id, "stage_knn": knn, **{k: v for k, v in res.items() if k != "cm"}}

def main(cfg):
    out_dir = cfg["paths"]["out_dir"]
    base_run = cfg["paths"]["run_name"]

    train_tbl = pd.read_csv(os.path.join(out_dir, f"{base_run}_train_tbl.csv"))
    test_tbl  = pd.read_csv(os.path.join(out_dir, f"{base_run}_test_tbl.csv"))
    with open(os.path.join(out_dir, f"{base_run}_eeg_cols.json"), "r") as f:
        eeg_cols = json.load(f)
    with open(os.path.join(out_dir, f"{base_run}_meta.json"), "r") as f:
        meta = json.load(f)
    stage_order = meta["stage_order"]

    summary = []
    for ab in cfg["ablations"]:
        print("\n===============================")
        print("Running:", ab["id"])
        print("===============================")
        row = run_one(cfg, ab, train_tbl, test_tbl, eeg_cols, stage_order)
        print(row)
        summary.append(row)

    df = pd.DataFrame(summary)
    out_path = os.path.join(out_dir, f"{base_run}_ablations_summary.csv")
    df.to_csv(out_path, index=False)
    print("✅ Saved ablation summary:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
