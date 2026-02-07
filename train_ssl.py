import os
import json
import argparse
import yaml

import pandas as pd
import torch

from src.model import Encoder, Projector
from src.train_ssl import train_castcssl

def main(cfg: dict) -> None:
    out_dir = cfg["paths"]["out_dir"]
    run_name = cfg["paths"]["run_name"]

    train_tbl = pd.read_csv(os.path.join(out_dir, f"{run_name}_train_tbl.csv"))
    with open(os.path.join(out_dir, f"{run_name}_eeg_cols.json"), "r") as f:
        eeg_cols = json.load(f)
    with open(os.path.join(out_dir, f"{run_name}_meta.json"), "r") as f:
        meta = json.load(f)

    stage_order = meta["stage_order"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ device:", device)

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

    train_cfg = {
        "seed": tcfg["seed"],
        "fs": wcfg["fs"],
        "win_seconds": wcfg["win_seconds"],
        "stride_seconds": wcfg["stride_seconds"],
        "epochs": tcfg["epochs"],
        "steps_per_epoch": tcfg["steps_per_epoch"],
        "batch_size": tcfg["batch_size"],
        "temperature": tcfg["temperature"],
        "lambda_temporal": tcfg["lambda_temporal"],
        "p_stage_positive": cfg["pair_sampling"]["p_stage_positive"],
        "lr": tcfg["lr"],
        "weight_decay": tcfg["weight_decay"],
        "aug": cfg["augment"],
    }

    train_castcssl(enc, proj, train_tbl, stage_order, train_cfg, out_dir, run_name)
    print("✅ SSL training complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
