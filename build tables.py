import os
import argparse
import yaml
from sklearn.model_selection import train_test_split

from src.io import (
    build_table_eeg,
    merge_labels,
    infer_numeric_cols_from_csv,
    save_json,
)

def main(cfg: dict) -> None:
    paths = cfg["paths"]
    out_dir = paths["out_dir"]
    run_name = paths["run_name"]
    os.makedirs(out_dir, exist_ok=True)

    tbl = build_table_eeg(paths["dataset_root"], eeg_filename=cfg["data"]["eeg_filename"])
    tbl = merge_labels(tbl, paths["labels_csv"])

    subjects = sorted(tbl["subject"].unique())
    subj_label = [int(tbl[tbl["subject"] == s]["label"].iloc[0]) for s in subjects]

    train_subs, test_subs = train_test_split(
        subjects,
        test_size=0.25,
        random_state=123,
        shuffle=True,
        stratify=subj_label,
    )

    train_tbl = tbl[tbl["subject"].isin(train_subs)].copy().reset_index(drop=True)
    test_tbl  = tbl[tbl["subject"].isin(test_subs)].copy().reset_index(drop=True)

    eeg_cols = infer_numeric_cols_from_csv(train_tbl.iloc[0]["eeg_path"])

    train_tbl.to_csv(os.path.join(out_dir, f"{run_name}_train_tbl.csv"), index=False)
    test_tbl.to_csv(os.path.join(out_dir, f"{run_name}_test_tbl.csv"), index=False)
    save_json(os.path.join(out_dir, f"{run_name}_eeg_cols.json"), eeg_cols)
    save_json(os.path.join(out_dir, f"{run_name}_meta.json"), {"stage_order": cfg["data"]["stage_order"]})

    print("✅ Saved train/test tables + channel list to:", out_dir)
    print("  train records:", len(train_tbl), "| test records:", len(test_tbl))
    print("  channels:", len(eeg_cols), eeg_cols[:12])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
