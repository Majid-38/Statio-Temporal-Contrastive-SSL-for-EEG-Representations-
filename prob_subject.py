import os
import argparse
import yaml
import pandas as pd

from src.probe import probe_subject_logreg, stage_knn_emergence
from src.viz import save_confusion

def main(cfg: dict) -> None:
    out_dir = cfg["paths"]["out_dir"]
    run_name = cfg["paths"]["run_name"]

    train_subj = pd.read_csv(os.path.join(out_dir, f"{run_name}_train_subject_mean.csv"))
    test_subj  = pd.read_csv(os.path.join(out_dir, f"{run_name}_test_subject_mean.csv"))
    train_ss   = pd.read_csv(os.path.join(out_dir, f"{run_name}_train_subject_stage_mean.csv"))
    test_ss    = pd.read_csv(os.path.join(out_dir, f"{run_name}_test_subject_stage_mean.csv"))

    knn_acc = stage_knn_emergence(train_ss, test_ss, k=5)
    print("[Stage kNN acc test]:", knn_acc)

    res = probe_subject_logreg(train_subj, test_subj)
    print("[Subject-level LogReg probe]:")
    print({k: v for k, v in res.items() if k != "cm"})

    fig_path = os.path.join(out_dir, f"{run_name}_cm_subject_probe.png")
    save_confusion(res["cm"], labels=["Low", "High"], out_path=fig_path, title="Subject-level resilience (LogReg)")
    print("✅ saved:", fig_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
