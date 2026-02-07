import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def main(cfg):
    out_dir = cfg["paths"]["out_dir"]
    run_name = cfg["paths"]["run_name"]

    # Plot baselines vs CAST-CSSL from CSVs if present
    baseline_csv = os.path.join(out_dir, f"{run_name}_supervised_handcrafted_baselines.csv")
    if os.path.exists(baseline_csv):
        df = pd.read_csv(baseline_csv)
        # keep common models (compact plot)
        keep = df[df["model"].isin(["LogReg","SVM-linear","SVM-RBF","kNN","RF"])]
        plt.figure(figsize=(6.5, 3.8))
        plt.bar(keep["model"], keep["acc"])
        plt.ylim(0, 1)
        plt.title("Handcrafted supervised baselines (subject-level)")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        outp = os.path.join(out_dir, f"{run_name}_handcrafted_baselines_acc.png")
        plt.savefig(outp, dpi=300)
        plt.close()
        print("✅ saved:", outp)

    # Ablations plot if present
    ab_csv = os.path.join(out_dir, f"{run_name}_ablations_summary.csv")
    if os.path.exists(ab_csv):
        ab = pd.read_csv(ab_csv)
        plt.figure(figsize=(6.8, 3.6))
        plt.bar(ab["id"], ab["acc"])
        plt.ylim(0, 1)
        plt.title("Ablations (subject-level accuracy)")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        outp = os.path.join(out_dir, f"{run_name}_ablations_acc.png")
        plt.savefig(outp, dpi=300)
        plt.close()
        print("✅ saved:", outp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
