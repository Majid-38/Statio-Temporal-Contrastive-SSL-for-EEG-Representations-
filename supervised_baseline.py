import os
import json
import argparse
import yaml
import gc

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from src.windowing import open_memmap, sample_random_window_mm
from src.features import extract_window_features
from src.metrics import binary_metrics

def subject_feature_table(tbl, eeg_cols, fs, win_seconds, bands, n_wins_per_record=6):
    win_len = int(fs * win_seconds)
    rows = []
    for sid, g in tbl.groupby("sid"):
        feats = []
        for _, r in g.iterrows():
            mm = open_memmap(r["eeg_path"])
            for _ in range(n_wins_per_record):
                w, _ = sample_random_window_mm(mm, win_len)
                feats.append(extract_window_features(w, fs=fs, bands=bands))
            del mm
            gc.collect()
        x = np.mean(np.stack(feats), axis=0)
        rows.append({"sid": int(sid), "label": int(g["label"].iloc[0]), **{f"f{k}": float(x[k]) for k in range(len(x))}})
    return pd.DataFrame(rows)

def fit_eval(name, clf, Xtr, ytr, Xte, yte):
    sc = StandardScaler()
    Xtr2 = sc.fit_transform(Xtr)
    Xte2 = sc.transform(Xte)
    clf.fit(Xtr2, ytr)
    yhat = clf.predict(Xte2)
    yprob = clf.predict_proba(Xte2)[:, 1] if hasattr(clf, "predict_proba") else None
    out = binary_metrics(yte, yhat, yprob)
    out["model"] = name
    out["n_feat"] = int(Xtr.shape[1])
    return out

def main(cfg):
    out_dir = cfg["paths"]["out_dir"]
    run_name = cfg["paths"]["run_name"]

    train_tbl = pd.read_csv(cfg["paths"]["train_tbl_csv"])
    test_tbl  = pd.read_csv(cfg["paths"]["test_tbl_csv"])
    with open(os.path.join(out_dir, f"{run_name}_eeg_cols.json"), "r") as f:
        eeg_cols = json.load(f)

    fs = cfg["features"]["fs"]
    win_seconds = cfg["features"]["win_seconds"]
    bands = cfg["features"]["bands"]
    n_wins = int(cfg["features"].get("n_windows_per_recording", 6))

    print("➡️ Extracting handcrafted features (train subjects)...")
    tr = subject_feature_table(train_tbl, eeg_cols, fs, win_seconds, bands, n_wins_per_record=n_wins)
    print("➡️ Extracting handcrafted features (test subjects)...")
    te = subject_feature_table(test_tbl, eeg_cols, fs, win_seconds, bands, n_wins_per_record=n_wins)

    feat_cols = [c for c in tr.columns if c.startswith("f")]
    Xtr = tr[feat_cols].to_numpy()
    ytr = tr["label"].to_numpy().astype(int)
    Xte = te[feat_cols].to_numpy()
    yte = te["label"].to_numpy().astype(int)

    models = [
        ("LogReg", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ("SVM-linear", SVC(kernel="linear", probability=True, class_weight="balanced")),
        ("SVM-RBF", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ("kNN", KNeighborsClassifier(n_neighbors=5)),
        ("RF", RandomForestClassifier(n_estimators=300, random_state=0)),
    ]

    results = []
    for name, clf in models:
        res = fit_eval(name, clf, Xtr, ytr, Xte, yte)
        results.append(res)
        print(res)

    # ANOVA feature selection sweep
    ks = [10, 20, 30, 50, 80, 120]
    for k in ks:
        k = min(k, Xtr.shape[1])
        selector = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
        Xtr_k = selector.transform(Xtr)
        Xte_k = selector.transform(Xte)
        for name, clf in models:
            res = fit_eval(f"{name}+ANOVA{k}", clf, Xtr_k, ytr, Xte_k, yte)
            results.append(res)

    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, f"{run_name}_supervised_handcrafted_baselines.csv")
    df.to_csv(out_path, index=False)
    print("✅ Saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
