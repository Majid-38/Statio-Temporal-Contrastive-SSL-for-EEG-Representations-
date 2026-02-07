import os
import re
import json
from typing import List
import pandas as pd

def parse_subject_id(subject_str: str) -> int:
    m = re.search(r"(\d+)", str(subject_str))
    return int(m.group(1)) if m else -1

def infer_numeric_cols_from_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, nrows=30)
    drop_like = {"time","timestamp","index","sample","marker","event","label"}
    cols = []
    for c in df.columns:
        if str(c).strip().lower() in drop_like:
            continue
        try:
            pd.to_numeric(df[c], errors="raise")
            cols.append(c)
        except Exception:
            pass
    if len(cols) < 2:
        cols = [c for c in df.columns if str(c).strip().lower() not in drop_like]
    return cols

def build_table_eeg(dataset_root: str, eeg_filename: str = "eeg_clean.csv") -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(dataset_root):
        if "preprocessed" not in root.lower():
            continue
        files_low = {f.lower(): f for f in files}
        if eeg_filename.lower() not in files_low:
            continue

        eeg_path = os.path.join(root, files_low[eeg_filename.lower()])
        parts = eeg_path.replace("\\", "/").split("/")

        subj = next((p for p in parts if p.startswith("subject_")), None)
        if subj is None:
            continue

        sid = parse_subject_id(subj)

        stage = "unknown"
        try:
            si = parts.index(subj)
            stage = parts[si + 1].lower() if si + 1 < len(parts) else "unknown"
        except Exception:
            pass

        rows.append({"subject": subj, "sid": sid, "stage": stage, "eeg_path": eeg_path})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No EEG files found. Expected .../subject_XX/<stage>/preprocessed/eeg_clean.csv")
    return df.sort_values(["sid", "stage"]).reset_index(drop=True)

def merge_labels(tbl: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    lab = pd.read_csv(labels_csv)

    sid_col = next((c for c in ["sid","subject_id","subject","id"] if c in lab.columns), None)
    label_col = next((c for c in ["label_new","label","y","target"] if c in lab.columns), None)
    if sid_col is None or label_col is None:
        raise ValueError(f"labels CSV must have subject id + label columns. Found: {lab.columns.tolist()}")

    lab = lab[[sid_col, label_col]].rename(columns={sid_col:"sid", label_col:"label"})
    lab["sid"] = lab["sid"].astype(int)
    lab["label"] = lab["label"].astype(int)

    out = tbl.merge(lab, on="sid", how="left")
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)

def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
