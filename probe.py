import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .metrics import binary_metrics

def probe_subject_logreg(train_subj: pd.DataFrame, test_subj: pd.DataFrame):
    feat_cols = [c for c in train_subj.columns if c.startswith("e")]
    Xtr = train_subj[feat_cols].to_numpy()
    ytr = train_subj["label"].to_numpy().astype(int)
    Xte = test_subj[feat_cols].to_numpy()
    yte = test_subj["label"].to_numpy().astype(int)

    sc = StandardScaler()
    Xtr2 = sc.fit_transform(Xtr)
    Xte2 = sc.transform(Xte)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(Xtr2, ytr)

    yhat = clf.predict(Xte2)
    yprob = clf.predict_proba(Xte2)[:, 1]
    return binary_metrics(yte, yhat, yprob)

def stage_knn_emergence(train_ss: pd.DataFrame, test_ss: pd.DataFrame, k: int = 5):
    feat_cols = [c for c in train_ss.columns if c.startswith("e")]
    Xtr = train_ss[feat_cols].to_numpy()
    ytr = train_ss["stage"].astype(str).to_numpy()
    Xte = test_ss[feat_cols].to_numpy()
    yte = test_ss["stage"].astype(str).to_numpy()

    k = min(k, len(Xtr)) if len(Xtr) > 0 else 1
    if len(np.unique(ytr)) < 2 or len(Xtr) < k:
        return None

    knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    pred = knn.predict(Xte)
    return float((pred == yte).mean())
