import os
import time
import random
import gc
from typing import Dict, Tuple

import numpy as np
import torch

from .windowing import open_memmap, sample_random_window_mm, sample_adjacent_window_mm
from .augment import aug_view
from .losses import simclr_loss, temporal_consistency

# Used by ablation runner. Defaults to CAST-CSSL full behavior.
GLOBAL_ABLATION_FLAGS = {"stage_on": True, "temp_on": True}

def seed_everything(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def fit_scaler(train_tbl, win_len: int, n_files: int = 12, n_wins: int = 5):
    paths = train_tbl["eeg_path"].tolist()
    random.shuffle(paths)
    paths = paths[:min(len(paths), n_files)]

    sum_ = None
    sumsq_ = None
    count = 0

    for p in paths:
        mm = open_memmap(p)
        for _ in range(n_wins):
            w, _ = sample_random_window_mm(mm, win_len)
            if sum_ is None:
                sum_ = w.sum(axis=0, dtype=np.float64)
                sumsq_ = (w*w).sum(axis=0, dtype=np.float64)
            else:
                sum_ += w.sum(axis=0, dtype=np.float64)
                sumsq_ += (w*w).sum(axis=0, dtype=np.float64)
            count += w.shape[0]
        del mm
        gc.collect()

    mean = (sum_ / max(1, count)).astype(np.float32)[None, :]
    var = (sumsq_ / max(1, count) - (mean[0].astype(np.float64)**2)).astype(np.float32)
    std = (np.sqrt(np.maximum(var, 1e-8)) + 1e-6).astype(np.float32)[None, :]
    return mean, std

def scale_np(w: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (w - mean) / std

def build_stage_index(train_tbl, stage_order):
    by_stage = {}
    for i, r in train_tbl.reset_index(drop=True).iterrows():
        by_stage.setdefault(str(r["stage"]).lower(), []).append(i)
    stages = [s for s in stage_order if s in by_stage] + [s for s in by_stage.keys() if s not in stage_order]
    return by_stage, stages

class MemmapCache:
    def __init__(self, tbl):
        self.tbl = tbl.reset_index(drop=True)
        self._key = None
        self._mm = None

    def get(self, idx: int):
        p = self.tbl.loc[idx, "eeg_path"]
        if self._key == p and self._mm is not None:
            return self._mm
        self._mm = None
        self._key = None
        gc.collect()
        self._mm = open_memmap(p)
        self._key = p
        return self._mm

def sample_pair(
    train_tbl,
    cache: MemmapCache,
    by_stage,
    stages,
    win_len: int,
    step: int,
    p_stage_positive: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixed positive sampling:
    - stage-aware (prob p): same stage, different recordings
    - temporal continuity (prob 1-p): adjacent windows within same recording
    Ablations toggle these via GLOBAL_ABLATION_FLAGS.
    """
    stage_on = GLOBAL_ABLATION_FLAGS.get("stage_on", True)
    temp_on  = GLOBAL_ABLATION_FLAGS.get("temp_on", True)

    if stage_on and (random.random() < p_stage_positive) and len(stages) > 0:
        st = random.choice(stages)
        idxs = by_stage[st]
        a, b = (random.sample(idxs, 2) if len(idxs) >= 2 else (idxs[0], idxs[0]))
        mma, mmb = cache.get(a), cache.get(b)
        w1, _ = sample_random_window_mm(mma, win_len)
        w2, _ = sample_random_window_mm(mmb, win_len)
        return w1, w2

    if temp_on:
        i = random.randint(0, len(train_tbl) - 1)
        mm = cache.get(i)
        w1, s1 = sample_random_window_mm(mm, win_len)
        w2 = sample_adjacent_window_mm(mm, s1, win_len, step)
        return w1, w2

    # fallback: random crops (SimCLR-like without the extra structure)
    i = random.randint(0, len(train_tbl) - 1)
    mm = cache.get(i)
    w1, _ = sample_random_window_mm(mm, win_len)
    w2, _ = sample_random_window_mm(mm, win_len)
    return w1, w2

def train_castcssl(enc, proj, train_tbl, stage_order, cfg: Dict, out_dir: str, run_name: str):
    device = next(enc.parameters()).device
    use_amp = (device.type == "cuda")
    seed_everything(cfg["seed"], use_cuda=use_amp)

    fs = cfg["fs"]
    win_len = int(fs * cfg["win_seconds"])
    step = int(fs * cfg["stride_seconds"])

    mean, std = fit_scaler(train_tbl, win_len=win_len, n_files=12, n_wins=5)
    by_stage, stages = build_stage_index(train_tbl, stage_order)
    cache = MemmapCache(train_tbl)

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(proj.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    enc.train()
    proj.train()

    for ep in range(1, cfg["epochs"] + 1):
        losses = []
        t0 = time.time()

        for step_i in range(cfg["steps_per_epoch"]):
            w1, w2 = [], []
            for _ in range(cfg["batch_size"]):
                a, b = sample_pair(train_tbl, cache, by_stage, stages, win_len, step, cfg["p_stage_positive"])
                w1.append(scale_np(a, mean, std).astype(np.float32))
                w2.append(scale_np(b, mean, std).astype(np.float32))

            x1 = torch.from_numpy(np.stack(w1)).to(device)
            x2 = torch.from_numpy(np.stack(w2)).to(device)

            v1 = torch.stack([aug_view(xx.clone(), **cfg["aug"]) for xx in x1], dim=0)
            v2 = torch.stack([aug_view(xx.clone(), **cfg["aug"]) for xx in x2], dim=0)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                z1 = enc(v1)
                z2 = enc(v2)
                p1 = proj(z1)
                p2 = proj(z2)

                ls = simclr_loss(p1, p2, temperature=cfg["temperature"])
                lt = temporal_consistency(z1, z2)
                loss = ls + cfg["lambda_temporal"] * lt

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu().item()))
            if (step_i + 1) % 10 == 0:
                print(f"    step {step_i+1}/{cfg['steps_per_epoch']} | loss={losses[-1]:.4f}", flush=True)

        ckpt_path = os.path.join(out_dir, f"{run_name}_ssl_epoch{ep}.pt")
        torch.save(
            {"enc": enc.state_dict(), "proj": proj.state_dict(), "mean": mean, "std": std, "cfg": cfg},
            ckpt_path,
        )
        print(f"[SSL] ep {ep}/{cfg['epochs']} loss={np.mean(losses):.4f} time={time.time()-t0:.1f}s | saved {ckpt_path}")

    return mean, std
