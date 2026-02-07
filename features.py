import numpy as np
from scipy.signal import welch

def hjorth_params(x: np.ndarray):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x)
    var1 = np.var(dx)
    var2 = np.var(ddx) if len(ddx) > 0 else 0.0
    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt((var2 / (var1 + 1e-12)) / (var1 / (var0 + 1e-12) + 1e-12))
    return activity, mobility, complexity

def bandpower(freqs, psd, f_lo, f_hi):
    m = (freqs >= f_lo) & (freqs < f_hi)
    return float(np.trapz(psd[m], freqs[m])) if np.any(m) else 0.0

def extract_window_features(w_NC: np.ndarray, fs: int, bands: dict):
    """
    Handcrafted per-channel features:
    - mean, std
    - Hjorth activity/mobility/complexity
    - bandpowers (delta/theta/alpha/beta)
    - ratios: theta/beta, alpha/beta, delta/beta
    """
    feats = []
    for c in range(w_NC.shape[1]):
        x = w_NC[:, c]

        feats.extend([float(np.mean(x)), float(np.std(x))])

        a, m, comp = hjorth_params(x)
        feats.extend([float(a), float(m), float(comp)])

        f, Pxx = welch(x, fs=fs, nperseg=min(len(x), fs * 2))
        bp = {bn: bandpower(f, Pxx, lo, hi) for bn, (lo, hi) in bands.items()}
        feats.extend([bp["delta"], bp["theta"], bp["alpha"], bp["beta"]])

        feats.extend([
            float(bp["theta"] / (bp["beta"] + 1e-12)),
            float(bp["alpha"] / (bp["beta"] + 1e-12)),
            float(bp["delta"] / (bp["beta"] + 1e-12)),
        ])

    return np.asarray(feats, dtype=np.float32)
