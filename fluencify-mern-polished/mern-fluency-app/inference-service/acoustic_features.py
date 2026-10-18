"""STEP 4 / Branch B - interpretable acoustic + temporal features.

Groups: pauses, speech rate, rhythm, repetition (MFCC self-similarity),
prolongation (long spectrally-stable runs), energy, spectral, pitch, MFCC stats.
"""
from itertools import groupby
import numpy as np
import librosa
from config import SR

HOP, WIN, MIN_PAUSE = 160, 400, 0.15
FPS = SR / HOP


def _runs(mask):
    out, i = [], 0
    for val, grp in groupby(mask):
        n = len(list(grp)); out.append((i, n, bool(val))); i += n
    return out


def _speech_mask(db):
    noise, peak = np.percentile(db, 10), np.percentile(db, 95)
    thr = min(max(noise + 6.0, peak - 35.0), peak - 12.0)
    mask = db > thr
    for s, n, v in _runs(mask):                 # drop blips < 30 ms
        if v and n < 3:
            mask[s:s + n] = False
    return mask


def acoustic_features(y, sr=SR) -> dict:
    y = np.asarray(y, dtype=np.float32)
    if len(y) < WIN * 4:
        y = np.pad(y, (0, WIN * 4 - len(y)))
    dur = len(y) / sr

    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-6)
    mask = _speech_mask(db)
    speech_time = max(mask.sum() / FPS, 1e-3)

    def sel(x):
        L = min(len(x), len(mask)); v = x[:L][mask[:L]]
        return v if len(v) else x

    f = {}

    # ---- pauses ----
    runs = _runs(mask)
    pauses = [n / FPS for k, (s, n, v) in enumerate(runs)
              if (not v) and 0 < k < len(runs) - 1 and n / FPS >= MIN_PAUSE]
    f["speech_ratio"] = float(mask.mean())
    f["n_pauses"] = len(pauses)
    f["pause_rate"] = len(pauses) / dur
    f["mean_pause"] = float(np.mean(pauses)) if pauses else 0.0
    f["max_pause"] = float(np.max(pauses)) if pauses else 0.0
    f["pause_ratio"] = float(np.sum(pauses)) / dur
    f["pause_cv"] = float(np.std(pauses) / np.mean(pauses)) if len(pauses) > 1 else 0.0

    # ---- speech rate + rhythm (onsets as syllable proxy) ----
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, units="time")
    f["speech_rate"] = len(onsets) / max(speech_time, 0.3)
    ioi = np.diff(onsets)
    if len(ioi) >= 2:
        f["ioi_mean"], f["ioi_std"] = float(ioi.mean()), float(ioi.std())
        f["ioi_cv"] = f["ioi_std"] / (f["ioi_mean"] + 1e-6)
        f["npvi"] = float(100 * np.mean(np.abs(ioi[:-1] - ioi[1:]) / ((ioi[:-1] + ioi[1:]) / 2 + 1e-6)))
    else:
        f["ioi_mean"] = f["ioi_std"] = f["ioi_cv"] = f["npvi"] = 0.0

    # ---- repetition & prolongation from pooled MFCC (50 ms) ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=HOP, win_length=WIN)
    X = mfcc[1:].T
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    n = min(len(X), len(mask)) // 5
    f["rep_max"] = f["rep_lag"] = f["rep_mean"] = 0.0
    f["stable_longest"] = f["stable_ratio"] = 0.0
    if n > 16:
        Xp = X[:n * 5].reshape(n, 5, -1).mean(1)
        mp = mask[:n * 5].reshape(n, 5).mean(1) > 0.5
        Xn = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-6)
        S = Xn @ Xn.T
        vals = []
        for lag in range(2, 15):                                 # 100-700 ms
            ok = mp[:-lag] & mp[lag:]
            vals.append(S[np.arange(n - lag), np.arange(lag, n)][ok].mean() if ok.any() else 0.0)
        vals = np.array(vals)
        f["rep_max"], f["rep_lag"], f["rep_mean"] = float(vals.max()), (int(vals.argmax()) + 2) * 0.05, float(vals.mean())

        dm = np.linalg.norm(np.diff(Xp, axis=0), axis=1)
        valid = mp[1:] & mp[:-1]
        if valid.sum() > 3:
            stable = (dm < 0.5 * np.median(dm[valid])) & valid
            longest = max([m for _, m, v in _runs(stable) if v] or [0])
            f["stable_longest"], f["stable_ratio"] = longest * 0.05, float(stable.sum() / valid.sum())

    # ---- energy ----
    dbs = sel(db)
    f["rms_db_mean"], f["rms_db_std"] = float(dbs.mean()), float(dbs.std())
    f["rms_db_range"] = float(dbs.max() - dbs.min())
    lin = sel(rms); f["rms_cv"] = float(lin.std() / (lin.mean() + 1e-6))

    # ---- spectral ----
    f["centroid_mean"] = float(sel(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=HOP)[0]).mean())
    f["centroid_std"] = float(sel(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=HOP)[0]).std())
    f["flatness_mean"] = float(sel(librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=HOP)[0]).mean())
    f["zcr_mean"] = float(sel(librosa.feature.zero_crossing_rate(y, frame_length=WIN, hop_length=HOP)[0]).mean())

    # ---- pitch ----
    f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    lf0 = np.log(sel(f0) + 1e-6)
    f["f0_mean"], f["f0_std"] = float(lf0.mean()), float(lf0.std())

    # ---- MFCC statistics ----
    L = min(mfcc.shape[1], len(mask)); m = mfcc[:, :L][:, mask[:L]]
    m = m if m.shape[1] else mfcc
    for i in range(13):
        f[f"mfcc{i}_mean"], f[f"mfcc{i}_std"] = float(m[i].mean()), float(m[i].std())

    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in f.items()}


def to_vector(d: dict) -> np.ndarray:
    return np.array(list(d.values()), dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature names / groups (used for ablations and importance analysis)
# ---------------------------------------------------------------------------
_GROUP_RULES = [
    ("pause", ("speech_ratio", "n_pauses", "pause_", "mean_pause", "max_pause")),
    ("rate_rhythm", ("speech_rate", "ioi_", "npvi")),
    ("repetition", ("rep_",)),
    ("prolongation", ("stable_",)),
    ("energy", ("rms_",)),
    ("spectral", ("centroid", "flatness", "zcr")),
    ("pitch", ("f0",)),
    ("mfcc", ("mfcc",)),
]
_NAMES_CACHE = None


def feature_names() -> list:
    """Ordered names of the acoustic vector (computed once from a dummy signal)."""
    global _NAMES_CACHE
    if _NAMES_CACHE is None:
        rng = np.random.default_rng(0)
        _NAMES_CACHE = list(acoustic_features(rng.normal(0, 0.1, SR * 3).astype(np.float32)).keys())
    return _NAMES_CACHE


def feature_groups() -> dict:
    """{group_name: [column indices in the acoustic vector]}"""
    groups = {g: [] for g, _ in _GROUP_RULES}
    for i, n in enumerate(feature_names()):
        for g, prefixes in _GROUP_RULES:
            if n.startswith(prefixes):
                groups[g].append(i)
                break
        else:
            raise ValueError(f"feature {n} has no group")
    return groups
