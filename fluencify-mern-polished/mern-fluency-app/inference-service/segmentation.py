"""STEP 2 + 3 - Audio preprocessing (librosa) and adaptive segmentation.

Instead of cutting fixed windows, we
  1. estimate the recording's own noise floor and speech level (adaptive threshold),
  2. drop leading/trailing silence and non-speech chunks,
  3. cut at quiet points that are never inside a long silence, so blocks stay intact.
Segments are <= 3 s so they match the SEP-28k clip length the models are trained on.
"""
from itertools import groupby
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
from config import SR

HOP, WIN = 160, 400          # 10 ms hop, 25 ms window at 16 kHz


def load_and_preprocess(path_or_array, sr: int = SR) -> np.ndarray:
    if isinstance(path_or_array, (str, bytes)) or hasattr(path_or_array, "__fspath__"):
        y, _ = librosa.load(path_or_array, sr=sr, mono=True)
    else:
        y = np.asarray(path_or_array, dtype=np.float32)
    if len(y) == 0:
        return y
    sos = butter(4, 80, btype="highpass", fs=sr, output="sos")   # remove rumble / DC
    y = sosfiltfilt(sos, y).astype(np.float32)
    peak = np.max(np.abs(y))
    return y / peak * 0.95 if peak > 1e-6 else y


def _frame_db(y):
    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP)[0]
    return 20 * np.log10(rms + 1e-6)


def _adaptive_threshold(db):
    noise, peak = np.percentile(db, 10), np.percentile(db, 95)
    return min(max(noise + 6.0, peak - 35.0), peak - 12.0)


def _runs(mask):
    out, i = [], 0
    for val, grp in groupby(mask):
        n = len(list(grp)); out.append((i, n, bool(val))); i += n
    return out


def adaptive_segments(y, sr=SR, min_len=1.0, max_len=3.0, min_speech_ratio=0.15, silence_run=0.3):
    """Pause-preserving, energy-aware segmentation. Returns list of {start, end, wave} (seconds).

    * Leading/trailing silence and non-speech chunks are dropped.
    * Cuts are placed at the quietest point that is NOT inside a silent stretch >= `silence_run`,
      so a block/pause (a long silence inside a phrase) always stays whole inside one segment
      and the classifier can see it. Cutting mid-silence would destroy exactly the event we
      want to detect.
    * Segments are <= max_len (3 s = SEP-28k clip length) and >= min_len when possible.
    """
    if len(y) < WIN * 2:
        return []
    fps = sr / HOP
    db = _frame_db(y)
    active = db > _adaptive_threshold(db)
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return []
    start, end = int(idx[0]), int(idx[-1]) + 1

    forbid = np.zeros(len(active), dtype=bool)                 # frames inside long silences
    for s0, n, v in _runs(active):
        if not v and n / fps >= silence_run:
            forbid[s0:s0 + n] = True
    cost = np.where(forbid, np.inf, np.convolve(db, np.ones(5) / 5, mode="same"))

    segs, s = [], start
    while s < end:
        if (end - s) / fps <= max_len:
            segs.append([s, end]); break
        lo, hi = s + int(min_len * fps), min(s + int(max_len * fps), len(cost))
        win = cost[lo:hi]
        cut = lo + int(np.argmin(win)) if len(win) and np.isfinite(win).any() else hi
        segs.append([s, cut]); s = cut
    if len(segs) > 1 and (segs[-1][1] - segs[-1][0]) / fps < 0.5:   # merge a tiny tail
        segs[-2][1] = segs[-1][1]; segs.pop()

    out = []
    for s0, e0 in segs:
        if (e0 - s0) / fps >= 0.3 and active[s0:e0].mean() >= min_speech_ratio:
            a, b = s0 * HOP, min(len(y), e0 * HOP)
            out.append({"start": a / sr, "end": b / sr, "wave": y[a:b]})
    return out
