"""STEP 7 - Fluency score (0-100) using the weights from the proposal:

 25 f(speech-rate stability) + 20 f(block frequency) + 15 f(prolongation ratio)
 + 15 f(repetition rate) + 10 f(pause regularity) + 10 f(smoothness) + 5 f(recovery speed)

Every f() is mapped to [0, 1] where 1 = perfectly fluent. The saturation constants below
(0.15, 0.5, 1.5, 3 ...) are tunable design choices - calibrate them on your data and report them.
Interjections are captured through `smoothness` (uses P(disfluent) of every segment).
"""
import numpy as np

WEIGHTS = {"speech_rate_stability": 25, "block_frequency": 20, "prolongation_ratio": 15,
           "repetition_rate": 15, "pause_regularity": 10, "smoothness": 10, "recovery_speed": 5}


def _c01(x):
    return float(np.clip(x, 0.0, 1.0))


# Saturation constants: value at which a component reaches 0 (calibrate with score_validation.py)
SAT = {"rate_cv": 0.5, "block": 0.15, "prolong": 0.15, "repeat": 0.15, "pause_cv": 1.5, "recovery_run": 3.0}


def compute_components(segments: list[dict], sat: dict | None = None) -> dict:
    """segments: dicts with keys label ('Fluent'|type), p_disfluent, speech_rate, pause_cv."""
    sat = {**SAT, **(sat or {})}
    n = len(segments)
    if n == 0:
        return {k: 1.0 for k in WEIGHTS}
    labels = [s["label"] for s in segments]
    frac = lambda c: sum(l == c for l in labels) / n

    rates = np.array([s["speech_rate"] for s in segments])
    rate_cv = rates.std() / (rates.mean() + 1e-6) if n > 1 else 0.0

    # recovery: mean length of consecutive disfluent runs (1 segment = fastest recovery)
    runs, cur = [], 0
    for l in labels:
        if l != "Fluent":
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    recovery = 1.0 if not runs else _c01(1 - (np.mean(runs) - 1) / sat['recovery_run'])

    return {
        "speech_rate_stability": _c01(1 - rate_cv / sat['rate_cv']),
        "block_frequency": _c01(1 - frac("Block") / sat["block"]),
        "prolongation_ratio": _c01(1 - frac("Prolongation") / sat["prolong"]),
        "repetition_rate": _c01(1 - frac("Repetition") / sat["repeat"]),
        "pause_regularity": _c01(1 - np.mean([s["pause_cv"] for s in segments]) / sat["pause_cv"]),
        "smoothness": _c01(1 - np.mean([s["p_disfluent"] for s in segments])),
        "recovery_speed": recovery,
    }


def fluency_score(components: dict) -> float:
    return round(sum(WEIGHTS[k] * components[k] for k in WEIGHTS), 1)
