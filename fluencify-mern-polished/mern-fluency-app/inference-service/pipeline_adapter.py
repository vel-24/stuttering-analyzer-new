"""
Adapter between the FastAPI service (main.py) and the real pipeline code
(infer.py, fluency.py, acoustic_features.py, segmentation.py,
whisper_features.py, whisper_ft_embedder.py) -- copied in verbatim from
the Streamlit app's source, unmodified.

Two differences from infer.analyse() (the Streamlit entry point):

1. History/feedback: tracker.py's SQLite table (`sessions.db`) is no
   longer the source of truth once this runs behind Express/MongoDB --
   Mongo is. So `analyze_audio_file` takes `prior` (a list of this user's
   recent {"score", "components"} dicts, fetched from MongoDB by Express
   and passed through) instead of letting tracker.feedback() read its own
   SQLite file. Nothing is written to sessions.db from here; Express
   persists the result after this returns.

2. Missing model artifacts: the trained classifier bundles
   (models/*.joblib), fine-tuned Whisper checkpoints
   (data/whisper_ft_seed*.pt) and finetune_whisper.py (the WhisperFT
   class definition) are large binaries that weren't uploaded to this
   session, so this service can't load them yet. REAL_PIPELINE_READY
   checks for them at import time; until they're in place, calls fall
   back to the same stub response used during initial scaffolding, so
   the full React/Express/Mongo stack keeps working end to end for a
   demo while you copy the real artifacts in (see README.md).
"""

import random
import time
from pathlib import Path

from config import MODEL_DIR, TYPE_CLASSES
from segmentation import load_and_preprocess, adaptive_segments
from acoustic_features import acoustic_features
from fluency import compute_components, fluency_score, WEIGHTS
from tracker import feedback as compute_feedback
from infer import get_embedder, predict_segments

import joblib


def _artifacts_present(model_name: str) -> bool:
    bundle_path = MODEL_DIR / f"{model_name}.joblib"
    if not bundle_path.exists():
        return False
    if model_name in ("whisper_ft_only", "fusion_ft", "whisper_ft_head"):
        # also needs finetune_whisper.py + a checkpoint; whisper_ft_embedder.py
        # raises FileNotFoundError itself if the .pt is missing, and ImportError
        # if finetune_whisper.py isn't present -- both are treated as "not ready".
        try:
            import finetune_whisper  # noqa: F401
        except ImportError:
            return False
    return True


VALID_MODELS = ["whisper_ft_head", "fusion_ft", "whisper_ft_only", "fusion", "whisper_only", "acoustic_only"]


def analyze_audio_file(
    file_path: str,
    model_name: str = "whisper_ft_head",
    ft_seed: int = 0,
    prior: list | None = None,
) -> dict:
    if model_name not in VALID_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {VALID_MODELS}")

    if not _artifacts_present(model_name):
        return _stub_result(model_name, reason="model artifacts not found on disk yet")

    bundle = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    y = load_and_preprocess(file_path)
    segs = adaptive_segments(y)
    if not segs:
        raise ValueError("No speech detected in the uploaded audio.")

    embedder = None if model_name == "acoustic_only" else get_embedder(model_name, ft_seed)
    results = predict_segments(segs, bundle, model_name, embedder, ft_seed)

    components = compute_components(results)
    score = fluency_score(components)
    counts = {label: sum(r["label"] == label for r in results) for label in ["Fluent"] + bundle["type_classes"]}
    duration = len(y) / 16000

    # prior=[] (not None) forces tracker.feedback() to use what we pass rather than
    # falling back to its own SQLite lookup -- see module docstring.
    msgs = compute_feedback(user="_unused_", components=components, score=score, prior=prior or [])

    dominant = max(
        (l for l in counts if l != "Fluent" and counts[l] > 0),
        key=lambda l: counts[l],
        default="none",
    )

    return {
        "fluencyScore": score,
        "scoreBreakdown": components,
        "segments": results,
        "summary": {
            "totalSegments": len(results),
            "durationSec": round(duration, 2),
            "dominantIssue": dominant,
            "counts": counts,
        },
        "feedback": msgs,
        "modelUsed": model_name,
        "stub": False,
    }


def _stub_result(model_name: str = "whisper_ft_head", reason: str = "") -> dict:
    time.sleep(0.3)
    classes = ["Fluent"] + TYPE_CLASSES
    n_segments = random.randint(4, 8)
    segments = []
    t = 0.0
    for i in range(n_segments):
        dur = round(random.uniform(1.5, 3.0), 2)
        label = random.choice(classes)
        segments.append(
            {
                "start": round(t, 2),
                "end": round(t + dur, 2),
                "label": label,
                "p_disfluent": round(random.uniform(0.05, 0.95), 3),
                "speech_rate": round(random.uniform(2.0, 5.0), 2),
                "pause_cv": round(random.uniform(0.1, 1.2), 2),
            }
        )
        t += dur

    components = {k: round(random.uniform(0.4, 0.95), 2) for k in WEIGHTS}
    score = fluency_score(components)
    counts = {c: sum(s["label"] == c for s in segments) for c in classes}
    dominant = max(
        (l for l in counts if l != "Fluent" and counts[l] > 0),
        key=lambda l: counts[l],
        default="none",
    )

    return {
        "fluencyScore": score,
        "scoreBreakdown": components,
        "segments": segments,
        "summary": {
            "totalSegments": n_segments,
            "durationSec": round(t, 2),
            "dominantIssue": dominant,
            "counts": counts,
        },
        "feedback": [f"[STUB DATA — {reason}] Drop in models/*.joblib (and fine-tuned "
                     f"checkpoints + finetune_whisper.py for {model_name}) to get real scores."],
        "modelUsed": model_name,
        "stub": True,
    }
