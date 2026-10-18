"""STEPS 1-8 end to end on a new recording.

    python infer.py my_speech.wav --user lokesh --model fusion
"""
import argparse
import joblib
import numpy as np
from config import MODEL_DIR, SR
from segmentation import load_and_preprocess, adaptive_segments
from acoustic_features import acoustic_features, to_vector
from fluency import compute_components, fluency_score
import tracker


def get_embedder(model_name, ft_seed=0):
    """Pick the frozen or fine-tuned Whisper embedder based on the model name."""
    if model_name in ("whisper_ft_only", "fusion_ft", "whisper_ft_head"):
        from whisper_ft_embedder import WhisperFTEmbedder
        return WhisperFTEmbedder(seed=ft_seed)
    from whisper_features import WhisperEmbedder
    return WhisperEmbedder()


def predict_segments(segs, bundle, model_name="fusion", embedder=None, ft_seed=0):
    """Steps 4-6 for a list of segments -> list of dicts (label, p_disfluent, speech_rate, pause_cv)."""
    ac = [acoustic_features(s["wave"]) for s in segs]       # step 4 / branch B, always computed:
    waves = [s["wave"] for s in segs]                       # speech_rate/pause_cv feed the fluency
    # score regardless of which classifier is used (step 7 reads them straight off `ac`).

    if model_name == "whisper_ft_head":
        # The fine-tuned network's OWN head end-to-end -- no XGBoost, no acoustic fusion for
        # classification. Per the 5-seed significance test this was the best-performing model
        # and fusion was NOT significantly better, so it's the deployment default.
        if embedder is None:
            embedder = get_embedder(model_name, ft_seed)
        p_chunks, t_chunks = [], []
        for i in range(0, len(waves), 16):
            p, t = embedder.predict_head(waves[i:i + 16])
            p_chunks.append(p); t_chunks.append(t)
        p_dis = np.concatenate(p_chunks)
        type_logits = np.vstack(t_chunks)
        is_dis = p_dis >= bundle["threshold"]
        types = type_logits.argmax(1)
        return [{"start": round(s["start"], 2), "end": round(s["end"], 2),
                 "label": bundle["type_classes"][types[i]] if is_dis[i] else "Fluent",
                 "p_disfluent": float(p_dis[i]), "speech_rate": ac[i]["speech_rate"],
                 "pause_cv": ac[i]["pause_cv"]} for i, s in enumerate(segs)]

    A = np.vstack([to_vector(a) for a in ac])
    if model_name == "acoustic_only":
        X = A
    else:
        if embedder is None:
            embedder = get_embedder(model_name, ft_seed)
        W = np.vstack([embedder.embed(waves[i:i + 16])      # step 4 / branch A
                       for i in range(0, len(waves), 16)])
        X = W if model_name in ("whisper_only", "whisper_ft_only") else np.hstack([W, A])

    p_dis = bundle["binary"].predict_proba(X)[:, 1]         # steps 5-6
    is_dis = p_dis >= bundle["threshold"]
    types = bundle["type"].predict(X)
    return [{"start": round(s["start"], 2), "end": round(s["end"], 2),
             "label": bundle["type_classes"][types[i]] if is_dis[i] else "Fluent",
             "p_disfluent": float(p_dis[i]), "speech_rate": ac[i]["speech_rate"],
             "pause_cv": ac[i]["pause_cv"]} for i, s in enumerate(segs)]


def analyse(wav_path, user="default", model_name="fusion", embedder=None, save=True, ft_seed=0):
    bundle = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    y = load_and_preprocess(wav_path)                       # step 2
    segs = adaptive_segments(y)                             # step 3
    if not segs:
        raise ValueError("No speech detected.")
    results = predict_segments(segs, bundle, model_name, embedder, ft_seed)

    comps = compute_components(results)                     # step 7
    score = fluency_score(comps)
    counts = {l: sum(r["label"] == l for r in results) for l in ["Fluent"] + bundle["type_classes"]}
    prior = tracker.history(user, 5)
    msgs = tracker.feedback(user, comps, score, prior)      # step 8
    if save:
        tracker.save_session(user, score, comps, counts, len(y) / SR)
    return {"segments": results, "components": comps, "score": score, "counts": counts,
            "feedback": msgs, "duration": len(y) / SR}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav"); ap.add_argument("--user", default="default")
    ap.add_argument("--model", default="whisper_ft_head",
                    choices=["whisper_ft_head", "fusion_ft", "whisper_ft_only",
                             "fusion", "whisper_only", "acoustic_only"])
    ap.add_argument("--ft-seed", type=int, default=0, help="which fine-tuned checkpoint to use")
    a = ap.parse_args()
    out = analyse(a.wav, a.user, a.model, ft_seed=a.ft_seed)
    print("\nTimeline:")
    for r in out["segments"]:
        print(f"  {r['start']:6.2f}-{r['end']:6.2f}s  {r['label']:13s} (p_disfluent={r['p_disfluent']:.2f})")
    print("\nCounts:", out["counts"])
    print("Components:", {k: round(v, 2) for k, v in out["components"].items()})
    print(f"\nFLUENCY SCORE: {out['score']}/100")
    for m in out["feedback"]:
        print(" -", m)
