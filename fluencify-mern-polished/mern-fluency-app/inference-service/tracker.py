"""STEP 8 - Session tracking (SQLite) and personalised feedback."""
import json
import sqlite3
import time
import numpy as np
from config import DB_PATH
from fluency import WEIGHTS

TIPS = {
    "speech_rate_stability": "Your pace varies a lot. Practise reading aloud with a metronome or steady, slightly slower pacing.",
    "block_frequency": "Blocks are your biggest cost. Try easy onsets: breathe out gently, then start the word softly.",
    "prolongation_ratio": "Sounds are being stretched. Practise light contacts and moving smoothly into the next sound.",
    "repetition_rate": "Repetitions are frequent. Pause briefly, slow down and use 'cancellation': stop, then repeat the word smoothly.",
    "pause_regularity": "Pauses are uneven. Plan short, regular pauses at phrase boundaries.",
    "smoothness": "Overall flow is uneven. Try phrasing: speak in short groups of words with soft transitions.",
    "recovery_speed": "After a disfluency you take a while to recover. Practise resuming calmly right after a stall.",
}


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS sessions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, ts REAL, duration REAL,
        score REAL, components TEXT, counts TEXT)""")
    return c


def save_session(user, score, components, counts, duration):
    with _conn() as c:
        c.execute("INSERT INTO sessions(user,ts,duration,score,components,counts) VALUES(?,?,?,?,?,?)",
                  (user, time.time(), duration, score, json.dumps(components), json.dumps(counts)))


def history(user, n=20):
    with _conn() as c:
        rows = c.execute("SELECT ts,duration,score,components,counts FROM sessions "
                         "WHERE user=? ORDER BY ts DESC LIMIT ?", (user, n)).fetchall()
    return [{"ts": r[0], "duration": r[1], "score": r[2],
             "components": json.loads(r[3]), "counts": json.loads(r[4])} for r in rows[::-1]]


def feedback(user, components, score, prior=None):
    """`prior` = sessions before the current one (call BEFORE save_session, or pass history)."""
    prior = history(user, 5) if prior is None else prior
    msgs = []
    if prior:
        base = float(np.mean([p["score"] for p in prior]))
        d = score - base
        trend = "improved" if d > 2 else "dropped" if d < -2 else "stayed steady"
        msgs.append(f"Your fluency score is {score:.1f}, which has {trend} ({d:+.1f}) vs your last "
                    f"{len(prior)} session(s) (avg {base:.1f}).")
        gains = {k: components[k] - np.mean([p["components"][k] for p in prior]) for k in WEIGHTS}
        best = max(gains, key=gains.get)
        if gains[best] > 0.05:
            msgs.append(f"Biggest improvement: {best.replace('_', ' ')}.")
    else:
        msgs.append(f"First session recorded - your baseline fluency score is {score:.1f}.")
    weakest = max(WEIGHTS, key=lambda k: WEIGHTS[k] * (1 - components[k]))   # largest weighted loss
    msgs.append(f"Focus area: {weakest.replace('_', ' ')}. {TIPS[weakest]}")
    return msgs
