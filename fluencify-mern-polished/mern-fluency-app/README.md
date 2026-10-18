# Fluency App — React + Express/MongoDB + Python inference

This replaces the Streamlit UI with a React frontend and an Express/MongoDB
backend, while keeping your ML pipeline in Python behind a small FastAPI
service. Nothing about your model code needs to be rewritten in JavaScript.

```
Browser (React)  -->  Express API (Node, Mongo: auth + history)  -->  FastAPI (Python: your ML pipeline)
```

## Why this shape

- Express/Mongo handles accounts (JWT + bcrypt, replacing `auth.py`'s
  SQLite/PBKDF2 logic) and attempt history (replacing `tracker.py`).
- The Python service is untouched ML code wrapped in a thin HTTP layer.
  Whisper fine-tuned embeddings, XGBoost, librosa features — none of that
  has a practical JS equivalent, so it stays where it already works.
- React replaces `app.py`'s UI: recording/upload, charts, history.

## 1. Pipeline wiring status

Your real pipeline code is already copied into `inference-service/` and
wired up — `config.py`, `segmentation.py`, `acoustic_features.py`,
`whisper_features.py`, `whisper_ft_embedder.py`, `fluency.py`,
`tracker.py` (feedback/tip logic only) and `infer.py` are in there
unmodified, and `pipeline_adapter.py` calls the same
`get_embedder()` / `predict_segments()` / `compute_components()` /
`fluency_score()` functions `infer.analyse()` uses — just with the
history source swapped (see below).

**What's still missing** is the large binary artifacts, which weren't
practical to upload into this session:

- `inference-service/models/*.joblib` — the trained XGBoost bundles for
  whichever model(s) you want to serve (`fusion.joblib`,
  `whisper_only.joblib`, `acoustic_only.joblib`, etc.)
- `inference-service/data/whisper_ft_seed<N>.pt` — fine-tuned Whisper
  checkpoints, needed only for `whisper_ft_head` / `fusion_ft` /
  `whisper_ft_only`
- `inference-service/finetune_whisper.py` — defines the `WhisperFT`
  class and `N_SAMPLES` that `whisper_ft_embedder.py` imports; also
  needed only for the three fine-tuned models above

Copy those in from wherever you trained them (not from this chat), and
everything downstream works unchanged. **Until they're in place, the
service auto-falls-back to clearly-labeled stub data** per model — so
you can run the full React → Express → Mongo → FastAPI round trip today,
and swap in real numbers incrementally, model by model, without
touching any other code. Check `GET /health` on the inference service to
see which models currently have their artifacts in place
(`modelsReady: {...}`).

### What changed vs. `infer.analyse()`

Two small differences, both in `pipeline_adapter.py`:

1. **History/feedback source.** `tracker.py`'s SQLite table
   (`sessions.db`) is no longer the source of truth — MongoDB is, via
   Express's `Attempt` collection. So `analyze_audio_file()` takes a
   `prior` list (this user's last 5 `{score, components}`, fetched from
   Mongo and forwarded by `server/src/routes/analyze.js`) instead of
   letting `tracker.feedback()` read its own SQLite file. Nothing is
   written to `sessions.db` — Express persists the result in Mongo
   after the service returns it.
2. **Auth.** `auth.py`'s SQLite-backed username/password check is
   superseded by Express's JWT + bcrypt + MongoDB `User` model
   (`server/src/routes/auth.js`). `auth.py` wasn't copied into
   `inference-service/` since nothing there needs it.

Everything else — segmentation, acoustic features, Whisper embeddings,
the XGBoost/fine-tuned-head classifiers, the 7-component fluency score
and its weights, and the personalized feedback/tips text — is your
original code, untouched.

## 2. Run all three services locally

You'll need Node 18+, Python 3.10+, and a MongoDB instance (local install
or a free Atlas cluster).

**Inference service (Python):**
```bash
cd inference-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**API server (Node/Express):**
```bash
cd server
cp .env.example .env   # edit MONGO_URI / JWT_SECRET as needed
npm install
npm run dev
```

**Frontend (React):**
```bash
cd client
cp .env.example .env
npm install
npm run dev
```

Then open http://localhost:5173, sign up, and record or upload a clip.

## 3. What's included

- `inference-service/` — FastAPI app, `/analyze` and `/health` endpoints,
  stub pipeline you'll swap for your real one.
- `server/` — Express API: `/api/auth/signup`, `/api/auth/login`,
  `/api/analyze` (forwards to FastAPI, saves result), `/api/history`
  (list/get/delete past attempts). MongoDB models for `User` and `Attempt`.
- `client/` — React (Vite) app: signup/login, mic recording via
  `MediaRecorder` or file upload, results view with a Recharts bar chart,
  and an attempts history table.

## 4. Known gaps to close before this is submission-ready

- Model artifacts (`*.joblib`, fine-tuned checkpoints, `finetune_whisper.py`)
  need to be copied into `inference-service/` from wherever you trained
  them — see section 1. Until then, responses are clearly marked
  `"stub": true` both in the API response and in the React UI.
- `inference-service/requirements.txt` now lists the real dependencies
  (torch, transformers, librosa, xgboost, scikit-learn) — these are
  heavy installs (especially torch), so budget time/disk for `pip install`.
- No file-size/type validation beyond what's in `main.py`/`analyze.js`
  (25MB cap, common audio extensions) — tighten if needed.
- No automated tests yet. Given your paper already reports evaluation
  numbers from the Python pipeline directly, you likely don't need new ML
  tests here — just a couple of integration checks that the Express →
  FastAPI → Mongo round trip works once the artifacts are in place.
- Deployment (Docker, hosting) isn't set up — say the word if you want a
  `docker-compose.yml` tying all three services together.
