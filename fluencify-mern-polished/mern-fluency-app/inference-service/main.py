"""
FastAPI inference microservice.

Keeps your Python ML pipeline exactly where it belongs (Python), and
exposes it over HTTP so the Node/Express backend can call it. Express
never re-implements any model logic -- it just forwards audio files
here and relays the JSON result.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""

import json
import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline_adapter import analyze_audio_file, VALID_MODELS

app = FastAPI(title="Fluency Analysis Inference Service")

# Express is the only expected caller in production; CORS is kept open here
# for local development convenience. Tighten ALLOWED_ORIGINS for deployment.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


class HealthResponse(BaseModel):
    status: str
    modelsReady: dict


@app.get("/health", response_model=HealthResponse)
def health():
    from pipeline_adapter import VALID_MODELS, _artifacts_present

    return {"status": "ok", "modelsReady": {m: _artifacts_present(m) for m in VALID_MODELS}}


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    model: str = Form("whisper_ft_head"),
    ft_seed: int = Form(0),
    prior: str = Form("[]"),  # JSON-encoded list of {"score", "components"} from Express/Mongo
):
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    if model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"model must be one of {VALID_MODELS}")

    try:
        prior_list = json.loads(prior)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="prior must be JSON-encoded")

    tmp_dir = tempfile.mkdtemp(prefix="fluency_")
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{ext}")

    try:
        size = 0
        with open(tmp_path, "wb") as out_file:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (max 25MB).")
                out_file.write(chunk)

        result = analyze_audio_file(tmp_path, model_name=model, ft_seed=ft_seed, prior=prior_list)
        return result
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
