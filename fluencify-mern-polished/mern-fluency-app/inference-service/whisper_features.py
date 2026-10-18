"""STEP 4 / Branch A - Whisper encoder embeddings.

The Whisper encoder emits 50 frames/s. We keep only the frames that cover real audio
(the rest is padding to 30 s) and pool with mean + std -> 2*d_model vector.
Branch A here is a frozen encoder; see README for the optional fine-tuning extension.
"""
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel
from config import SR, WHISPER_NAME


class WhisperEmbedder:
    def __init__(self, name: str = WHISPER_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.encoder = WhisperModel.from_pretrained(name).encoder.to(self.device).eval()

    @torch.no_grad()
    def embed(self, waves: list[np.ndarray]) -> np.ndarray:
        feats = self.fe([np.asarray(w, dtype=np.float32) for w in waves],
                        sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
        hidden = self.encoder(feats).last_hidden_state            # (B, 1500, d)
        out = []
        for i, w in enumerate(waves):
            n = int(np.clip(len(w) / SR * 50, 2, hidden.shape[1]))
            h = hidden[i, :n]
            out.append(torch.cat([h.mean(0), h.std(0)]).float().cpu().numpy())
        return np.stack(out)
