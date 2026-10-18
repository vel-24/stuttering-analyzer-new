"""Inference-time embedder using a fine-tuned Whisper encoder (see finetune_whisper.py).
Loads the checkpoint saved as data/whisper_ft_seed<seed>.pt.

    from whisper_ft_embedder import WhisperFTEmbedder
    embedder = WhisperFTEmbedder(seed=0)
    vectors = embedder.embed([wave1, wave2, ...])   # each wave is a 1-D float32 array at SR
"""
import numpy as np
import torch
from transformers import WhisperFeatureExtractor
from config import SR, WHISPER_NAME, DATA_DIR
from finetune_whisper import WhisperFT, N_SAMPLES


class WhisperFTEmbedder:
    def __init__(self, seed: int = 0, device: str | None = None, unfreeze: int = 2, tag: str = ""):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        suffix = f"_{tag}" if tag else ""
        ckpt = DATA_DIR / f"whisper_ft_seed{seed}{suffix}.pt"
        if not ckpt.exists():
            hint = f" --tag {tag}" if tag else ""
            raise FileNotFoundError(
                f"{ckpt} not found. Run: python finetune_whisper.py --seed {seed}{hint}"
            )
        self.model = WhisperFT(WHISPER_NAME, n_unfreeze=unfreeze).to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.fe = WhisperFeatureExtractor.from_pretrained(WHISPER_NAME)

    def _mels(self, waves: list[np.ndarray]):
        clipped = [np.asarray(w, dtype=np.float32)[:N_SAMPLES] for w in waves]
        return self.fe(clipped, sampling_rate=SR, return_tensors="pt",
                       padding="max_length", max_length=N_SAMPLES).input_features.to(self.device)

    @torch.no_grad()
    def embed(self, waves: list[np.ndarray]) -> np.ndarray:
        """Pooled embedding only -- for feeding into a separate XGBoost model (fusion_ft etc)."""
        _, e = self.model(self._mels(waves))
        return e.cpu().numpy()

    @torch.no_grad()
    def predict_head(self, waves: list[np.ndarray]):
        """Run the fine-tuned network's OWN classification head end-to-end (no XGBoost stage).
        Returns (p_disfluent, type_logits) as numpy arrays.
        """
        out, _ = self.model(self._mels(waves))
        p_bin = torch.sigmoid(out[:, 0]).cpu().numpy()
        type_logits = out[:, 1:].cpu().numpy()
        return p_bin, type_logits
