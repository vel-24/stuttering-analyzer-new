"""BRANCH A FINE-TUNING - adapt the top Whisper encoder layers to stuttering on SEP-28k.

    python finetune_whisper.py --seed 0 --epochs 3 --unfreeze 2        # GPU recommended

    # with SpecAugment + warmup/cosine LR schedule (opt-in, off by default so old runs
    # stay reproducible -- compare against your existing ft_seed0.npz before trusting this):
    python finetune_whisper.py --seed 0 --epochs 3 --augment --schedule

* Encoder is truncated to a 3 s input (150 frames) instead of Whisper's 30 s padding -> ~10x faster.
* Only the last `--unfreeze` encoder blocks + attentive pooling + a small multi-task head are trained
  (fluent/disfluent + 4 types), using the SAME episode-wise split as experiments.py for that seed.
* --augment applies SpecAugment (random time/frequency masking on the mel spectrogram) to the
  TRAINING split only, each epoch gets a different random mask. Standard fix for small fine-tuning
  sets; should reduce overfitting to the specific clips rather than the disfluency pattern itself.
* --schedule adds a short linear warmup followed by cosine decay, instead of a flat learning rate
  for all epochs. Targets the val-loss bounce you see if a later epoch's val loss is worse than an
  earlier one (the flat LR is probably too high once the model is already close to converged).
* Saves data/ft_seed<seed>.npz with, for EVERY clip in features.npz (same row order):
      emb          fine-tuned pooled embedding   (-> XGBoost / fusion in experiments.py)
      p_bin        P(disfluent) from the network's own head
      type_logits  4 type logits from the head
  Evaluate on the test split only. Train one model per seed (encoder never sees that seed's test episodes).
"""
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor, WhisperModel, WhisperConfig, get_cosine_schedule_with_warmup

from config import FEAT_FILE, DATA_DIR, SR, CLIP_SEC, WHISPER_NAME, TYPE_CLASSES
from splits import group_split
from segmentation import load_and_preprocess

N_SAMPLES = int(SR * CLIP_SEC)
N_POS = int(CLIP_SEC * 50)          # encoder frames for 3 s (50 frames / s)


def load_encoder(name):
    if name == "__tiny_test__":     # random tiny model, used only for offline unit tests
        cfg = WhisperConfig(d_model=32, encoder_layers=3, decoder_layers=1, encoder_attention_heads=2,
                            decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=80)
        return WhisperModel(cfg).encoder
    return WhisperModel.from_pretrained(name).encoder


def truncate_encoder(enc, n_pos=N_POS):
    """Make the encoder accept 3 s inputs by shortening its positional table."""
    w = enc.embed_positions.weight.data[:n_pos].clone()
    emb = nn.Embedding(n_pos, w.shape[1]); emb.weight.data.copy_(w); emb.weight.requires_grad = False
    enc.embed_positions = emb
    enc.config.max_source_positions = n_pos


class WhisperFT(nn.Module):
    def __init__(self, name=WHISPER_NAME, n_unfreeze=2, n_types=len(TYPE_CLASSES)):
        super().__init__()
        enc = load_encoder(name)
        truncate_encoder(enc)
        for p in enc.parameters():
            p.requires_grad = False
        for layer in enc.layers[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in enc.layer_norm.parameters():
            p.requires_grad = True
        self.enc, d = enc, enc.config.d_model
        self.att = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * d, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, 1 + n_types))

    def embed(self, x):
        h = self.enc(x).last_hidden_state                              # (B, 150, d)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1).unsqueeze(-1)  # attentive pooling
        mu = (a * h).sum(1)
        sd = torch.sqrt((a * (h - mu.unsqueeze(1)) ** 2).sum(1) + 1e-6)
        return torch.cat([mu, sd], -1)

    def forward(self, x):
        e = self.embed(x)
        return self.head(e), e


def spec_augment(mel, freq_mask_param=8, time_mask_param=30, num_freq_masks=1, num_time_masks=1):
    """Randomly zero out contiguous frequency bands and time steps of a (n_mels, n_frames)
    log-mel spectrogram. Standard SpecAugment, no time-warping (not worth it for a 3 s clip).
    Operates on a numpy array and returns a new array; fills masked regions with the clip's own
    mean so it doesn't introduce an artificial silence/energy spike.
    """
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    fill = mel.mean()
    for _ in range(num_freq_masks):
        w = np.random.randint(0, freq_mask_param + 1)
        if w == 0 or w >= n_mels:
            continue
        f0 = np.random.randint(0, n_mels - w)
        mel[f0:f0 + w, :] = fill
    for _ in range(num_time_masks):
        w = np.random.randint(0, time_mask_param + 1)
        if w == 0 or w >= n_frames:
            continue
        t0 = np.random.randint(0, n_frames - w)
        mel[:, t0:t0 + w] = fill
    return mel


class ClipDataset(Dataset):
    def __init__(self, paths, yb, yt, fe, augment=False, freq_mask_param=8, time_mask_param=30):
        self.paths, self.yb, self.yt, self.fe = paths, yb, yt, fe
        self.augment = augment
        self.freq_mask_param, self.time_mask_param = freq_mask_param, time_mask_param

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        y = load_and_preprocess(self.paths[i])[:N_SAMPLES]
        mel = self.fe(y, sampling_rate=SR, return_tensors="np", padding="max_length",
                      max_length=N_SAMPLES).input_features[0]           # (n_mels, 300)
        if self.augment:
            mel = spec_augment(mel, self.freq_mask_param, self.time_mask_param)
        return torch.from_numpy(mel), int(self.yb[i]), int(self.yt[i])


def loss_fn(out, yb, yt, pos_w, type_w):
    bce = nn.functional.binary_cross_entropy_with_logits(out[:, 0], yb.float(), pos_weight=pos_w)
    m = yb == 1
    ce = nn.functional.cross_entropy(out[m, 1:], yt[m], weight=type_w) if m.any() else out.sum() * 0
    return bce + ce


@torch.no_grad()
def run_eval(model, loader, dev, pos_w, type_w):
    model.eval(); tot, n = 0.0, 0
    for x, yb, yt in loader:
        x, yb, yt = x.to(dev), yb.to(dev), yt.to(dev)
        out, _ = model(x); tot += loss_fn(out, yb, yt, pos_w, type_w).item() * len(x); n += len(x)
    return tot / max(n, 1)


@torch.no_grad()
def extract_all(model, loader, dev):
    model.eval(); E, P, L = [], [], []
    for x, _, _ in loader:
        out, e = model(x.to(dev))
        E.append(e.cpu().numpy()); P.append(torch.sigmoid(out[:, 0]).cpu().numpy()); L.append(out[:, 1:].cpu().numpy())
    return np.vstack(E), np.concatenate(P), np.vstack(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--whisper", default=WHISPER_NAME)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only first N clips")
    ap.add_argument("--augment", action="store_true",
                    help="SpecAugment (freq/time masking) on the training split. Off by default.")
    ap.add_argument("--freq_mask", type=int, default=8, help="max width of a masked frequency band (mel bins)")
    ap.add_argument("--time_mask", type=int, default=30, help="max width of a masked time span (frames)")
    ap.add_argument("--schedule", action="store_true",
                    help="linear warmup + cosine decay instead of a flat LR. Off by default.")
    ap.add_argument("--warmup_frac", type=float, default=0.1,
                    help="fraction of total training steps used for warmup when --schedule is set")
    ap.add_argument("--tag", default="",
                    help="suffix for output files, e.g. --tag augonly -> ft_seed0_augonly.npz. "
                         "Lets you A/B different configs for the same seed without overwriting "
                         "earlier results. Default '' keeps the original ft_seed<seed>.npz naming.")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    d = np.load(FEAT_FILE, allow_pickle=True)
    paths, yb, yt, groups = d["path"], d["y_bin"], d["y_type"], d["group"]
    if a.limit:
        paths, yb, yt, groups = paths[:a.limit], yb[:a.limit], yt[:a.limit], groups[:a.limit]
    tr, va, te = group_split(groups, a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = WhisperFeatureExtractor.from_pretrained(a.whisper) if a.whisper != "__tiny_test__" else WhisperFeatureExtractor()

    def loader(idx, shuffle, augment=False):
        ds = ClipDataset(paths[idx], yb[idx], yt[idx], fe, augment=augment,
                         freq_mask_param=a.freq_mask, time_mask_param=a.time_mask)
        return DataLoader(ds, batch_size=a.batch, shuffle=shuffle, num_workers=a.workers, drop_last=False)

    pos = yb[tr].sum()
    pos_w = torch.tensor((len(tr) - pos) / max(pos, 1), dtype=torch.float32, device=dev)
    cnt = np.bincount(yt[tr][yt[tr] >= 0], minlength=len(TYPE_CLASSES)).clip(1)
    type_w = torch.tensor(cnt.sum() / (len(cnt) * cnt), dtype=torch.float32, device=dev)

    model = WhisperFT(a.whisper, a.unfreeze).to(dev)
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc},
                             {"params": list(model.att.parameters()) + list(model.head.parameters()), "lr": a.lr_head}],
                            weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    tl, vl = loader(tr, True, augment=a.augment), loader(va, False, augment=False)

    sched = None
    if a.schedule:
        total_steps = a.epochs * len(tl)
        warmup_steps = max(1, int(total_steps * a.warmup_frac))
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps, cosine decay to 0")

    best, best_state = 1e9, None
    for ep in range(a.epochs):
        model.train()
        for k, (x, b, t) in enumerate(tl):
            x, b, t = x.to(dev), b.to(dev), t.to(dev)
            with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
                out, _ = model(x)
            loss = loss_fn(out.float(), b, t, pos_w, type_w)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched is not None:
                sched.step()
            if k % 100 == 0:
                lr_now = opt.param_groups[1]["lr"]
                print(f"ep{ep} it{k}/{len(tl)} loss {loss.item():.3f} lr_head {lr_now:.2e}", flush=True)
        v = run_eval(model, vl, dev, pos_w, type_w)
        print(f"epoch {ep}: val loss {v:.4f}")
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    E, P, L = extract_all(model, loader(np.arange(len(paths)), False, augment=False), dev)
    suffix = f"_{a.tag}" if a.tag else ""
    out = DATA_DIR / f"ft_seed{a.seed}{suffix}.npz"
    np.savez_compressed(out, emb=E, p_bin=P, type_logits=L)
    torch.save(model.state_dict(), DATA_DIR / f"whisper_ft_seed{a.seed}{suffix}.pt")
    print("saved", out)
    if suffix:
        print(f"(tagged run -- train_finetuned.py currently expects ft_seed{a.seed}.npz; "
              f"use --ft-file to point it at this file, see below)")


if __name__ == "__main__":
    main()
