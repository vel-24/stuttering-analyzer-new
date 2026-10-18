"""Central configuration for the stuttering detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"                      # SEP-28k wav clips (see README step 1)
LABELS_CSV = DATA_DIR / "SEP-28k_labels.csv"        # from apple/ml-stuttering-events-dataset
FEAT_FILE = DATA_DIR / "features.npz"
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "sessions.db"

SR = 16000                       # Whisper expects 16 kHz
CLIP_SEC = 3.0                   # SEP-28k clips are 3 s long
WHISPER_NAME = "openai/whisper-base"   # use whisper-small for better accuracy (768-d)

VOTE_THRESHOLD = 2               # >=2 of 3 annotators must agree on a label
TYPE_CLASSES = ["Block", "Prolongation", "Repetition", "Interjection"]
SEED = 42

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
