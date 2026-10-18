"""Episode-wise (group) train/val/test splits, reproducible per seed."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def group_split(groups, seed: int = 42, test_size=0.2, val_size=0.1):
    """Clips from the same show/episode never appear in more than one split."""
    groups = np.asarray(groups)
    idx = np.arange(len(groups))
    tv, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, groups=groups))
    rel_val = val_size / (1 - test_size)
    tr, va = next(GroupShuffleSplit(1, test_size=rel_val, random_state=seed).split(tv, groups=groups[tv]))
    return tv[tr], tv[va], te
