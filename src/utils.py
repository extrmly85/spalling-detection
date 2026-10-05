"""Small shared helpers: seeding, device selection, metrics, JSON saving."""
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

CLASS_NAMES = ["non-spalling", "spalling"]  # label 0, label 1 (matches the dataset README)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Spalling (label 1) is the positive class."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spalling": float(p[1]),
        "recall_spalling": float(r[1]),
        "f1_spalling": float(f[1]),
        "macro_f1": float(f.mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return m


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
