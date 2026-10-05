"""Loading and splitting the PHI-Net Task 3 (spalling) arrays. NumPy only, no torch needed.

Original files are float32, BGR order, ImageNet channel means subtracted (Keras 'caffe' style).
prepare_data.py converts them to uint8 BGR (about 4x smaller). Everything here accepts either.
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def batch_to_rgb_uint8(batch) -> np.ndarray:
    """(H,W,3) or (N,H,W,3), uint8-BGR or mean-subtracted float BGR  ->  uint8 RGB."""
    b = np.asarray(batch)
    if b.dtype != np.uint8:
        b = np.clip(np.rint(b + BGR_MEAN), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(b[..., ::-1])


def _find(d: Path, key: str) -> Path:
    # prefer the uint8 version; fall back to any matching .npy
    cands = sorted(d.glob(f"*{key}*u8*.npy")) or sorted(d.glob(f"*{key}*.npy"))
    if not cands:
        raise FileNotFoundError(f"No file matching '*{key}*.npy' in {d.resolve()}")
    return cands[0]


def _labels(path: Path) -> np.ndarray:
    y = np.load(path)
    return y.argmax(1).astype(np.int64) if y.ndim == 2 else y.astype(np.int64)


def load_data(data_dir):
    """Returns X_train, y_train, X_test, y_test. X arrays are memory-mapped (not loaded into RAM)."""
    d = Path(data_dir)
    X_train = np.load(_find(d, "X_train"), mmap_mode="r")
    X_test = np.load(_find(d, "X_test"), mmap_mode="r")
    y_train = _labels(_find(d, "y_train"))
    y_test = _labels(_find(d, "y_test"))
    assert len(X_train) == len(y_train), f"X_train {len(X_train)} vs y_train {len(y_train)}"
    assert len(X_test) == len(y_test), f"X_test {len(X_test)} vs y_test {len(y_test)}"
    assert X_train.shape[1:] == (224, 224, 3), X_train.shape
    # Note: the provided label files are sorted by class (all 0s, then all 1s). Always shuffle/stratify; never slice.
    return X_train, y_train, X_test, y_test


def split_train_val(y_train, val_frac=0.15, seed=42):
    """Stratified train/validation indices. The official test set is never touched here."""
    idx = np.arange(len(y_train))
    tr, va = train_test_split(idx, test_size=val_frac, stratify=y_train, random_state=seed)
    return np.sort(tr), np.sort(va)
