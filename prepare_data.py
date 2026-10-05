"""Convert the float32 (mean-subtracted BGR) arrays to uint8 BGR. Lossless, ~4x smaller.

Usage:  python src/prepare_data.py --src ~/Downloads/task3 --dst data
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="folder with the original task3_*.npy files")
ap.add_argument("--dst", default="data")
args = ap.parse_args()
src, dst = Path(args.src), Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)

for f in sorted(src.glob("*.npy")):
    name = f.stem
    if "u8" in name:
        continue
    if "X" not in name and "x_" not in name.lower():
        shutil.copy(f, dst / f.name)            # labels are copied as they are
        print("copied", f.name)
        continue
    X = np.load(f, mmap_mode="r")
    out = np.lib.format.open_memmap(dst / f"{name}_u8.npy", mode="w+", dtype=np.uint8, shape=X.shape)
    for i in range(0, len(X), 500):
        out[i:i + 500] = np.clip(np.rint(X[i:i + 500] + MEAN), 0, 255).astype(np.uint8)
    out.flush()
    print("converted", f.name, X.shape)
