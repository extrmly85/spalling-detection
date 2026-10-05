"""Save a few test images as PNGs (for the demo's example gallery and your slides)."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from data import batch_to_rgb_uint8, load_data
from utils import CLASS_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data")
ap.add_argument("--out_dir", default="examples")
ap.add_argument("--n_per_class", type=int, default=4)
args = ap.parse_args()

_, _, X, y = load_data(args.data_dir)
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for c in (0, 1):
    for k, i in enumerate(rng.choice(np.where(y == c)[0], args.n_per_class, replace=False)):
        Image.fromarray(batch_to_rgb_uint8(X[i])).save(out / f"{CLASS_NAMES[c]}_{k}.png")
print("saved examples to", out.resolve())
