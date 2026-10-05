"""Evaluate a trained checkpoint on the held-out TEST set, and draw Grad-CAM for its worst mistakes.

Outputs (in --out_dir, default: the checkpoint's folder): test_metrics.json, test_probs.npy,
confusion_matrix.png, errors_gradcam.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import batch_to_rgb_uint8, load_data
from gradcam import GradCAM, overlay
from models import gradcam_layer, load_checkpoint, preprocess_rgb_uint8
from utils import CLASS_NAMES, compute_metrics, get_device, save_json


def predict(model, X, device, bs=64):
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = preprocess_rgb_uint8(batch_to_rgb_uint8(X[i:i + bs]), device)
            probs.append(model(x).softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def plot_confusion(cm, path, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_errors", type=int, default=6, help="how many mistakes to explain with Grad-CAM")
    args = ap.parse_args()

    device = get_device()
    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, arch = load_checkpoint(args.ckpt, device)
    _, _, X_test, y_test = load_data(args.data_dir)

    probs = predict(model, X_test, device)
    pred = (probs > 0.5).astype(int)
    m = compute_metrics(y_test, pred, probs)
    save_json(m, out_dir / "test_metrics.json")
    np.save(out_dir / "test_probs.npy", probs)
    plot_confusion(m["confusion_matrix"], out_dir / "confusion_matrix.png", f"{arch} - test set")
    print(f"TEST ({len(y_test)} images) | " + " | ".join(f"{k} {v:.3f}" for k, v in m.items() if k != "confusion_matrix"))
    print("confusion matrix [[TN FP],[FN TP]]:", m["confusion_matrix"])

    # --- Grad-CAM on the most confident mistakes (false negatives and false positives) ---
    wrong_conf = np.where(pred == 1, probs, 1 - probs)          # confidence in the wrong answer
    fn = [i for i in np.argsort(-wrong_conf) if y_test[i] == 1 and pred[i] == 0][: args.n_errors // 2]
    fp = [i for i in np.argsort(-wrong_conf) if y_test[i] == 0 and pred[i] == 1][: args.n_errors - len(fn)]
    chosen = fn + fp
    if not chosen:
        print("no mistakes to plot")
        return
    cam_engine = GradCAM(model, gradcam_layer(model, arch))
    fig, axes = plt.subplots(2, len(chosen), figsize=(2.6 * len(chosen), 5.6), squeeze=False)
    for k, i in enumerate(chosen):
        img = batch_to_rgb_uint8(X_test[i])
        cam, p = cam_engine(preprocess_rgb_uint8(img[None], device), class_idx=int(pred[i]))
        axes[0, k].imshow(img)
        axes[0, k].set_title(f"true: {CLASS_NAMES[y_test[i]]}\npred: {CLASS_NAMES[pred[i]]} ({p[pred[i]]:.2f})", fontsize=8)
        axes[1, k].imshow(overlay(img, cam))
        for a in axes[:, k]:
            a.axis("off")
    fig.suptitle("Most confident test mistakes (top: image, bottom: Grad-CAM for the predicted class)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "errors_gradcam.png", dpi=140)
    print("saved", out_dir / "errors_gradcam.png")


if __name__ == "__main__":
    main()
