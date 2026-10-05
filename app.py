"""Gradio demo: upload a photo -> spalling probability + Grad-CAM explanation.
Run:  python src/app.py --ckpt outputs/resnet50/best.pt
"""
import argparse

import gradio as gr
import numpy as np
from PIL import Image

from gradcam import GradCAM, overlay
from models import gradcam_layer, load_checkpoint, preprocess_rgb_uint8
from utils import CLASS_NAMES, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--examples", default="examples", help="optional folder of example images")
    args = ap.parse_args()

    device = get_device()
    model, arch = load_checkpoint(args.ckpt, device)
    cam_engine = GradCAM(model, gradcam_layer(model, arch))

    def predict(pil_img):
        if pil_img is None:
            return None, None
        img = np.array(pil_img.convert("RGB").resize((224, 224), Image.BILINEAR))   # RGB uint8
        cam, probs = cam_engine(preprocess_rgb_uint8(img[None], device))
        return {CLASS_NAMES[0]: float(probs[0]), CLASS_NAMES[1]: float(probs[1])}, overlay(img, cam)

    from pathlib import Path
    ex = sorted(str(p) for p in Path(args.examples).glob("*.png")) if Path(args.examples).is_dir() else None
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Photo of a structure"),
        outputs=[gr.Label(num_top_classes=2, label="Prediction"), gr.Image(label="Grad-CAM (regions that drove the decision)")],
        examples=ex or None,
        title="Spalling detection (PHI-Net Task 3)",
        description=f"Model: {arch}. Course mini-project demo; not for real structural safety decisions.",
        flagging_mode="never",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
