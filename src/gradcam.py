"""Minimal Grad-CAM: highlights the image regions that drove the model's decision."""
import matplotlib
import numpy as np
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W). Returns (cam in [0,1] as HxW numpy array, class probabilities)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = logits.softmax(1)
        idx = int(probs.argmax(1)) if class_idx is None else int(class_idx)
        logits[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), probs[0].detach().cpu().numpy()


def overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    return ((1 - alpha) * img_rgb.astype(np.float32) + alpha * heat).astype(np.uint8)
