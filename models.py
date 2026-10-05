"""Model construction (torchvision, ImageNet-pretrained) and input preprocessing."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

ARCHS = ["resnet50", "efficientnet_b0", "mobilenet_v2"]

# torchvision ImageNet statistics (RGB, inputs scaled to [0, 1])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(arch: str, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 2))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, 2))
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, 2))
    else:
        raise ValueError(f"arch must be one of {ARCHS}")
    return m


def head_and_blocks(model: nn.Module, arch: str):
    """(new classification head, [last backbone blocks to unfreeze in stage 2])."""
    if arch == "resnet50":
        return model.fc, [model.layer4]
    if arch == "efficientnet_b0":
        return model.classifier, [model.features[-3:]]
    if arch == "mobilenet_v2":
        return model.classifier, [model.features[-4:]]
    raise ValueError(arch)


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    return model.layer4[-1] if arch == "resnet50" else model.features[-1]


def preprocess_rgb_uint8(batch: np.ndarray, device) -> torch.Tensor:
    """(N,224,224,3) uint8 RGB  ->  normalised float tensor (N,3,224,224) on `device`."""
    t = torch.from_numpy(np.ascontiguousarray(batch)).to(device).permute(0, 3, 1, 2).float().div(255)
    return (t - MEAN.to(device)) / STD.to(device)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["arch"]
