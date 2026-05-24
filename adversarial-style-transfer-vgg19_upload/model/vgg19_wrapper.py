from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG19_Weights

from utils.image_utils import normalize_for_vgg


VGG_LAYER_NAME_MAP = {
    0: "conv1_1",
    5: "conv2_1",
    10: "conv3_1",
    19: "conv4_1",
    21: "conv4_2",
    28: "conv5_1",
}


def _load_pretrained_vgg19() -> models.VGG:
    """加载 ImageNet 预训练 VGG19，并给权重下载失败提供清晰提示。"""
    try:
        return models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    except Exception as exc:
        try:
            return models.vgg19(pretrained=True)
        except Exception as fallback_exc:
            raise RuntimeError(
                "加载 torchvision 预训练 VGG19 失败。请检查网络连接，或提前下载权重到 "
                "torch cache 后重试。"
            ) from fallback_exc


class VGG19Classifier(nn.Module):
    """固定参数的 VGG19 分类器，输入为 [0,1] RGB tensor。"""

    def __init__(self):
        super().__init__()
        self.vgg = _load_pretrained_vgg19()
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_for_vgg(x)
        return self.vgg(x)


class VGG19FeatureExtractor(nn.Module):
    """固定参数的 VGG19 中间层特征提取器，输入为 [0,1] RGB tensor。"""

    def __init__(self, layers: Iterable[str]):
        super().__init__()
        self.requested_layers = set(layers)
        valid_layers = set(VGG_LAYER_NAME_MAP.values())
        unknown = self.requested_layers - valid_layers
        if unknown:
            raise ValueError(f"未知 VGG19 特征层: {sorted(unknown)}，可选: {sorted(valid_layers)}")

        vgg = _load_pretrained_vgg19()
        self.features = vgg.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = normalize_for_vgg(x)
        outputs: Dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            name = VGG_LAYER_NAME_MAP.get(idx)
            if name in self.requested_layers:
                outputs[name] = x
        return outputs
