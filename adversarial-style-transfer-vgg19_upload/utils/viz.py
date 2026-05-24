from __future__ import annotations

import os
from pathlib import Path

# matplotlib 后端初始化前设置，避免 libomp / libiomp5md 冲突导致保存图片失败。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.image_utils import tensor_to_pil


def save_comparison_figure(
    content: torch.Tensor,
    style: torch.Tensor,
    adv: torch.Tensor,
    original_pred: str,
    final_pred: str,
    target_prob: float,
    success: bool,
    save_path: Path,
) -> Path:
    """保存 content/style/adversarial 横向对比图。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    images = [tensor_to_pil(content), tensor_to_pil(style), tensor_to_pil(adv)]
    titles = [
        f"Original\n{original_pred}",
        "Style",
        f"Adv\n{final_pred}\ntarget={target_prob:.3f} {'success' if success else 'fail'}",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
