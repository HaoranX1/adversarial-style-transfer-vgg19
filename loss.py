from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """计算归一化 Gram matrix，用于描述纹理风格。"""
    b, c, h, w = features.shape
    flat = features.view(b, c, h * w)
    gram = torch.bmm(flat, flat.transpose(1, 2))
    return gram / (c * h * w)


def compute_style_loss(
    adv_features: Dict[str, torch.Tensor],
    style_features: Dict[str, torch.Tensor],
    style_layers: Iterable[str],
) -> torch.Tensor:
    """多层 Gram matrix 的 MSE 风格损失。"""
    loss = None
    for layer in style_layers:
        layer_loss = F.mse_loss(gram_matrix(adv_features[layer]), gram_matrix(style_features[layer]))
        loss = layer_loss if loss is None else loss + layer_loss
    return loss


def compute_content_loss(
    adv_features: Dict[str, torch.Tensor],
    content_features: Dict[str, torch.Tensor],
    content_layers: Iterable[str],
) -> torch.Tensor:
    """内容特征 MSE 损失，用于保留原图结构。"""
    loss = None
    for layer in content_layers:
        layer_loss = F.mse_loss(adv_features[layer], content_features[layer])
        loss = layer_loss if loss is None else loss + layer_loss
    return loss


def compute_adv_loss(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    """目标攻击交叉熵损失，使分类结果靠近目标类别。"""
    target = torch.full((logits.shape[0],), target_idx, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def compute_tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation 平滑损失，减少高频噪声。"""
    h_diff = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    w_diff = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return h_diff + w_diff


def compute_total_loss(
    logits: torch.Tensor,
    adv_features: Dict[str, torch.Tensor],
    content_features: Dict[str, torch.Tensor],
    style_features: Dict[str, torch.Tensor],
    x_adv: torch.Tensor,
    target_idx: int,
    style_layers: Iterable[str],
    content_layers: Iterable[str],
    lambda_adv: float,
    style_weight: float,
    content_weight: float,
    tv_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """汇总 adversarial/style/content/TV 损失并返回日志字典。"""
    adv_loss = compute_adv_loss(logits, target_idx)
    style_loss = compute_style_loss(adv_features, style_features, style_layers)
    content_loss = compute_content_loss(adv_features, content_features, content_layers)
    tv_loss = compute_tv_loss(x_adv)
    total_loss = (
        lambda_adv * adv_loss
        + style_weight * style_loss
        + content_weight * content_loss
        + tv_weight * tv_loss
    )
    probs = torch.softmax(logits.detach(), dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    return total_loss, {
        "adv_loss": float(adv_loss.detach().item()),
        "style_loss": float(style_loss.detach().item()),
        "content_loss": float(content_loss.detach().item()),
        "tv_loss": float(tv_loss.detach().item()),
        "target_prob": float(probs[0, target_idx].item()),
        "pred_idx": pred_idx,
    }
