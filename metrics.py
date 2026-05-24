from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from utils.imagenet_labels import label_for_idx


def get_topk_predictions(logits: torch.Tensor, labels: Iterable[str], k: int = 5) -> List[Dict[str, object]]:
    """返回 top-k 预测类别名和概率。"""
    probs = torch.softmax(logits.detach(), dim=1)
    top_probs, top_indices = torch.topk(probs, k=k, dim=1)
    label_list = list(labels)
    return [
        {
            "idx": int(idx.item()),
            "label": label_for_idx(int(idx.item()), label_list),
            "prob": float(prob.item()),
        }
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]


def compute_attack_metrics(logits: torch.Tensor, target_idx: int, labels: Iterable[str]) -> Dict[str, object]:
    """计算 top1、目标类别概率和攻击是否成功。"""
    label_list = list(labels)
    probs = torch.softmax(logits.detach(), dim=1)
    pred_prob, pred_idx_tensor = torch.max(probs, dim=1)
    pred_idx = int(pred_idx_tensor.item())
    target_prob = float(probs[0, target_idx].item())
    top5 = get_topk_predictions(logits, label_list, k=5)
    return {
        "pred_idx": pred_idx,
        "pred_label": label_for_idx(pred_idx, label_list),
        "pred_prob": float(pred_prob.item()),
        "target_prob": target_prob,
        "attack_success": pred_idx == target_idx,
        "top5_labels": "; ".join(item["label"] for item in top5),
        "top5_probs": "; ".join(f"{item['prob']:.6f}" for item in top5),
    }


def success_rate(rows: List[Dict[str, object]]) -> float:
    """计算攻击成功率。"""
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get("attack_success"))) / len(rows)
