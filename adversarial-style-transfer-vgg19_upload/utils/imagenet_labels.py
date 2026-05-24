from __future__ import annotations

from typing import Iterable, List, Optional

try:
    from torchvision.models import VGG19_Weights
except Exception:  # pragma: no cover - torchvision may be unavailable during static checks
    VGG19_Weights = None


FALLBACK_LABELS: List[str] = [""] * 1000
FALLBACK_LABELS[498] = "cinema, movie theater, movie theatre, movie house, picture palace"


def get_imagenet_labels() -> List[str]:
    """读取 torchvision 内置 ImageNet 类别名，失败时使用最小兜底标签表。"""
    if VGG19_Weights is not None:
        try:
            categories = VGG19_Weights.IMAGENET1K_V1.meta.get("categories", [])
            if len(categories) == 1000:
                return list(categories)
        except Exception:
            pass
    return FALLBACK_LABELS


def find_target_idx(target_name: str = "cinema", labels: Optional[Iterable[str]] = None) -> int:
    """根据类别名查找目标类别，优先匹配 cinema / movie theater。"""
    label_list = list(labels) if labels is not None else get_imagenet_labels()
    needles = [target_name.lower(), "movie theater", "cinema"]
    for idx, label in enumerate(label_list):
        text = label.lower()
        if any(needle in text for needle in needles):
            return idx
    raise ValueError(
        f"无法在 ImageNet 标签中找到目标类别 {target_name!r}，"
        "请通过 --target_idx 手动指定类别 id。"
    )


def label_for_idx(idx: int, labels: Optional[Iterable[str]] = None) -> str:
    """返回类别 id 对应的可读标签。"""
    label_list = list(labels) if labels is not None else get_imagenet_labels()
    if 0 <= idx < len(label_list) and label_list[idx]:
        return label_list[idx]
    return f"class_{idx}"
