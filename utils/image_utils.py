from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def list_image_files(directory: Path) -> List[Path]:
    """列出目录下支持的图片文件。"""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"图片目录不存在: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return files


def build_image_transform(image_size: int) -> transforms.Compose:
    """构建统一 resize 到方图并转为 [0,1] tensor 的变换。"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_image(path: Path, image_size: int) -> torch.Tensor:
    """读取图片为 [C,H,W]、范围 [0,1] 的 tensor。"""
    image = Image.open(path).convert("RGB")
    return build_image_transform(image_size)(image)


def normalize_for_vgg(x: torch.Tensor) -> torch.Tensor:
    """将 [0,1] 图像归一化为 VGG19 需要的 ImageNet 分布。"""
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """把 [C,H,W] 或 [1,C,H,W] tensor 转成 PIL 图片。"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> Path:
    """保存 [0,1] tensor 图片。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)
    return path


def make_unique_path(path: Path) -> Path:
    """若目标文件已存在，则追加序号避免覆盖。"""
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 10000):
        candidate = path.with_name(f"{stem}_{i:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法为文件生成不覆盖路径: {path}")
