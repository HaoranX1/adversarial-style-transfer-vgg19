from __future__ import annotations

from pathlib import Path
from typing import Dict

from torch.utils.data import Dataset

from utils.image_utils import list_image_files, load_image


class AdvStyleDataset(Dataset):
    """读取内容图与风格图，并按 index 轮流匹配风格图。"""

    def __init__(self, content_dir: Path, style_dir: Path, num_images: int = 10, image_size: int = 224):
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.num_images = num_images
        self.image_size = image_size

        content_images = list_image_files(self.content_dir)
        style_images = list_image_files(self.style_dir)
        if not content_images:
            raise ValueError(f"content_dir 中没有图片: {self.content_dir}")
        if not style_images:
            raise ValueError(f"style_dir 中没有图片: {self.style_dir}")
        if len(content_images) < num_images:
            raise ValueError(
                f"内容图片数量不足: 需要 {num_images} 张，实际只有 {len(content_images)} 张。"
            )

        self.content_images = content_images[:num_images]
        self.style_images = style_images

    def __len__(self) -> int:
        return len(self.content_images)

    def __getitem__(self, index: int) -> Dict[str, object]:
        content_path = self.content_images[index]
        style_path = self.style_images[index % len(self.style_images)]
        return {
            "content_tensor": load_image(content_path, self.image_size),
            "style_tensor": load_image(style_path, self.image_size),
            "content_path": str(content_path),
            "style_path": str(style_path),
            "index": index,
        }
