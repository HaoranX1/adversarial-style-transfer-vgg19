from __future__ import annotations

from torch.utils.data import DataLoader

from data.adv_style_dataset import AdvStyleDataset


def build_dataloader(args) -> DataLoader:
    """根据参数构建 batch_size=1 的 DataLoader。"""
    dataset = AdvStyleDataset(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        num_images=args.num_images,
        image_size=args.image_size,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
