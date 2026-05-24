from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, Tuple

# Windows/Conda 环境中 torch、numpy、matplotlib 可能加载不同 OpenMP 运行时。
# 这里放在数值库 import 前，避免保存可视化图时因重复初始化中断进程。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from tqdm import tqdm

from data.data_entry import build_dataloader
from loss import compute_total_loss
from metrics import compute_attack_metrics
from model.model_entry import build_models
from options import prepare_attack_args
from utils.image_utils import make_unique_path, save_tensor_image
from utils.imagenet_labels import find_target_idx, get_imagenet_labels, label_for_idx
from utils.logger import SimpleLogger
from utils.viz import save_comparison_figure
from utils.zip_utils import make_submission_zip


def set_seed(seed: int) -> None:
    """设置随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Attacker:
    """固定 VGG19 参数，只优化输入图片 x_adv 的攻击器。"""

    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.labels = get_imagenet_labels()
        self.target_idx = args.target_idx if args.target_idx is not None else find_target_idx(args.target_name, self.labels)
        self.target_label = label_for_idx(self.target_idx, self.labels)
        self.dataloader = build_dataloader(args)
        self.classifier, self.feature_extractor = build_models(args)
        self.rows = []

        # 明确保证优化器不会接触 VGG19 参数。
        for module in [self.classifier, self.feature_extractor]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def attack(self) -> None:
        """遍历数据集，逐张生成对抗风格迁移图片。"""
        print(f"目标类别: {self.target_idx} - {self.target_label}")
        for sample in tqdm(self.dataloader, desc="Attacking images"):
            row = self.attack_one_image(sample)
            self.rows.append(row)

        results_path = self.args.out_dir / "results.csv"
        self._write_results(results_path)
        print(f"已保存结果表: {results_path}")

        if self.args.make_zip:
            zip_path = make_submission_zip(Path(__file__).resolve().parent, self.args.out_dir)
            print(f"已生成提交包: {zip_path}")

    def attack_one_image(self, sample: Dict[str, object]) -> Dict[str, object]:
        """对单张内容图优化 x_adv 并保存所有产物。"""
        args = self.args
        content = sample["content_tensor"].to(args.device)
        style = sample["style_tensor"].to(args.device)
        index = int(sample["index"].item() if torch.is_tensor(sample["index"]) else sample["index"])
        content_path = Path(sample["content_path"][0] if isinstance(sample["content_path"], list) else sample["content_path"])
        style_path = Path(sample["style_path"][0] if isinstance(sample["style_path"], list) else sample["style_path"])
        stem = f"{args.save_prefix}_{index:02d}_{content_path.stem}"

        log_path = args.log_dir / f"{stem}.log"
        adv_path = make_unique_path(args.adv_dir / f"{stem}.png")
        figure_path = make_unique_path(args.figure_dir / f"{stem}.png")

        x_adv = content.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_adv], lr=args.lr)

        with torch.no_grad():
            original_logits = self.classifier(content)
            original_metrics = compute_attack_metrics(original_logits, self.target_idx, self.labels)
            content_features = {
                key: value.detach()
                for key, value in self.feature_extractor(content).items()
            }
            style_features = {
                key: value.detach()
                for key, value in self.feature_extractor(style).items()
            }

        steps_used = args.steps
        final_metrics = original_metrics
        with SimpleLogger(log_path) as logger:
            logger.log(f"content={content_path}")
            logger.log(f"style={style_path}")
            logger.log(f"target={self.target_idx} {self.target_label}")
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                logits = self.classifier(x_adv)
                adv_features = self.feature_extractor(x_adv)
                total_loss, loss_dict = compute_total_loss(
                    logits=logits,
                    adv_features=adv_features,
                    content_features=content_features,
                    style_features=style_features,
                    x_adv=x_adv,
                    target_idx=self.target_idx,
                    style_layers=args.style_layers,
                    content_layers=args.content_layers,
                    lambda_adv=args.lambda_adv,
                    style_weight=args.style_weight,
                    content_weight=args.content_weight,
                    tv_weight=args.tv_weight,
                )
                total_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    x_adv.clamp_(0, 1)

                should_log = step == 1 or step % args.print_freq == 0 or step == args.steps
                if should_log:
                    final_metrics = compute_attack_metrics(logits, self.target_idx, self.labels)
                    logger.log(
                        "step={step} total={total:.6f} adv={adv:.6f} style={style:.6f} "
                        "content={content_loss:.6f} tv={tv:.6f} pred={pred} target_prob={target_prob:.6f}".format(
                            step=step,
                            total=float(total_loss.detach().item()),
                            adv=loss_dict["adv_loss"],
                            style=loss_dict["style_loss"],
                            content_loss=loss_dict["content_loss"],
                            tv=loss_dict["tv_loss"],
                            pred=final_metrics["pred_label"],
                            target_prob=final_metrics["target_prob"],
                        )
                    )

                if args.save_freq > 0 and step % args.save_freq == 0 and step != args.steps:
                    preview_dir = args.out_dir / "intermediate"
                    save_tensor_image(x_adv, preview_dir / f"{stem}_step{step:04d}.png")

                if args.early_stop:
                    with torch.no_grad():
                        current_metrics = compute_attack_metrics(self.classifier(x_adv), self.target_idx, self.labels)
                    if current_metrics["attack_success"] and current_metrics["target_prob"] > 0.8:
                        steps_used = step
                        final_metrics = current_metrics
                        logger.log(f"early_stop step={step} target_prob={current_metrics['target_prob']:.6f}")
                        break

            with torch.no_grad():
                final_logits = self.classifier(x_adv)
                final_metrics = compute_attack_metrics(final_logits, self.target_idx, self.labels)

        save_tensor_image(x_adv, adv_path)
        save_comparison_figure(
            content=content,
            style=style,
            adv=x_adv,
            original_pred=f"{original_metrics['pred_label']} ({original_metrics['pred_prob']:.3f})",
            final_pred=f"{final_metrics['pred_label']} ({final_metrics['pred_prob']:.3f})",
            target_prob=float(final_metrics["target_prob"]),
            success=bool(final_metrics["attack_success"]),
            save_path=figure_path,
        )

        return {
            "index": index,
            "content_path": str(content_path),
            "style_path": str(style_path),
            "adv_image_path": str(adv_path),
            "figure_path": str(figure_path),
            "original_pred_label": original_metrics["pred_label"],
            "original_pred_prob": original_metrics["pred_prob"],
            "final_pred_label": final_metrics["pred_label"],
            "final_pred_prob": final_metrics["pred_prob"],
            "target_label": self.target_label,
            "target_prob": final_metrics["target_prob"],
            "attack_success": final_metrics["attack_success"],
            "steps_used": steps_used,
            "top5_labels": final_metrics["top5_labels"],
            "top5_probs": final_metrics["top5_probs"],
        }

    def _write_results(self, path: Path) -> None:
        """写出 results.csv。"""
        if not self.rows:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


def main() -> None:
    args = prepare_attack_args()
    attacker = Attacker(args)
    attacker.attack()


if __name__ == "__main__":
    main()
