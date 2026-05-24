from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

# Windows/Conda 环境中可能出现 OpenMP 运行时重复初始化，评估入口同样兜底。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from metrics import compute_attack_metrics
from model.vgg19_wrapper import VGG19Classifier
from options import prepare_eval_args
from utils.image_utils import list_image_files, load_image
from utils.imagenet_labels import find_target_idx, get_imagenet_labels, label_for_idx


class Evaluator:
    """对某个 run 的 adv_images 重新跑 VGG19 评估。"""

    def __init__(self, args):
        self.args = args
        self.labels = get_imagenet_labels()
        self.target_idx = args.target_idx if args.target_idx is not None else find_target_idx(args.target_name, self.labels)
        self.target_label = label_for_idx(self.target_idx, self.labels)
        self.classifier = VGG19Classifier().to(args.device).eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

    def evaluate(self) -> None:
        """生成 eval_results.csv 和 result.txt。"""
        images = list_image_files(self.args.adv_dir)
        if not images:
            raise ValueError(f"adv_dir 中没有生成图片: {self.args.adv_dir}")

        rows: List[Dict[str, object]] = []
        for path in images:
            image = load_image(path, self.args.image_size).unsqueeze(0).to(self.args.device)
            with torch.no_grad():
                logits = self.classifier(image)
            metrics = compute_attack_metrics(logits, self.target_idx, self.labels)
            rows.append(
                {
                    "adv_image_path": str(path),
                    "pred_idx": metrics["pred_idx"],
                    "pred_label": metrics["pred_label"],
                    "pred_prob": metrics["pred_prob"],
                    "target_label": self.target_label,
                    "target_prob": metrics["target_prob"],
                    "attack_success": metrics["attack_success"],
                    "top5_labels": metrics["top5_labels"],
                    "top5_probs": metrics["top5_probs"],
                }
            )

        csv_path = self.args.out_dir / "eval_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        success_count = sum(1 for row in rows if row["attack_success"])
        avg_target_prob = sum(float(row["target_prob"]) for row in rows) / len(rows)
        result_path = self.args.out_dir / "result.txt"
        lines = [
            f"评估目录: {self.args.adv_dir}",
            f"总图片数: {len(rows)}",
            f"攻击成功数: {success_count}",
            f"attack success rate: {success_count / len(rows):.4f}",
            f"平均 target probability: {avg_target_prob:.6f}",
            "",
            "每张图片 top1 预测结果:",
        ]
        for row in rows:
            lines.append(
                f"{Path(row['adv_image_path']).name}: {row['pred_label']} "
                f"({float(row['pred_prob']):.6f}), target_prob={float(row['target_prob']):.6f}"
            )
        result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已保存评估结果: {csv_path}")
        print(f"已保存摘要: {result_path}")


def main() -> None:
    args = prepare_eval_args()
    Evaluator(args).evaluate()


if __name__ == "__main__":
    main()
