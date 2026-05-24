from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import torch


def _str_list(values: List[str]) -> List[str]:
    return list(values)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adversarial Examples Using Style Transfer")
    parser.add_argument("--content_dir", type=Path, default=Path("data/content"))
    parser.add_argument("--style_dir", type=Path, default=Path("data/style"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--target_name", type=str, default="cinema")
    parser.add_argument("--target_idx", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_prefix", type=str, default="adv_style_vgg19")
    return parser


def _safe_name(text: str) -> str:
    """把 run 名称整理成适合做文件夹名的字符串。"""
    allowed = []
    for char in text.strip().replace(" ", "_"):
        allowed.append(char if char.isalnum() or char in "-_." else "_")
    return "".join(allowed).strip("._") or "run"


def _make_run_name(args) -> str:
    """自动生成本次实验的 run 名称。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = args.target_name if args.target_idx is None else f"idx{args.target_idx}"
    parts = [
        timestamp,
        f"target-{target}",
        f"n{args.num_images}",
        f"s{getattr(args, 'steps', 'eval')}",
    ]
    return _safe_name("_".join(str(part) for part in parts))


def _unique_run_dir(runs_root: Path, run_name: str) -> tuple[str, Path]:
    """如果 run 目录已存在，自动追加序号，避免覆盖旧实验。"""
    candidate = runs_root / run_name
    if not candidate.exists():
        return run_name, candidate
    for idx in range(1, 10000):
        next_name = f"{run_name}_{idx:03d}"
        candidate = runs_root / next_name
        if not candidate.exists():
            return next_name, candidate
    raise RuntimeError(f"无法生成不重复的 run 目录: {runs_root / run_name}")


def _write_args(args, path: Path) -> None:
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _finalize_attack_args(args):
    """创建按 run 归档的训练输出目录，并保存参数。"""
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("警告: CUDA 不可用，自动切换到 CPU。")
        args.device = "cpu"

    args.out_root = Path(args.out_dir)
    requested_run_name = _safe_name(args.run_name) if args.run_name else _make_run_name(args)
    args.run_name, args.out_dir = _unique_run_dir(args.out_root / "runs", requested_run_name)
    args.adv_dir = args.out_dir / "adv_images"
    args.figure_dir = args.out_dir / "figures"
    args.log_dir = args.out_dir / "logs"
    for directory in [args.out_root, args.out_dir, args.adv_dir, args.figure_dir, args.log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    _write_args(args, args.out_dir / "args.json")
    latest_path = args.out_root / "latest_run.txt"
    latest_path.write_text(str(args.out_dir.resolve()), encoding="utf-8")
    return args


def _finalize_eval_args(args):
    """解析评估目录；默认使用 outputs/latest_run.txt 指向的最近一次训练。"""
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("警告: CUDA 不可用，自动切换到 CPU。")
        args.device = "cpu"

    args.out_root = Path(args.out_root)
    if args.run_dir is None and args.adv_dir is None:
        latest_path = args.out_root / "latest_run.txt"
        if not latest_path.exists():
            raise FileNotFoundError(
                f"找不到最近一次训练记录: {latest_path}。请使用 --run_dir 或 --adv_dir 指定评估目录。"
            )
        args.run_dir = Path(latest_path.read_text(encoding="utf-8").strip())

    if args.run_dir is not None:
        args.run_dir = Path(args.run_dir)
        args.adv_dir = args.run_dir / "adv_images"
        if args.out_dir is None:
            args.out_dir = args.run_dir / "eval"
    else:
        args.adv_dir = Path(args.adv_dir)
        if args.out_dir is None:
            args.out_dir = args.out_root / "eval"

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_args(args, args.out_dir / "eval_args.json")
    return args


def prepare_attack_args():
    """解析攻击参数，并创建本次 run 的输出目录。"""
    parser = _common_parser()
    parser.add_argument("--run_name", type=str, default=None, help="本次实验名称；默认按时间和参数自动生成")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lambda_adv", type=float, default=100)
    parser.add_argument("--style_weight", type=float, default=200000)
    parser.add_argument("--content_weight", type=float, default=10)
    parser.add_argument("--tv_weight", type=float, default=0.0001)
    parser.add_argument(
        "--style_layers",
        nargs="+",
        default=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
    )
    parser.add_argument("--content_layers", nargs="+", default=["conv4_2"])
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--make_zip", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    args = parser.parse_args()
    args.style_layers = _str_list(args.style_layers)
    args.content_layers = _str_list(args.content_layers)
    return _finalize_attack_args(args)


def prepare_eval_args():
    """解析评估参数。可指定 --run_dir，也可默认评估最新一次 run。"""
    parser = argparse.ArgumentParser(description="Evaluate generated adversarial images")
    parser.add_argument("--out_root", type=Path, default=Path("outputs"))
    parser.add_argument("--run_dir", type=Path, default=None, help="某次训练 run 目录，例如 outputs/runs/xxx")
    parser.add_argument("--adv_dir", type=Path, default=None, help="直接指定 adv_images 目录")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--target_name", type=str, default="cinema")
    parser.add_argument("--target_idx", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_prefix", type=str, default="adv_style_vgg19")
    args = parser.parse_args()
    return _finalize_eval_args(args)
