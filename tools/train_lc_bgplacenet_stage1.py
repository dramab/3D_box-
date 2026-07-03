#!/usr/bin/env python
"""
Train LC-BGPlaceNet Stage 1: source grounding.

使用示例:
     python tools/train_lc_bgplacenet_stage1.py \
        --config configs/lc_bgplacenet_stage1.yaml

     torchrun --nproc_per_node=4 tools/train_lc_bgplacenet_stage1.py \
        --config configs/lc_bgplacenet_stage1.yaml

     python tools/train_lc_bgplacenet_stage1.py \
        --config configs/lc_bgplacenet_stage1.yaml \
        --max-steps 2 --max-train-samples 4 --max-val-samples 2

     python tools/train_lc_bgplacenet_stage1.py \
        --config configs/lc_bgplacenet_stage1.yaml \
        --resume outputs/lc_bgplacenet_stage1/last.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

from src.training.lc_bgplacenet_stage1 import load_config, train_stage1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LC-BGPlaceNet Stage 1.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 1 YAML config path.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a saved checkpoint.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps for smoke tests.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples for smoke tests.")
    return parser.parse_args()


def main() -> None:
    """Load config and start Stage 1 training."""
    args = parse_args()
    cfg = load_config(args.config)
    try:
        train_stage1(
            cfg,
            resume_checkpoint=args.resume,
            max_steps=args.max_steps,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
