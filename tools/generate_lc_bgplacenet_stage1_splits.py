#!/usr/bin/env python
"""
Generate fixed train/valid/test split files for LC-BGPlaceNet Stage 1.

使用示例:
    conda run -n spatial python tools/generate_lc_bgplacenet_stage1_splits.py \
        --config configs/lc_bgplacenet_stage1.yaml

    conda run -n spatial python tools/generate_lc_bgplacenet_stage1_splits.py \
        --config configs/lc_bgplacenet_stage1.yaml \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.lc_bgplacenet_stage1 import (
    STAGE1_SPLIT_NAMES,
    build_sources_from_config,
    build_stage1_index,
    load_config,
    write_stage1_splits,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate fixed LC-BGPlaceNet Stage 1 splits.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 1 YAML config path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Split output directory. Defaults to data.split_dir from the config.",
    )
    parser.add_argument("--valid-fraction", type=float, default=None, help="Validation group fraction.")
    parser.add_argument("--test-fraction", type=float, default=None, help="Test group fraction.")
    parser.add_argument("--seed", type=int, default=None, help="Split seed. Defaults to data.split_seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    return parser.parse_args()


def main() -> None:
    """Generate split files from the current Stage 1 index."""
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    output_dir = args.output_dir or Path(data_cfg.get("split_dir", "data/splits/lc_bgplacenet_stage1"))
    valid_fraction = args.valid_fraction
    if valid_fraction is None:
        valid_fraction = float(data_cfg.get("valid_fraction", data_cfg.get("val_fraction", 0.1)))
    test_fraction = args.test_fraction
    if test_fraction is None:
        test_fraction = float(data_cfg.get("test_fraction", 0.1))
    seed = int(args.seed if args.seed is not None else data_cfg.get("split_seed", 0))

    existing = [output_dir / f"{split}.json" for split in STAGE1_SPLIT_NAMES]
    existing.append(output_dir / "manifest.json")
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing[:3])
        raise SystemExit(f"ERROR: Split files already exist: {joined}; pass --overwrite to replace them")

    sources = build_sources_from_config(cfg)
    items = build_stage1_index(sources)
    try:
        manifest = write_stage1_splits(
            items=items,
            output_dir=output_dir,
            valid_fraction=valid_fraction,
            test_fraction=test_fraction,
            seed=seed,
            overwrite=bool(args.overwrite),
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
