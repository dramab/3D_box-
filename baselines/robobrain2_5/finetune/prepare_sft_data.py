#!/usr/bin/env python
"""Build RoboBrain2.5 bottom-center ``(x, y, d)`` SFT annotations.

Usage:
    conda activate spatial
    python baselines/robobrain2_5/finetune/prepare_sft_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.robobrain2_5.run_zero_shot_test import (
    ENRICHED_BOTTOM_CENTER_XYD_PROMPT_VARIANT,
    bottom_center_xyd_prompt_from_item,
    load_ground_truth_placement,
    load_sources,
)
from baselines.robobrain2_5.zero_shot_visualization import Camera, project_world
from src.datasets.canonical import load_sample_record


SCHEMA_VERSION = "robobrain2_5_bottom_center_xyd_sft/v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/lc_bgplacenet_stage2_enriched.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/robobrain2_5_sft_bottom_center_xyd_lora_enriched/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare enriched bottom-center (x, y, d) SFT annotations.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Smoke-test limit applied independently to train and valid; omit for full data.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def pixel_to_normalized(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    """Convert a visible image point to RoboBrain's documented 0--1000 coordinates."""
    x, y = np.asarray(point, dtype=np.float64)
    if not np.all(np.isfinite([x, y])) or not (0.0 <= x < width and 0.0 <= y < height):
        raise ValueError(f"Projected point is outside the RGB image: point=({x}, {y}), size=({width}, {height})")
    x_normalized = int(np.clip(np.rint(x / width * 1000.0), 0, 1000))
    y_normalized = int(np.clip(np.rint(y / height * 1000.0), 0, 1000))
    return x_normalized, y_normalized


@lru_cache(maxsize=4096)
def load_camera_and_rgb_path(dataset_dir_text: str, sample_id: str) -> tuple[Camera, Path]:
    """Load only the camera and RGB path needed to construct one SFT target."""
    dataset_dir = Path(dataset_dir_text)
    record = load_sample_record(dataset_dir / "samples" / f"{sample_id}.json")
    camera_record = record["camera"]
    camera = Camera(
        fx=float(camera_record["fx"]),
        fy=float(camera_record["fy"]),
        cx=float(camera_record["cx"]),
        cy=float(camera_record["cy"]),
        width=int(camera_record["img_w"]),
        height=int(camera_record["img_h"]),
        e_c2w=np.asarray(camera_record["E_c2w"], dtype=np.float64),
    )
    return camera, dataset_dir / str(record["rgb_path"])


def build_sft_row(item: dict[str, Any], source: dict[str, Path]) -> dict[str, Any]:
    """Project one GT legal placement center and serialize the exact SFT exchange."""
    ground_truth = load_ground_truth_placement(item, source)
    camera, rgb_path = load_camera_and_rgb_path(str(source["dataset_dir"]), str(item["sample_id"]))
    uv, depth = project_world(np.asarray([ground_truth["bottom_center_world"]], dtype=np.float64), camera)
    if depth[0] <= 0.0:
        raise ValueError(f"GT placement is behind the camera: {item['item_id']}")
    point = pixel_to_normalized(uv[0], camera.width, camera.height)
    prompt = bottom_center_xyd_prompt_from_item(item, source)
    depth_cm = float(depth[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": str(item["item_id"]),
        "source_name": str(item["source_name"]),
        "sample_id": str(item["sample_id"]),
        "object_id": str(item["object_id"]),
        "image_path": str(rgb_path.resolve()),
        "instruction": str(item["instruction"]),
        "model_prompt": prompt,
        "answer": f"[({point[0]}, {point[1]}, {depth_cm:.2f})]",
        "normalized_point": list(point),
        "depth_cm": depth_cm,
    }


def load_split_items(split_dir: Path, split: str) -> list[dict[str, Any]]:
    split_path = split_dir / f"{split}.json"
    with split_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("split") != split:
        raise ValueError(f"Expected {split!r} split in {split_path}, got {payload.get('split')!r}")
    return list(payload["items"])


def write_split(
    split: str,
    items: list[dict[str, Any]],
    sources: dict[str, dict[str, Path]],
    output_path: Path,
) -> int:
    """Write one complete JSONL split through an atomic temporary file."""
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(items, start=1):
            row = build_sft_row(item, sources[str(item["source_name"])])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 5000 == 0:
                print(f"[{split}] prepared {index}/{len(items)}", flush=True)
    temporary_path.replace(output_path)
    return len(items)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    split_dir = resolve_path(Path(config["data"]["split_dir"]))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources(config_path)
    counts: dict[str, int] = {}
    for split in ("train", "valid"):
        items = load_split_items(split_dir, split)
        if args.max_samples_per_split is not None:
            items = items[: int(args.max_samples_per_split)]
        counts[split] = write_split(split, items, sources, output_dir / f"{split}.jsonl")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": str(config_path),
        "split_dir": str(split_dir),
        "coordinate_convention": "RoboBrain normalized image coordinates in [0, 1000]",
        "target_definition": (
            "GT legal placement bottom-center projected to normalized canonical RGB x/y plus camera depth in cm"
        ),
        "prompt_variant": ENRICHED_BOTTOM_CENTER_XYD_PROMPT_VARIANT,
        "counts": counts,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Wrote SFT annotations to {output_dir}: {counts}")


if __name__ == "__main__":
    main()
