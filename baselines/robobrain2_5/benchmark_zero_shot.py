#!/usr/bin/env python
"""Evaluate RoboBrain2.5 point predictions with the official placement metrics.

Usage:
    conda activate spatial
    python baselines/robobrain2_5/benchmark_zero_shot.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

from baselines.robobrain2_5.zero_shot_visualization import load_jsonl
from src.models.lc_bgplacenet.stage2 import NUM_YAW_BINS
from src.training.lc_bgplacenet_stage2 import (
    Stage2IndexItem,
    build_sources_from_config,
    build_stage2_index,
    load_config,
    normalize_stage1_split,
    select_stage2_split_items,
)
from tools.benchmark_lc_bgplacenet_stage2 import (
    resolve_config_paths,
    run_benchmark,
    save_outputs,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/lc_bgplacenet_stage2.yaml"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs/robobrain2_5_front_behind_swapped/predictions.jsonl"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/robobrain2_5_front_behind_swapped/benchmark_official_top1_oracle_geometry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RoboBrain2.5 Top-1 point-derived boxes with the official placement benchmark."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--model-label", default="RoboBrain2.5-8B-NV")
    parser.add_argument("--training-protocol", choices=("zero_shot", "lora_sft"), default="zero_shot")
    parser.add_argument("--prediction-format", choices=("xy", "xyd"), default="xy")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def yaw_radians_to_bin(yaw: float) -> int:
    """Map an observed source yaw to the nearest official 180-degree yaw bin."""
    bin_width = math.pi / NUM_YAW_BINS
    return int(math.floor((float(yaw) % math.pi) / bin_width + 0.5)) % NUM_YAW_BINS


def convert_prediction_row(row: dict[str, Any], item: Stage2IndexItem) -> dict[str, Any]:
    """Convert one point-baseline row to the official benchmark prediction schema."""
    raw_box = row.get("render_box_world")
    placements: list[dict[str, Any]] = []
    if raw_box is not None:
        box = np.asarray(raw_box, dtype=np.float64)
        if box.shape != (7,) or not np.all(np.isfinite(box)):
            raise ValueError(f"Invalid render_box_world for {item.item_id}: {raw_box!r}")
        placements.append(
            {
                "box": box.tolist(),
                "score": 1.0,
                "yaw_bin": yaw_radians_to_bin(float(box[6])),
            }
        )
    return {
        "item_id": item.item_id,
        # RoboBrain does not predict a source box. The oracle value only satisfies
        # the common evaluator schema and is removed from reported source metrics.
        "source_box": item.source_box_gt.tolist(),
        "place_box_gt": item.place_box_gt.tolist(),
        "placements": placements,
        "prediction_status": str(row.get("status", "unknown")),
    }


def build_item_lookup(cfg: dict[str, Any], split: str) -> dict[str, Stage2IndexItem]:
    sources = build_sources_from_config(cfg)
    items = build_stage2_index(sources)
    selected = select_stage2_split_items(
        items,
        cfg["data"]["split_dir"],
        normalize_stage1_split(split),
    )
    return {item.item_id: item for item in selected}


def validate_and_convert(
    rows: list[dict[str, Any]],
    item_by_id: dict[str, Stage2IndexItem],
) -> list[dict[str, Any]]:
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id"))
        if item_id in row_by_id:
            raise ValueError(f"Duplicate RoboBrain prediction: {item_id}")
        row_by_id[item_id] = row
    unknown = sorted(set(row_by_id) - set(item_by_id))
    missing = sorted(set(item_by_id) - set(row_by_id))
    if unknown or missing:
        raise ValueError(
            f"Prediction/split mismatch: unknown={len(unknown)}, missing={len(missing)}; "
            f"unknown_preview={unknown[:3]}, missing_preview={missing[:3]}"
        )
    return [convert_prediction_row(row_by_id[item_id], item) for item_id, item in item_by_id.items()]


def build_direction_metadata(item_by_id: dict[str, Stage2IndexItem]) -> dict[str, dict[str, Any]]:
    """Use the structured benchmark relation fields as the single metadata source."""
    return {
        item_id: {
            "target_relation": item.target_relation,
            "reference_object_id": item.reference_object_id,
            "reference_name": "",
        }
        for item_id, item in item_by_id.items()
    }


def mark_non_applicable_source_metrics(
    per_sample_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Prevent the evaluator's oracle schema value from being reported as a model result."""
    for row in per_sample_rows:
        row["source_iou"] = None
        row["source_iou_correct"] = None
        row["source_metric_status"] = "not_applicable_model_does_not_predict_source_box"
    for aggregate in [summary["overall"], *summary["by_source"].values()]:
        aggregate["source_iou"] = None
        aggregate["source_iou_accuracy"] = None


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    predictions_path = resolve_path(args.predictions)
    output_dir = resolve_path(args.output_dir)
    cfg = resolve_config_paths(load_config(config_path))
    item_by_id = build_item_lookup(cfg, args.split)
    raw_rows = load_jsonl(predictions_path)
    predictions = validate_and_convert(raw_rows, item_by_id)

    validation_cfg = cfg.get("validation", {})
    per_sample_rows, summary = run_benchmark(
        cfg,
        predictions,
        direction_metadata=build_direction_metadata(item_by_id),
        split=args.split,
        size_iou_threshold=float(validation_cfg.get("size_iou_threshold", 0.8)),
        source_iou_threshold=float(validation_cfg.get("source_iou_threshold", 0.5)),
        center_match_threshold_cm=float(validation_cfg.get("center_match_threshold_cm", 2.0)),
        support_downward_cm=float(validation_cfg.get("support_downward_cm", 3.0)),
        support_upper_cm=float(validation_cfg.get("support_upper_cm", 1.0)),
    )
    mark_non_applicable_source_metrics(per_sample_rows, summary)

    status_counts = Counter(str(row.get("status", "unknown")) for row in raw_rows)
    emitted_count = sum(bool(row["placements"]) for row in predictions)
    if any(str(row.get("prediction_format", args.prediction_format)) != args.prediction_format for row in raw_rows):
        raise ValueError(f"Prediction rows do not consistently use {args.prediction_format!r} format")
    uses_model_depth = args.prediction_format == "xyd"
    summary["protocol"] = {
        "model": str(args.model_label),
        "training_protocol": str(args.training_protocol),
        "benchmark_config": str(config_path),
        "split_dir": str(cfg["data"]["split_dir"]),
        "prompt_variant": (
            "enriched_label_bottom_center_xyd"
            if uses_model_depth
            else "official_vacant_space_front_behind_swapped"
        ),
        "model_input": (
            "RGB plus verbatim enriched label and bottom-center output-format instruction"
            if uses_model_depth
            else "RGB and transformed language instruction only"
        ),
        "model_output": (
            "one normalized image point plus absolute camera depth in centimeters"
            if uses_model_depth
            else "one normalized 2D point"
        ),
        "candidate_construction": (
            "model-predicted (x,y,d) backprojection plus oracle source dimensions and observed source yaw"
            if uses_model_depth
            else "RGB-D backprojection plus oracle source dimensions and observed source yaw"
        ),
        "depth_source": "model_absolute_camera_depth_cm" if uses_model_depth else "rgbd_local_median_cm",
        "top_k": 1,
        "source_iou_status": "not_applicable_model_does_not_predict_source_box",
        "placement_size_iou_status": "oracle_source_dimensions",
        "yaw_bin_mapping": f"nearest of {NUM_YAW_BINS} bins over [0, pi)",
        "invalid_point_predictions_count_as_failures": True,
        "prediction_status_counts": dict(sorted(status_counts.items())),
        "emitted_candidate_count": int(emitted_count),
        "missing_candidate_count": int(len(predictions) - emitted_count),
        "predictions_path": str(predictions_path),
    }
    save_outputs(output_dir, per_sample_rows, summary)

    overall = summary["overall"]
    print(f"Wrote RoboBrain benchmark outputs to {output_dir}")
    print(f"Samples: {overall['sample_count']}")
    print(f"Emitted candidates: {emitted_count}/{len(predictions)}")
    print(f"Placement size accuracy (oracle size): {overall['placement_size_accuracy']:.4f}")
    print(f"Language relation correct: {overall['language_relation_correct_rate']:.4f}")
    print(f"Supported and stable: {overall['supported_and_stable_rate']:.4f}")
    print(f"Collision free: {overall['collision_free_rate']:.4f}")
    print(f"Center match rate: {overall['center_match_rate']:.4f}")
    print(f"Yaw valid given center match: {overall['yaw_valid_given_center_match']:.4f}")
    print(f"Placement Success@1: {overall['placement_success_at_1']:.4f}")
    print(f"Placement Success@5: {overall['placement_success_at_5']:.4f} (Top-1 baseline)")


if __name__ == "__main__":
    main()
