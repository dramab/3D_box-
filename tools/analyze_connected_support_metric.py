#!/usr/bin/env python
"""Compare connected-area support metrics on the fixed Stage 2 benchmark.

使用示例:
    python tools/analyze_connected_support_metric.py \
        --config configs/lc_bgplacenet_stage2_enriched.yaml \
        --predictions outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_stage2_test/predictions.json \
        --output-dir outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/support_connected_area_analysis_test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label

from src.annotation.free_bbox.io_utils import load_ply
from src.training.lc_bgplacenet_stage2 import load_config
from tools.analyze_support_coverage_metric import footprint_voxel_keys
from tools.benchmark_lc_bgplacenet_stage2 import (
    _build_item_lookup,
    load_predictions,
    resolve_config_paths,
)


DEFAULT_RUN_DIR = PROJECT_ROOT / (
    "outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_"
    "full_gt_guass_8_enriched"
)
VARIANTS = ("fill_only", "closing_fill")
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class ConnectedRegion:
    """One Z-band projection and its filled 8-connected component labels."""

    origin_xy: np.ndarray
    raw_mask: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze connected-area support coverage.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/lc_bgplacenet_stage2_enriched.yaml",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_RUN_DIR / "inference_stage2_test/predictions.json",
    )
    parser.add_argument("--split", choices=("train", "valid", "val", "test"), default="test")
    parser.add_argument("--downward-cm", type=float, default=3.0)
    parser.add_argument("--upper-cm", type=float, default=1.0)
    parser.add_argument("--floating-offset-cm", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "support_connected_area_analysis_test",
    )
    args = parser.parse_args()
    if args.downward_cm <= 0.0 or args.upper_cm < 0.0:
        parser.error("The downward range must be positive and upper tolerance non-negative")
    return args


def quantize_occupied_points(points_world: np.ndarray, voxel_size_cm: float) -> np.ndarray:
    """Return unique global XYZ voxel keys for the complete scene point cloud."""
    return np.unique(
        np.floor(np.asarray(points_world, dtype=np.float64) / float(voxel_size_cm)).astype(
            np.int64
        ),
        axis=0,
    )


def z_key_bounds(
    bottom_z: float,
    downward_cm: float,
    upper_cm: float,
    voxel_size_cm: float,
) -> tuple[int, int]:
    """Convert a world-space center-inclusive Z band to voxel-key bounds."""
    voxel_size = float(voxel_size_cm)
    min_key = math.ceil((float(bottom_z) - float(downward_cm)) / voxel_size - 0.5)
    max_key = math.floor((float(bottom_z) + float(upper_cm)) / voxel_size - 0.5)
    return int(min_key), int(max_key)


def build_connected_region(
    occupied_keys: np.ndarray,
    z_min: int,
    z_max: int,
    apply_closing: bool,
) -> ConnectedRegion:
    """Project one Z band, optionally close 1-cell gaps, fill holes and label it."""
    keys = np.asarray(occupied_keys, dtype=np.int64)
    band = keys[(keys[:, 2] >= int(z_min)) & (keys[:, 2] <= int(z_max))]
    if len(band) == 0:
        return ConnectedRegion(
            origin_xy=np.zeros(2, dtype=np.int64),
            raw_mask=np.zeros((1, 1), dtype=bool),
            labels=np.zeros((1, 1), dtype=np.int32),
        )

    xy = np.unique(band[:, :2], axis=0)
    origin = xy.min(axis=0) - 1
    shape = xy.max(axis=0) - origin + 2
    raw = np.zeros(tuple(shape.tolist()), dtype=bool)
    local = xy - origin
    raw[local[:, 0], local[:, 1]] = True
    connected = binary_closing(raw, structure=STRUCTURE_8) if apply_closing else raw
    filled = binary_fill_holes(connected)
    labels, _ = label(filled, structure=STRUCTURE_8)
    return ConnectedRegion(origin_xy=origin, raw_mask=raw, labels=labels.astype(np.int32))


def component_coverage(
    place_box: np.ndarray,
    occupied_keys: np.ndarray,
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
    apply_closing: bool,
    cache: dict[tuple[int, int, bool], ConnectedRegion] | None = None,
) -> tuple[float, ConnectedRegion, np.ndarray, int]:
    """Return the fraction of footprint cells in the center's single filled component."""
    box = np.asarray(place_box, dtype=np.float64)
    bottom_z = float(box[2] - box[5] * 0.5)
    z_min, z_max = z_key_bounds(bottom_z, downward_cm, upper_cm, voxel_size_cm)
    key = (z_min, z_max, bool(apply_closing))
    region_cache = cache if cache is not None else {}
    if key not in region_cache:
        region_cache[key] = build_connected_region(
            occupied_keys, z_min, z_max, apply_closing=apply_closing
        )
    region = region_cache[key]
    footprint = footprint_voxel_keys(box, voxel_size_cm)
    center_key = np.floor(box[:2] / float(voxel_size_cm)).astype(np.int64)
    center_local = center_key - region.origin_xy
    center_in = bool(
        np.all(center_local >= 0) and np.all(center_local < np.asarray(region.labels.shape))
    )
    component_id = int(region.labels[tuple(center_local)]) if center_in else 0
    if component_id == 0 or len(footprint) == 0:
        return 0.0, region, footprint, component_id

    local = footprint - region.origin_xy
    in_bounds = np.all(local >= 0, axis=1) & np.all(
        local < np.asarray(region.labels.shape), axis=1
    )
    supported = np.zeros(len(footprint), dtype=bool)
    valid_local = local[in_bounds]
    supported[in_bounds] = region.labels[valid_local[:, 0], valid_local[:, 1]] == component_id
    return float(supported.mean()), region, footprint, component_id


def raw_band_coverage(
    place_box: np.ndarray,
    occupied_keys: np.ndarray,
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
) -> float:
    """Return direct footprint coverage before connected-area completion."""
    box = np.asarray(place_box, dtype=np.float64)
    bottom_z = float(box[2] - box[5] * 0.5)
    z_min, z_max = z_key_bounds(bottom_z, downward_cm, upper_cm, voxel_size_cm)
    band = occupied_keys[
        (occupied_keys[:, 2] >= z_min) & (occupied_keys[:, 2] <= z_max)
    ]
    footprint = footprint_voxel_keys(box, voxel_size_cm)
    if len(footprint) == 0:
        return 0.0
    band_xy = {tuple(key) for key in np.unique(band[:, :2], axis=0).tolist()}
    return float(np.mean([tuple(key) in band_xy for key in footprint.tolist()]))


def _box_results(
    box: np.ndarray,
    occupied_keys: np.ndarray,
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
    cache: dict[tuple[int, int, bool], ConnectedRegion],
) -> dict[str, float]:
    results = {
        "raw_coverage": raw_band_coverage(
            box, occupied_keys, voxel_size_cm, downward_cm, upper_cm
        )
    }
    for variant in VARIANTS:
        results[variant] = component_coverage(
            box,
            occupied_keys,
            voxel_size_cm,
            downward_cm,
            upper_cm,
            apply_closing=variant == "closing_fill",
            cache=cache,
        )[0]
    return results


def analyze(
    cfg: dict[str, Any],
    predictions: list[dict[str, Any]],
    split: str,
    downward_cm: float,
    upper_cm: float,
    floating_offset_cm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate both connected-area variants on GT, floating GT and Top-K predictions."""
    item_by_id = _build_item_lookup(cfg, split)
    by_scene: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        item = item_by_id.get(str(row["item_id"]))
        if item is None:
            raise ValueError(f"Prediction is not in split={split}: {row['item_id']}")
        by_scene[item.voxel_point_cloud_path].append(row)

    voxel_size = float(cfg["data"]["voxel_size_cm"])
    output_rows = []
    for scene_index, (point_path, scene_rows) in enumerate(by_scene.items(), start=1):
        points, _ = load_ply(point_path)
        occupied_keys = quantize_occupied_points(points, voxel_size)
        cache: dict[tuple[int, int, bool], ConnectedRegion] = {}
        for row in scene_rows:
            item = item_by_id[str(row["item_id"])]
            gt_box = np.asarray(row["place_box_gt"], dtype=np.float64)
            floating_box = gt_box.copy()
            floating_box[2] += float(floating_offset_cm)
            placements = row.get("placements", [])[:5]
            boxes = [np.asarray(entry["box"], dtype=np.float64) for entry in placements]
            if not boxes:
                boxes = [np.asarray(row["place_box"], dtype=np.float64)]
            gt = _box_results(
                gt_box, occupied_keys, voxel_size, downward_cm, upper_cm, cache
            )
            floating = _box_results(
                floating_box, occupied_keys, voxel_size, downward_cm, upper_cm, cache
            )
            candidates = [
                _box_results(box, occupied_keys, voxel_size, downward_cm, upper_cm, cache)
                for box in boxes
            ]
            output_rows.append(
                {
                    "item_id": str(row["item_id"]),
                    "source_name": item.source_name,
                    "voxel_point_cloud_path": os.fspath(point_path),
                    "gt_box": gt_box.tolist(),
                    "top1_box": boxes[0].tolist(),
                    "gt": gt,
                    "floating_gt": floating,
                    "top1": candidates[0],
                    "top5": {
                        variant: bool(any(candidate[variant] >= 1.0 - 1e-12 for candidate in candidates))
                        for variant in VARIANTS
                    },
                }
            )
        if scene_index % 100 == 0 or scene_index == len(by_scene):
            print(f"Processed {scene_index}/{len(by_scene)} point clouds")
    return output_rows, summarize(output_rows, voxel_size, downward_cm, upper_cm)


def summarize(
    rows: list[dict[str, Any]],
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
) -> dict[str, Any]:
    """Aggregate strict full-component coverage rates overall and by source."""
    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "voxel_size_cm": float(voxel_size_cm),
        "downward_cm": float(downward_cm),
        "upper_cm": float(upper_cm),
        "variants": {},
    }
    for variant in VARIANTS:
        gt = np.asarray([row["gt"][variant] for row in rows]) >= 1.0 - 1e-12
        floating = (
            np.asarray([row["floating_gt"][variant] for row in rows]) >= 1.0 - 1e-12
        )
        top1 = np.asarray([row["top1"][variant] for row in rows]) >= 1.0 - 1e-12
        top5 = np.asarray([row["top5"][variant] for row in rows], dtype=bool)
        by_source = {}
        for source in sorted({str(row["source_name"]) for row in rows}):
            mask = np.asarray([row["source_name"] == source for row in rows])
            by_source[source] = {
                "sample_count": int(mask.sum()),
                "gt_pass_rate": float(gt[mask].mean()),
                "top1_pass_rate": float(top1[mask].mean()),
            }
        summary["variants"][variant] = {
            "gt_pass_rate": float(gt.mean()),
            "floating_gt_pass_rate": float(floating.mean()),
            "top1_pass_rate": float(top1.mean()),
            "top5_pass_rate": float(top5.mean()),
            "by_source": by_source,
        }
    summary["raw_gt_full_coverage_rate"] = float(
        np.mean([row["gt"]["raw_coverage"] >= 1.0 - 1e-12 for row in rows])
    )
    return summary


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )


def plot_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """Plot strict pass rates for the two connected-area variants."""
    _plot_style()
    categories = ("GT", "Top-1", "Top-5", "GT +5 cm")
    keys = ("gt_pass_rate", "top1_pass_rate", "top5_pass_rate", "floating_gt_pass_rate")
    x = np.arange(len(categories))
    width = 0.34
    fig, axis = plt.subplots(figsize=(4.4, 2.6), constrained_layout=True)
    styles = (
        ("Fill holes", "fill_only", "#0072B2", "o"),
        ("3×3 closing + fill holes", "closing_fill", "#D55E00", "s"),
    )
    for index, (label_name, variant, color, marker) in enumerate(styles):
        values = [summary["variants"][variant][key] * 100.0 for key in keys]
        positions = x + (index - 0.5) * width
        axis.bar(positions, values, width, color=color, alpha=0.85, label=label_name)
        axis.scatter(positions, values, color="black", marker=marker, s=12, zorder=3)
        for position, value in zip(positions, values):
            axis.text(position, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5)
    axis.set_xticks(x, categories)
    axis.set_ylabel("Strict connected-area pass rate (%)")
    axis.set_ylim(0.0, 105.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, loc="lower left")
    axis.set_title(
        f"Z band: down {summary['downward_cm']:g} cm, upper {summary['upper_cm']:g} cm",
        loc="left",
        fontsize=8,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"connected_support_comparison.{suffix}", dpi=220)
    plt.close(fig)


def _selected_component_mask(
    box: np.ndarray,
    region: ConnectedRegion,
    voxel_size_cm: float,
) -> tuple[np.ndarray, int]:
    center_key = np.floor(np.asarray(box[:2]) / float(voxel_size_cm)).astype(np.int64)
    local = center_key - region.origin_xy
    if np.any(local < 0) or np.any(local >= np.asarray(region.labels.shape)):
        return np.zeros_like(region.labels, dtype=bool), 0
    component_id = int(region.labels[tuple(local)])
    return region.labels == component_id, component_id


def plot_case(
    label_name: str,
    row: dict[str, Any],
    occupied_keys: np.ndarray,
    voxel_size_cm: float,
    downward_cm: float,
    upper_cm: float,
    output_path: Path,
) -> None:
    """Show raw projection, hole filling and closing+filling for one GT box."""
    _plot_style()
    box = np.asarray(row["gt_box"], dtype=np.float64)
    bottom_z = float(box[2] - box[5] * 0.5)
    z_min, z_max = z_key_bounds(bottom_z, downward_cm, upper_cm, voxel_size_cm)
    regions = [
        build_connected_region(occupied_keys, z_min, z_max, apply_closing=False),
        build_connected_region(occupied_keys, z_min, z_max, apply_closing=True),
    ]
    footprint = footprint_voxel_keys(box, voxel_size_cm)
    pad = 4
    lo = footprint.min(axis=0) - pad
    hi = footprint.max(axis=0) + pad
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6), constrained_layout=True)
    titles = ("(a) Raw Z-band projection", "(b) Fill holes", "(c) Closing + fill holes")

    raw_region = regions[0]
    raw_local = footprint - raw_region.origin_xy
    raw_in = np.all(raw_local >= 0, axis=1) & np.all(
        raw_local < np.asarray(raw_region.raw_mask.shape), axis=1
    )
    raw_supported = np.zeros(len(footprint), dtype=bool)
    valid = raw_local[raw_in]
    raw_supported[raw_in] = raw_region.raw_mask[valid[:, 0], valid[:, 1]]

    for axis_index, axis in enumerate(axes):
        region = raw_region if axis_index == 0 else regions[axis_index - 1]
        if axis_index == 0:
            support_mask = region.raw_mask
            supported = raw_supported
        else:
            support_mask, component_id = _selected_component_mask(box, region, voxel_size_cm)
            local = footprint - region.origin_xy
            in_bounds = np.all(local >= 0, axis=1) & np.all(
                local < np.asarray(region.labels.shape), axis=1
            )
            supported = np.zeros(len(footprint), dtype=bool)
            valid = local[in_bounds]
            supported[in_bounds] = (
                region.labels[valid[:, 0], valid[:, 1]] == component_id
            ) & (component_id != 0)

        support_xy = np.argwhere(support_mask) + region.origin_xy
        visible = np.all(support_xy >= lo, axis=1) & np.all(support_xy <= hi, axis=1)
        support_xy = support_xy[visible]
        support_world = (support_xy.astype(np.float64) + 0.5) * voxel_size_cm
        footprint_world = (footprint.astype(np.float64) + 0.5) * voxel_size_cm
        axis.scatter(support_world[:, 0], support_world[:, 1], s=10, marker="s", c="#BDBDBD")
        axis.scatter(
            footprint_world[supported, 0], footprint_world[supported, 1],
            s=16, marker="s", c="#009E73", label="covered",
        )
        axis.scatter(
            footprint_world[~supported, 0], footprint_world[~supported, 1],
            s=16, marker="s", c="#D55E00", label="uncovered",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim((lo[0]) * voxel_size_cm, (hi[0] + 1) * voxel_size_cm)
        axis.set_ylim((lo[1]) * voxel_size_cm, (hi[1] + 1) * voxel_size_cm)
        axis.set_xlabel("world X (cm)")
        axis.set_ylabel("world Y (cm)")
        axis.set_title(titles[axis_index], loc="left", fontsize=8)
        axis.legend(frameon=False, loc="best")
    fig.suptitle(f"{label_name}: {row['item_id']}", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)


def select_cases(rows: list[dict[str, Any]], case_count: int) -> list[tuple[str, dict[str, Any]]]:
    """Select examples recovered by filling, recovered by closing, and still failing."""
    predicates = (
        (
            "hole_filled",
            lambda row: row["gt"]["raw_coverage"] < 1.0 - 1e-12
            and row["gt"]["fill_only"] >= 1.0 - 1e-12,
        ),
        (
            "closing_recovered",
            lambda row: row["gt"]["fill_only"] < 1.0 - 1e-12
            and row["gt"]["closing_fill"] >= 1.0 - 1e-12,
        ),
        ("still_fail", lambda row: row["gt"]["closing_fill"] < 1.0 - 1e-12),
    )
    selected = []
    used = set()
    for label_name, predicate in predicates:
        for index, row in enumerate(rows):
            if index not in used and predicate(row):
                selected.append((label_name, row))
                used.add(index)
                break
    for index, row in enumerate(rows):
        if len(selected) >= int(case_count):
            break
        if index not in used:
            selected.append(("additional", row))
            used.add(index)
    return selected[: int(case_count)]


def save_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    case_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    plot_summary(summary, output_dir)

    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for path in list(cases_dir.glob("case_*.png")) + list(cases_dir.glob("case_*.pdf")):
        path.unlink()
    for case_index, (label_name, row) in enumerate(select_cases(rows, case_count), start=1):
        points, _ = load_ply(row["voxel_point_cloud_path"])
        occupied_keys = quantize_occupied_points(points, float(summary["voxel_size_cm"]))
        plot_case(
            label_name,
            row,
            occupied_keys,
            float(summary["voxel_size_cm"]),
            float(summary["downward_cm"]),
            float(summary["upper_cm"]),
            cases_dir / f"case_{case_index:02d}_{label_name}",
        )


def main() -> None:
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    predictions = load_predictions(args.predictions)
    if args.max_samples is not None:
        predictions = predictions[: int(args.max_samples)]
    rows, summary = analyze(
        cfg,
        predictions,
        args.split,
        float(args.downward_cm),
        float(args.upper_cm),
        float(args.floating_offset_cm),
    )
    save_outputs(rows, summary, args.output_dir, int(args.case_count))
    print(json.dumps(summary["variants"], indent=2, ensure_ascii=False))
    print(f"Wrote connected support analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
