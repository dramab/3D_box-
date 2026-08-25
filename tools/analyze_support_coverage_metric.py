#!/usr/bin/env python
"""Analyze voxel support coverage below predicted placement boxes.

The script is intentionally separate from the production benchmark. It sweeps
the downward occupancy band before the metric definition is finalized.

使用示例:
    python tools/analyze_support_coverage_metric.py \
        --config configs/lc_bgplacenet_stage2_enriched.yaml \
        --predictions outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/inference_stage2_test/predictions.json \
        --direction-metadata outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/direction_metadata_test.json \
        --output-dir outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/support_coverage_analysis_test
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

from src.annotation.free_bbox.io_utils import load_ply
from src.placement_metrics import footprint_voxel_keys
from src.training.lc_bgplacenet_stage2 import (
    Stage2IndexItem,
    compute_collision_metrics,
    compute_size_metrics,
)
from tools.benchmark_lc_bgplacenet_stage2 import (
    _build_item_lookup,
    _get_collision_context,
    compute_direction_metrics,
    load_direction_metadata,
    load_direction_scene_context,
    load_predictions,
    resolve_config_paths,
)
from src.training.lc_bgplacenet_stage2 import load_config


DEFAULT_RUN_DIR = PROJECT_ROOT / (
    "outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_"
    "full_gt_guass_8_enriched"
)
DEFAULT_DEPTHS_CM = (1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_UPPER_TOLERANCES_CM = (0.0, 1.0)
PASS_EPS = 1e-12
COVERAGE_SENSITIVITY_THRESHOLDS = (0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 1.0)


@dataclass(frozen=True)
class OccupancyColumns:
    """Sorted occupied Z centers keyed by canonical XY voxel index."""

    voxel_size_cm: float
    z_centers_by_xy: dict[tuple[int, int], np.ndarray]
    points_world: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep box-footprint support coverage thickness.")
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
    parser.add_argument(
        "--direction-metadata",
        type=Path,
        default=DEFAULT_RUN_DIR / "direction_metadata_test.json",
    )
    parser.add_argument("--split", choices=("train", "valid", "val", "test"), default="test")
    parser.add_argument("--depths-cm", type=float, nargs="+", default=list(DEFAULT_DEPTHS_CM))
    parser.add_argument(
        "--upper-tolerances-cm",
        type=float,
        nargs="+",
        default=list(DEFAULT_UPPER_TOLERANCES_CM),
    )
    parser.add_argument("--coverage-threshold", type=float, default=1.0)
    parser.add_argument("--floating-offset-cm", type=float, default=5.0)
    parser.add_argument("--size-iou-threshold", type=float, default=0.8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "support_coverage_analysis_test",
    )
    args = parser.parse_args()
    if any(value <= 0.0 for value in args.depths_cm):
        parser.error("--depths-cm values must be positive")
    if any(value < 0.0 for value in args.upper_tolerances_cm):
        parser.error("--upper-tolerances-cm values must be non-negative")
    if not 0.0 < args.coverage_threshold <= 1.0:
        parser.error("--coverage-threshold must be in (0, 1]")
    return args


def build_occupancy_columns(points_world: np.ndarray, voxel_size_cm: float) -> OccupancyColumns:
    """Quantize the complete canonical point cloud without removing source voxels."""
    points = np.asarray(points_world, dtype=np.float64)
    voxel_size = float(voxel_size_cm)
    keys = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0)
    columns: dict[tuple[int, int], list[float]] = defaultdict(list)
    for x_key, y_key, z_key in keys:
        columns[(int(x_key), int(y_key))].append((float(z_key) + 0.5) * voxel_size)
    sorted_columns = {
        key: np.sort(np.asarray(values, dtype=np.float64)) for key, values in columns.items()
    }
    return OccupancyColumns(voxel_size, sorted_columns, points.astype(np.float32))


def footprint_support_distances(
    place_box: np.ndarray,
    occupancy: OccupancyColumns,
    upper_tolerance_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each footprint cell's nearest allowed occupied-voxel depth below the box."""
    box = np.asarray(place_box, dtype=np.float64)
    bottom_z = float(box[2] - box[5] * 0.5)
    footprint = footprint_voxel_keys(box, occupancy.voxel_size_cm)
    distances = np.full(len(footprint), np.inf, dtype=np.float64)
    upper_limit = bottom_z + float(upper_tolerance_cm)
    for index, xy_key in enumerate(footprint):
        z_centers = occupancy.z_centers_by_xy.get((int(xy_key[0]), int(xy_key[1])))
        if z_centers is None:
            continue
        z_index = int(np.searchsorted(z_centers, upper_limit, side="right")) - 1
        if z_index >= 0:
            distances[index] = bottom_z - float(z_centers[z_index])
    return footprint, distances


def coverage_curve(
    place_box: np.ndarray,
    occupancy: OccupancyColumns,
    depths_cm: np.ndarray,
    upper_tolerance_cm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute footprint coverage for every downward thickness."""
    footprint, distances = footprint_support_distances(place_box, occupancy, upper_tolerance_cm)
    if len(footprint) == 0:
        return np.zeros(len(depths_cm), dtype=np.float64), footprint, distances
    coverage = (distances[None, :] <= depths_cm[:, None] + PASS_EPS).mean(axis=1)
    return coverage.astype(np.float64), footprint, distances


def _curves_for_box(
    place_box: np.ndarray,
    occupancy: OccupancyColumns,
    depths_cm: np.ndarray,
    upper_tolerances_cm: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [coverage_curve(place_box, occupancy, depths_cm, upper)[0] for upper in upper_tolerances_cm]
    )


def _candidate_base_success(
    box: np.ndarray,
    gt_box: np.ndarray,
    item: Stage2IndexItem,
    direction_metadata: dict[str, Any],
    direction_scene: dict[str, Any],
    collision_context: dict[str, Any],
    size_iou_threshold: float,
) -> bool:
    return bool(
        compute_size_metrics(box, gt_box, size_iou_threshold)["size_correct"]
        and compute_direction_metrics(box, direction_metadata, direction_scene)["direction_hit"]
        and not compute_collision_metrics(box, collision_context)["collision"]
    )


def _serialize_array(values: np.ndarray) -> list:
    return np.asarray(values).tolist()


def analyze_predictions(
    cfg: dict[str, Any],
    predictions: list[dict[str, Any]],
    direction_metadata: dict[str, dict[str, Any]],
    split: str,
    depths_cm: np.ndarray,
    upper_tolerances_cm: np.ndarray,
    coverage_threshold: float,
    floating_offset_cm: float,
    size_iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute GT, floating-control, Top-1 and Top-5 coverage curves."""
    item_by_id = _build_item_lookup(cfg, split)
    rows_by_scene: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        item = item_by_id.get(str(row["item_id"]))
        if item is None:
            raise ValueError(f"Prediction is not in split={split}: {row['item_id']}")
        rows_by_scene[item.voxel_point_cloud_path].append(row)

    per_sample = []
    collision_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for scene_index, (point_path, scene_rows) in enumerate(rows_by_scene.items(), start=1):
        points, _ = load_ply(point_path)
        occupancy = build_occupancy_columns(points, float(cfg["data"]["voxel_size_cm"]))
        first_item = item_by_id[str(scene_rows[0]["item_id"])]
        direction_scene = load_direction_scene_context(first_item)
        collision_context = _get_collision_context(first_item, collision_cache)

        for row in scene_rows:
            item = item_by_id[str(row["item_id"])]
            gt_box = np.asarray(row["place_box_gt"], dtype=np.float64)
            float_box = gt_box.copy()
            float_box[2] += float(floating_offset_cm)
            placements = row.get("placements", [])[:5]
            candidate_boxes = [np.asarray(entry["box"], dtype=np.float64) for entry in placements]
            if not candidate_boxes:
                candidate_boxes = [np.asarray(row["place_box"], dtype=np.float64)]

            gt_curves = _curves_for_box(gt_box, occupancy, depths_cm, upper_tolerances_cm)
            float_curves = _curves_for_box(float_box, occupancy, depths_cm, upper_tolerances_cm)
            candidate_curves = np.stack(
                [
                    _curves_for_box(box, occupancy, depths_cm, upper_tolerances_cm)
                    for box in candidate_boxes
                ]
            )
            base_success = np.asarray(
                [
                    _candidate_base_success(
                        box,
                        gt_box,
                        item,
                        direction_metadata[str(row["item_id"])],
                        direction_scene,
                        collision_context,
                        size_iou_threshold,
                    )
                    for box in candidate_boxes
                ],
                dtype=bool,
            )
            support_pass = candidate_curves >= float(coverage_threshold) - PASS_EPS
            gated_success = support_pass & base_success[:, None, None]
            per_sample.append(
                {
                    "item_id": str(row["item_id"]),
                    "source_name": item.source_name,
                    "sample_id": item.sample_id,
                    "object_id": item.object_id,
                    "cluster_id": int(item.cluster_id),
                    "voxel_point_cloud_path": os.fspath(point_path),
                    "gt_box": _serialize_array(gt_box),
                    "top1_box": _serialize_array(candidate_boxes[0]),
                    "gt_coverage": _serialize_array(gt_curves),
                    "floating_gt_coverage": _serialize_array(float_curves),
                    "top1_coverage": _serialize_array(candidate_curves[0]),
                    "top5_support_pass": _serialize_array(support_pass.any(axis=0)),
                    "top1_base_success": bool(base_success[0]),
                    "top5_base_success": bool(base_success.any()),
                    "top1_gated_success": _serialize_array(gated_success[0]),
                    "top5_gated_success": _serialize_array(gated_success.any(axis=0)),
                }
            )
        if scene_index % 100 == 0 or scene_index == len(rows_by_scene):
            print(f"Processed {scene_index}/{len(rows_by_scene)} point clouds")

    summary = summarize(per_sample, depths_cm, upper_tolerances_cm, coverage_threshold)
    summary["voxel_size_cm"] = float(cfg["data"]["voxel_size_cm"])
    return per_sample, summary


def summarize(
    rows: list[dict[str, Any]],
    depths_cm: np.ndarray,
    upper_tolerances_cm: np.ndarray,
    coverage_threshold: float,
) -> dict[str, Any]:
    """Aggregate item-level curves and select the best calibration gap."""
    gt = np.asarray([row["gt_coverage"] for row in rows], dtype=np.float64)
    floating = np.asarray([row["floating_gt_coverage"] for row in rows], dtype=np.float64)
    top1 = np.asarray([row["top1_coverage"] for row in rows], dtype=np.float64)
    top5 = np.asarray([row["top5_support_pass"] for row in rows], dtype=bool)
    gated1 = np.asarray([row["top1_gated_success"] for row in rows], dtype=bool)
    gated5 = np.asarray([row["top5_gated_success"] for row in rows], dtype=bool)
    threshold = float(coverage_threshold) - PASS_EPS
    rates = {
        "gt_full_coverage_rate": (gt >= threshold).mean(axis=0),
        "floating_gt_full_coverage_rate": (floating >= threshold).mean(axis=0),
        "top1_full_coverage_rate": (top1 >= threshold).mean(axis=0),
        "top5_full_coverage_rate": top5.mean(axis=0),
        "top1_support_gated_placement_success_rate": gated1.mean(axis=0),
        "top5_support_gated_placement_success_rate": gated5.mean(axis=0),
    }
    gap = rates["gt_full_coverage_rate"] - rates["floating_gt_full_coverage_rate"]
    max_gap = float(gap.max())
    # Prefer the smallest thickness within 0.1 percentage points of the best gap.
    eligible = (gap >= max_gap - 0.001) & (
        rates["floating_gt_full_coverage_rate"] <= 0.01
    )
    eligible_indices = np.argwhere(eligible)
    if len(eligible_indices):
        best_upper, best_depth = min(
            eligible_indices.tolist(),
            key=lambda index: (depths_cm[index[1]], upper_tolerances_cm[index[0]]),
        )
    else:
        best_upper, best_depth = np.unravel_index(int(np.argmax(gap)), gap.shape)

    sensitivity_thresholds = np.asarray(COVERAGE_SENSITIVITY_THRESHOLDS, dtype=np.float64)
    selected_gt = gt[:, best_upper, best_depth]
    selected_floating = floating[:, best_upper, best_depth]
    selected_top1 = top1[:, best_upper, best_depth]
    base_success = np.asarray([row["top1_base_success"] for row in rows], dtype=bool)
    sensitivity = {
        "thresholds": sensitivity_thresholds,
        "gt_pass_rate": np.asarray(
            [(selected_gt >= threshold - PASS_EPS).mean() for threshold in sensitivity_thresholds]
        ),
        "floating_gt_pass_rate": np.asarray(
            [
                (selected_floating >= threshold - PASS_EPS).mean()
                for threshold in sensitivity_thresholds
            ]
        ),
        "top1_pass_rate": np.asarray(
            [(selected_top1 >= threshold - PASS_EPS).mean() for threshold in sensitivity_thresholds]
        ),
        "top1_gated_placement_success_rate": np.asarray(
            [
                (base_success & (selected_top1 >= threshold - PASS_EPS)).mean()
                for threshold in sensitivity_thresholds
            ]
        ),
    }
    gt_quantiles = np.percentile(selected_gt, [10, 25, 50, 75, 90])
    by_source = {}
    source_names = np.asarray([row["source_name"] for row in rows])
    for source_name in sorted(set(source_names.tolist())):
        source_mask = source_names == source_name
        by_source[source_name] = {
            "sample_count": int(source_mask.sum()),
            "gt_full_coverage_rate": float(
                (selected_gt[source_mask] >= 1.0 - PASS_EPS).mean()
            ),
            "gt_90pct_coverage_rate": float((selected_gt[source_mask] >= 0.9).mean()),
            "top1_full_coverage_rate": float(
                (selected_top1[source_mask] >= 1.0 - PASS_EPS).mean()
            ),
            "top1_90pct_coverage_rate": float((selected_top1[source_mask] >= 0.9).mean()),
        }
    return {
        "sample_count": len(rows),
        "depths_cm": _serialize_array(depths_cm),
        "upper_tolerances_cm": _serialize_array(upper_tolerances_cm),
        "coverage_threshold": float(coverage_threshold),
        "base_placement_without_support_at_1": float(np.mean([row["top1_base_success"] for row in rows])),
        "base_placement_without_support_at_5": float(np.mean([row["top5_base_success"] for row in rows])),
        **{key: _serialize_array(value) for key, value in rates.items()},
        "calibration_gap": _serialize_array(gap),
        "coverage_threshold_sensitivity": {
            key: _serialize_array(value) for key, value in sensitivity.items()
        },
        "gt_coverage_at_recommendation": {
            "mean": float(selected_gt.mean()),
            "p10": float(gt_quantiles[0]),
            "p25": float(gt_quantiles[1]),
            "median": float(gt_quantiles[2]),
            "p75": float(gt_quantiles[3]),
            "p90": float(gt_quantiles[4]),
        },
        "by_source_at_recommendation": by_source,
        "heuristic_recommendation": {
            "depth_cm": float(depths_cm[best_depth]),
            "upper_tolerance_cm": float(upper_tolerances_cm[best_upper]),
            "gt_full_coverage_rate": float(rates["gt_full_coverage_rate"][best_upper, best_depth]),
            "floating_gt_full_coverage_rate": float(
                rates["floating_gt_full_coverage_rate"][best_upper, best_depth]
            ),
            "gap": float(gap[best_upper, best_depth]),
            "criterion": (
                "smallest thickness within 0.1 percentage points of the best GT-minus-"
                "floating gap, with floating pass rate <= 1%"
            ),
        },
    }


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "savefig.bbox": "tight",
        }
    )


def plot_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """Plot calibration and support-gated benchmark sensitivity."""
    _plot_style()
    depths = np.asarray(summary["depths_cm"], dtype=np.float64)
    uppers = np.asarray(summary["upper_tolerances_cm"], dtype=np.float64)
    colors = ("#0072B2", "#D55E00")
    linestyles = ("-", "--")
    markers = ("o", "s")
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.45), constrained_layout=True)

    for upper_index, upper in enumerate(uppers):
        label_suffix = f"upper tol. {upper:g} cm"
        axes[0].plot(
            depths,
            np.asarray(summary["gt_full_coverage_rate"])[upper_index] * 100.0,
            color=colors[upper_index % len(colors)],
            linestyle=linestyles[upper_index % len(linestyles)],
            marker=markers[upper_index % len(markers)],
            label=f"GT, {label_suffix}",
        )
        axes[0].plot(
            depths,
            np.asarray(summary["floating_gt_full_coverage_rate"])[upper_index] * 100.0,
            color=colors[upper_index % len(colors)],
            linestyle=":",
            marker="x",
            label=f"GT +5 cm, {label_suffix}",
        )
        axes[1].plot(
            depths,
            np.asarray(summary["top1_full_coverage_rate"])[upper_index] * 100.0,
            color=colors[upper_index % len(colors)],
            linestyle=linestyles[upper_index % len(linestyles)],
            marker=markers[upper_index % len(markers)],
            label=f"Support@1, {label_suffix}",
        )
        axes[1].plot(
            depths,
            np.asarray(summary["top5_full_coverage_rate"])[upper_index] * 100.0,
            color=colors[upper_index % len(colors)],
            linestyle=":",
            marker="^",
            label=f"Support@5, {label_suffix}",
        )
        axes[2].plot(
            depths,
            np.asarray(summary["top1_support_gated_placement_success_rate"])[upper_index] * 100.0,
            color=colors[upper_index % len(colors)],
            linestyle=linestyles[upper_index % len(linestyles)],
            marker=markers[upper_index % len(markers)],
            label=f"Task@1 + support, {label_suffix}",
        )
        axes[2].plot(
            depths,
            np.asarray(summary["top5_support_gated_placement_success_rate"])[upper_index] * 100.0,
            color=colors[upper_index % len(colors)],
            linestyle=":",
            marker="^",
            label=f"Task@5 + support, {label_suffix}",
        )

    axes[2].axhline(
        float(summary["base_placement_without_support_at_1"]) * 100.0,
        color="#666666",
        linewidth=1.0,
        linestyle="-.",
        label="Current Task@1",
    )
    panel_titles = ("(a) GT calibration", "(b) Predicted support", "(c) Benchmark impact")
    for axis, ylabel, title in zip(
        axes,
        ("Full-coverage rate (%)", "Prediction pass rate (%)", "Task success rate (%)"),
        panel_titles,
    ):
        axis.set_xlabel("Downward band thickness (cm)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(depths)
        axis.set_ylim(0.0, 100.0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.legend(frameon=False, loc="best")
        axis.set_title(title, loc="left", fontsize=8, pad=3)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"support_depth_sweep.{suffix}", dpi=220)
    plt.close(fig)

    sensitivity = summary["coverage_threshold_sensitivity"]
    thresholds = np.asarray(sensitivity["thresholds"], dtype=np.float64) * 100.0
    fig, axis = plt.subplots(figsize=(3.35, 2.5), constrained_layout=True)
    series = (
        ("GT", "gt_pass_rate", "#0072B2", "-", "o"),
        ("Top-1 prediction", "top1_pass_rate", "#009E73", "--", "s"),
        ("Placement@1", "top1_gated_placement_success_rate", "#CC79A7", "-.", "^"),
        ("GT +5 cm", "floating_gt_pass_rate", "#D55E00", ":", "x"),
    )
    for label, key, color, linestyle, marker in series:
        axis.plot(
            thresholds,
            np.asarray(sensitivity[key]) * 100.0,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )
    recommendation = summary["heuristic_recommendation"]
    axis.set_xlabel("Required footprint coverage (%)")
    axis.set_ylabel("Pass rate (%)")
    axis.set_xlim(thresholds.min(), thresholds.max())
    axis.set_ylim(0.0, 100.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    axis.set_title(
        f"down={recommendation['depth_cm']:g} cm, upper={recommendation['upper_tolerance_cm']:g} cm",
        loc="left",
        fontsize=8,
        pad=3,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"coverage_threshold_sensitivity.{suffix}", dpi=220)
    plt.close(fig)


def _case_indices(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    case_count: int,
) -> list[tuple[str, int]]:
    recommendation = summary["heuristic_recommendation"]
    upper_index = summary["upper_tolerances_cm"].index(recommendation["upper_tolerance_cm"])
    depth_index = summary["depths_cm"].index(recommendation["depth_cm"])
    candidates: list[tuple[str, int]] = []
    predicates = (
        ("pass", lambda curve: curve[upper_index][depth_index] >= 1.0 - PASS_EPS),
        (
            "depth-sensitive",
            lambda curve: curve[upper_index][0] < 1.0 - PASS_EPS
            and curve[upper_index][depth_index] >= 1.0 - PASS_EPS,
        ),
        ("fail", lambda curve: curve[upper_index][-1] < 1.0 - PASS_EPS),
    )
    used = set()
    for label, predicate in predicates:
        for index, row in enumerate(rows):
            if index not in used and predicate(row["top1_coverage"]):
                candidates.append((label, index))
                used.add(index)
                break
    for index in range(len(rows)):
        if len(candidates) >= int(case_count):
            break
        if index not in used:
            candidates.append(("additional", index))
            used.add(index)
    return candidates[: int(case_count)]


def plot_case(
    label: str,
    row: dict[str, Any],
    occupancy: OccupancyColumns,
    upper_tolerance_cm: float,
    selected_depth_cm: float,
    output_path: Path,
) -> None:
    """Visualize the 3D band, top-down footprint cells and side profile."""
    _plot_style()
    box = np.asarray(row["top1_box"], dtype=np.float64)
    footprint, distances = footprint_support_distances(box, occupancy, upper_tolerance_cm)
    supported = distances <= float(selected_depth_cm) + PASS_EPS
    bottom_z = float(box[2] - box[5] * 0.5)
    voxel_size = occupancy.voxel_size_cm
    footprint_xy = (footprint.astype(np.float64) + 0.5) * voxel_size
    radius = float(max(box[3], box[4]) + selected_depth_cm + 3.0)
    points = occupancy.points_world
    local = (
        (np.abs(points[:, 0] - box[0]) <= radius)
        & (np.abs(points[:, 1] - box[1]) <= radius)
        & (points[:, 2] >= bottom_z - selected_depth_cm - 3.0)
        & (points[:, 2] <= bottom_z + 3.0)
    )
    local_points = points[local]

    fig = plt.figure(figsize=(8.2, 2.65), constrained_layout=True)
    ax3d = fig.add_subplot(131, projection="3d")
    ax_top = fig.add_subplot(132)
    ax_side = fig.add_subplot(133)
    if len(local_points):
        ax3d.scatter(
            local_points[:, 0], local_points[:, 1], local_points[:, 2],
            s=3, c="#808080", alpha=0.35, depthshade=False,
        )
    ax3d.scatter(
        footprint_xy[supported, 0], footprint_xy[supported, 1],
        np.full(int(supported.sum()), bottom_z), s=8, c="#009E73", marker="s", label="covered",
    )
    ax3d.scatter(
        footprint_xy[~supported, 0], footprint_xy[~supported, 1],
        np.full(int((~supported).sum()), bottom_z), s=8, c="#D55E00", marker="s", label="uncovered",
    )
    ax3d.set_xlabel("world X (cm)")
    ax3d.set_ylabel("world Y (cm)")
    ax3d.set_zlabel("world Z (cm)")
    ax3d.legend(frameon=False, loc="best")

    ax_top.scatter(footprint_xy[:, 0], footprint_xy[:, 1], c=np.where(supported, "#009E73", "#D55E00"), s=18, marker="s")
    if len(local_points):
        ax_top.scatter(local_points[:, 0], local_points[:, 1], c="#999999", s=2, alpha=0.25)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("world X (cm)")
    ax_top.set_ylabel("world Y (cm)")

    yaw_axis = np.array([math.cos(box[6]), math.sin(box[6])], dtype=np.float64)
    if len(local_points):
        along = (local_points[:, :2] - box[:2]) @ yaw_axis
        ax_side.scatter(along, local_points[:, 2], c="#777777", s=4, alpha=0.4)
    ax_side.axhspan(
        bottom_z - selected_depth_cm,
        bottom_z + upper_tolerance_cm,
        color="#56B4E9",
        alpha=0.18,
        label="tested Z band",
    )
    ax_side.axhline(bottom_z, color="#000000", linewidth=1.2, label="box bottom")
    ax_side.set_xlim(-radius, radius)
    ax_side.set_xlabel("box-local X (cm)")
    ax_side.set_ylabel("world Z (cm)")
    ax_side.legend(frameon=False, loc="best")

    coverage = float(supported.mean()) if len(supported) else 0.0
    fig.suptitle(
        f"{label}: {row['item_id']} | coverage={coverage:.3f}, "
        f"down={selected_depth_cm:g} cm, upper={upper_tolerance_cm:g} cm",
        fontsize=8,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)


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

    recommendation = summary["heuristic_recommendation"]
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in list(cases_dir.glob("case_*.png")) + list(cases_dir.glob("case_*.pdf")):
        stale_path.unlink()
    for case_number, (label, index) in enumerate(_case_indices(rows, summary, case_count), start=1):
        row = rows[index]
        points, _ = load_ply(row["voxel_point_cloud_path"])
        occupancy = build_occupancy_columns(points, float(summary["voxel_size_cm"]))
        plot_case(
            label,
            row,
            occupancy,
            float(recommendation["upper_tolerance_cm"]),
            float(recommendation["depth_cm"]),
            cases_dir / f"case_{case_number:02d}_{label}",
        )


def main() -> None:
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    predictions = load_predictions(args.predictions)
    if args.max_samples is not None:
        predictions = predictions[: int(args.max_samples)]
    direction_metadata = load_direction_metadata(args.direction_metadata)
    depths_cm = np.sort(np.unique(np.asarray(args.depths_cm, dtype=np.float64)))
    upper_tolerances_cm = np.sort(
        np.unique(np.asarray(args.upper_tolerances_cm, dtype=np.float64))
    )
    rows, summary = analyze_predictions(
        cfg,
        predictions,
        direction_metadata,
        args.split,
        depths_cm,
        upper_tolerances_cm,
        float(args.coverage_threshold),
        float(args.floating_offset_cm),
        float(args.size_iou_threshold),
    )
    save_outputs(rows, summary, args.output_dir, int(args.case_count))
    recommendation = summary["heuristic_recommendation"]
    print(
        "Recommendation: "
        f"down={recommendation['depth_cm']:.1f} cm, "
        f"upper={recommendation['upper_tolerance_cm']:.1f} cm, "
        f"GT={recommendation['gt_full_coverage_rate']:.4f}, "
        f"floating={recommendation['floating_gt_full_coverage_rate']:.4f}"
    )
    print(f"Wrote support coverage analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
