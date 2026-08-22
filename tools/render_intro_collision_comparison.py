#!/usr/bin/env python
"""Render camera-aligned Introduction comparisons for selected placement cases.

The RoboBrain 3D candidate is visualization-only: it is reconstructed from
the predicted 2D point using the source object's observed geometry.

使用示例:
    python tools/render_intro_collision_comparison.py --all-suitable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from baselines.robobrain2_5.run_zero_shot_test import (
    load_scene,
    load_sources,
)
from baselines.robobrain2_5.zero_shot_visualization import (
    BOX_EDGES,
    Camera,
    aabb_corners,
    convex_hull_xy,
    project_world,
    transform_points,
    upright_box_corners,
)
from src.annotation.auto_label import camera_image_axes_world_xy
from src.training.lc_bgplacenet_stage2 import build_collision_context, compute_collision_metrics


DEFAULT_CONFIG = PROJECT_ROOT / "configs/lc_bgplacenet_stage2_enriched.yaml"
DEFAULT_ROBOBRAIN_RESULTS = PROJECT_ROOT / "outputs/robobrain2_5_front_behind_swapped/predictions.jsonl"
DEFAULT_OURS_RESULTS = PROJECT_ROOT / (
    "outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/"
    "inference_stage2_test/predictions.json"
)
DEFAULT_OURS_METRICS = PROJECT_ROOT / (
    "outputs/lc_bgplacenet_stage2_space_former_aligned_loss_48query_full_gt_guass_8_enriched/"
    "benchmark_stage2_test/per_sample_metrics.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/introduction_collision_comparison_front_behind_swapped"

SOURCE_COLOR = "#0891B2"
ROBOBRAIN_COLOR = "#DC2626"
OURS_COLOR = "#2563EB"
REFERENCE_COLOR = "#64748B"
SCENE_FILL = "#F1F5F9"
SCENE_EDGE = "#CBD5E1"
COLLISION_FILL = "#991B1B"

BOX_FACES = (
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Introduction collision comparison figure.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all-suitable", action="store_true")
    mode.add_argument("--item-id", help="Render one selected item using its full ID or stable label key.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robobrain-results", type=Path, default=DEFAULT_ROBOBRAIN_RESULTS)
    parser.add_argument("--ours-results", type=Path, default=DEFAULT_OURS_RESULTS)
    parser.add_argument("--ours-metrics", type=Path, default=DEFAULT_OURS_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate complete existing sample folders.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_item_key(item_id: str) -> str:
    """Remove the content-derived hash while retaining source and label index."""
    key, separator, _ = str(item_id).rpartition("__")
    if not separator:
        raise ValueError(f"Item ID has no hash suffix: {item_id}")
    return key


def box_to_corners(box: list[float] | np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    return upright_box_corners(box[:3], box[3:6], float(box[6]))


def object_corners(obj: dict[str, Any]) -> np.ndarray:
    return transform_points(aabb_corners(np.asarray(obj["bbox3d_canonical"])), np.asarray(obj["pose_world"]))


def camera_aligned_basis(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Return the same image-right/image-up world-XY axes used by annotation."""
    axes = camera_image_axes_world_xy(camera.e_w2c)
    if axes is None:
        raise ValueError("Camera image X axis has no stable horizontal projection")
    return axes


def align_xy(points_xy: np.ndarray, camera: Camera) -> np.ndarray:
    right, image_up = camera_aligned_basis(camera)
    points = np.asarray(points_xy, dtype=np.float64)
    return np.column_stack((points @ right, points @ image_up))


def draw_rgb_box(
    ax: plt.Axes,
    corners: np.ndarray,
    camera: Camera,
    color: str,
    label: str | None,
    *,
    alpha: float,
    linestyle: str = "-",
) -> None:
    uv, depth = project_world(corners, camera)
    visible_faces = [face for face in BOX_FACES if np.all(depth[list(face)] > 0.0)]
    visible_faces.sort(key=lambda face: float(depth[list(face)].mean()), reverse=True)
    for face in visible_faces:
        ax.add_patch(Polygon(uv[list(face)], closed=True, facecolor=color, edgecolor="none", alpha=alpha))
    for start, end in BOX_EDGES:
        if depth[start] > 0.0 and depth[end] > 0.0:
            ax.plot(
                uv[[start, end], 0],
                uv[[start, end], 1],
                color=color,
                linewidth=1.4,
                linestyle=linestyle,
                solid_capstyle="round",
            )
    visible = depth > 0.0
    if label and np.any(visible):
        visible_uv = uv[visible]
        box_center = visible_uv.mean(axis=0)
        label_xy = visible_uv.min(axis=0) + np.array([4.0, 4.0])
        ax.annotate(
            label,
            box_center,
            xytext=label_xy,
            textcoords="data",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="semibold",
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8, "shrinkA": 2, "shrinkB": 2},
        )


def draw_status_chip(ax: plt.Axes, text: str, color: str) -> None:
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": "white", "linewidth": 0.8},
    )


def configure_rgb_axis(ax: plt.Axes, rgb: np.ndarray, title: str, title_color: str) -> None:
    ax.imshow(rgb)
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(1.0)
    ax.set_title(title, color=title_color, fontsize=12, fontweight="bold", pad=7)


def add_card_background(ax: plt.Axes) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (-0.015, -0.02),
            1.03,
            1.04,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#F8FAFC",
            edgecolor="#E2E8F0",
            linewidth=1.0,
            clip_on=False,
            zorder=-20,
        )
    )


def draw_camera_orientation(ax: plt.Axes) -> None:
    ax.annotate(
        "Image up / Front",
        xy=(0.50, 0.17),
        xytext=(0.50, 0.045),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#475569",
        arrowprops={"arrowstyle": "-|>", "color": "#475569", "linewidth": 1.2},
    )


def draw_topdown(
    ax: plt.Axes,
    scene_objects: list[dict[str, Any]],
    camera: Camera,
    source_id: str,
    candidate_corners: np.ndarray,
    candidate_color: str,
    highlighted_ids: set[str],
    object_labels: dict[str, str],
    colliding_ids: set[str],
    limits: tuple[float, float, float, float],
    status: str,
) -> None:
    polygons = {
        str(obj["obj_id"]): align_xy(convex_hull_xy(object_corners(obj)[:, :2]), camera)
        for obj in scene_objects
    }
    source_polygon = polygons[source_id]
    candidate_polygon = align_xy(convex_hull_xy(candidate_corners[:, :2]), camera)

    add_card_background(ax)
    for obj_id, polygon in polygons.items():
        if obj_id == source_id:
            continue
        is_highlighted = obj_id in highlighted_ids
        facecolor = "#F8FAFC" if is_highlighted else SCENE_FILL
        edgecolor = REFERENCE_COLOR if is_highlighted else SCENE_EDGE
        ax.add_patch(Polygon(polygon, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0))
        if is_highlighted:
            center = polygon.mean(axis=0)
            ax.text(
                *center,
                object_labels[obj_id],
                ha="center",
                va="center",
                fontsize=8,
                color=REFERENCE_COLOR,
                fontweight="bold",
            )

    ax.add_patch(
        Polygon(
            source_polygon,
            closed=True,
            facecolor=SOURCE_COLOR,
            edgecolor=SOURCE_COLOR,
            linewidth=1.4,
            linestyle=(0, (4, 2)),
            alpha=0.16,
        )
    )
    candidate_patch = Polygon(
        candidate_polygon,
        closed=True,
        facecolor=candidate_color,
        edgecolor=candidate_color,
        linewidth=1.6,
        alpha=0.10,
    )
    ax.add_patch(candidate_patch)

    for obj_id in colliding_ids:
        clip_polygon = Polygon(polygons[obj_id], closed=True, transform=ax.transData)
        overlap = Polygon(
            candidate_polygon,
            closed=True,
            facecolor=COLLISION_FILL,
            edgecolor="none",
            alpha=0.78,
        )
        overlap.set_clip_path(clip_polygon)
        ax.add_patch(overlap)

    source_center = source_polygon.mean(axis=0)
    candidate_center = candidate_polygon.mean(axis=0)
    ax.add_patch(
        FancyArrowPatch(
            source_center,
            candidate_center,
            connectionstyle="arc3,rad=-0.16",
            arrowstyle="-|>",
            mutation_scale=12,
            color=candidate_color,
            linewidth=1.5,
            alpha=0.9,
        )
    )
    movement = candidate_center - source_center
    label_offset = max((limits[1] - limits[0]) * 0.035, 0.8)
    horizontal_sign = -1.0 if movement[0] < 0.0 else 1.0
    ax.text(
        *(source_center - np.array([horizontal_sign * label_offset, 0.0])),
        "Source",
        ha="left" if horizontal_sign < 0.0 else "right",
        va="center",
        fontsize=8,
        color="#0E7490",
        fontweight="bold",
    )
    ax.text(
        *(candidate_center + np.array([horizontal_sign * label_offset, 0.0])),
        "Prediction",
        ha="right" if horizontal_sign < 0.0 else "left",
        va="center",
        fontsize=8,
        color=candidate_color,
        fontweight="bold",
    )
    draw_camera_orientation(ax)
    ax.text(
        0.02,
        0.97,
        status,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=candidate_color,
        fontweight="bold",
    )
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def shared_topdown_limits(
    scene_objects: list[dict[str, Any]],
    camera: Camera,
    candidate_sets: list[np.ndarray],
) -> tuple[float, float, float, float]:
    object_points = [align_xy(object_corners(obj)[:, :2], camera) for obj in scene_objects]
    candidate_points = [align_xy(corners[:, :2], camera) for corners in candidate_sets]
    points = np.vstack([*object_points, *candidate_points])
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    margin = max(float(span.max()) * 0.10, 1.0)
    return float(low[0] - margin), float(high[0] + margin), float(low[1] - margin), float(high[1] + margin)


def create_panel() -> tuple[plt.Figure, plt.Axes]:
    """Create a consistently sized standalone panel for later composition."""
    figure = plt.figure(figsize=(3.5, 2.7), facecolor="white")
    axis = figure.add_axes((0.04, 0.05, 0.92, 0.86))
    return figure, axis


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    return {"png": os.fspath(png_path), "pdf": os.fspath(pdf_path)}


def object_display_name(obj: dict[str, Any]) -> str:
    name = str(obj.get("class_name") or obj.get("obj_id", "Object"))
    return name.replace("_", " ").strip().title()


def draw_case_axes(
    axes: dict[str, plt.Axes],
    scene: dict[str, Any],
    source_id: str,
    collision_ids: set[str],
    object_labels: dict[str, str],
    source_corners: np.ndarray,
    robobrain_corners: np.ndarray,
    ours_corners: np.ndarray,
    limits: tuple[float, float, float, float],
) -> None:
    highlighted_objects = [obj for obj in scene["objects"] if str(obj["obj_id"]) in collision_ids]
    rgb_specs = (
        ("robobrain_rgb", "RoboBrain2.5", ROBOBRAIN_COLOR, robobrain_corners),
        ("ours_rgb", "Ours", OURS_COLOR, ours_corners),
    )
    for panel_name, title, color, candidate_corners in rgb_specs:
        if panel_name not in axes:
            continue
        axis = axes[panel_name]
        configure_rgb_axis(axis, scene["rgb"], title, color)
        draw_rgb_box(axis, source_corners, scene["camera"], SOURCE_COLOR, None, alpha=0.07, linestyle="--")
        for obj in highlighted_objects:
            obj_id = str(obj["obj_id"])
            draw_rgb_box(
                axis,
                object_corners(obj),
                scene["camera"],
                REFERENCE_COLOR,
                None,
                alpha=0.05,
            )
        draw_rgb_box(axis, candidate_corners, scene["camera"], color, None, alpha=0.10)
        if panel_name == "robobrain_rgb":
            draw_status_chip(axis, "×  Collision", ROBOBRAIN_COLOR)
        else:
            draw_status_chip(axis, "✓  Collision-free", "#15803D")

    topdown_specs = (
        (
            "robobrain_topdown",
            robobrain_corners,
            ROBOBRAIN_COLOR,
            collision_ids,
            "× Collision overlap",
        ),
        ("ours_topdown", ours_corners, OURS_COLOR, set(), "✓ Free placement region"),
    )
    for panel_name, candidate_corners, color, colliding_ids, status in topdown_specs:
        if panel_name not in axes:
            continue
        draw_topdown(
            axes[panel_name],
            scene["objects"],
            scene["camera"],
            source_id,
            candidate_corners,
            color,
            collision_ids,
            object_labels,
            colliding_ids,
            limits,
            status,
        )


def create_comparison_figure(instruction: str) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    figure = plt.figure(figsize=(7.16, 5.0), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.035,
        right=0.985,
        bottom=0.045,
        top=0.80,
        wspace=0.07,
        hspace=0.10,
        height_ratios=(1.0, 0.92),
    )
    axes = {
        "robobrain_rgb": figure.add_subplot(grid[0, 0]),
        "ours_rgb": figure.add_subplot(grid[0, 1]),
        "robobrain_topdown": figure.add_subplot(grid[1, 0]),
        "ours_topdown": figure.add_subplot(grid[1, 1]),
    }
    figure.text(
        0.5,
        0.955,
        textwrap.fill(instruction, width=92),
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="semibold",
        color="#0F172A",
    )
    figure.text(
        0.5,
        0.895,
        "Camera-aligned top view: image right = map right, image up / front = map up",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#64748B",
    )
    return figure, axes


def render_case(
    ours: dict[str, Any],
    scene: dict[str, Any],
    robobrain: dict[str, Any],
    collision_ids: set[str],
    output_dir: Path,
) -> dict[str, Any]:
    source_id = str(ours["object_id"])
    source_obj = next(obj for obj in scene["objects"] if str(obj["obj_id"]) == source_id)
    source_corners = object_corners(source_obj)
    robobrain_corners = box_to_corners(robobrain["render_box_world"])
    ours_corners = box_to_corners(ours["place_box"])
    object_labels = {
        str(obj["obj_id"]): object_display_name(obj)
        for obj in scene["objects"]
        if str(obj["obj_id"]) in collision_ids
    }
    if set(object_labels) != collision_ids:
        missing = sorted(collision_ids - set(object_labels))
        raise KeyError(f"Collision objects missing from scene {ours['sample_id']}: {missing}")
    limits = shared_topdown_limits(scene["objects"], scene["camera"], [robobrain_corners, ours_corners])

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = {}
    for panel_name in ("robobrain_rgb", "robobrain_topdown", "ours_rgb", "ours_topdown"):
        figure, axis = create_panel()
        draw_case_axes(
            {panel_name: axis},
            scene,
            source_id,
            collision_ids,
            object_labels,
            source_corners,
            robobrain_corners,
            ours_corners,
            limits,
        )
        panels[panel_name] = save_figure(figure, output_dir, panel_name)

    comparison_figure, comparison_axes = create_comparison_figure(str(ours["instruction"]))
    draw_case_axes(
        comparison_axes,
        scene,
        source_id,
        collision_ids,
        object_labels,
        source_corners,
        robobrain_corners,
        ours_corners,
        limits,
    )
    comparison = save_figure(comparison_figure, output_dir, "comparison")

    metadata = {
        "item_id": ours["item_id"],
        "stable_item_key": stable_item_key(str(ours["item_id"])),
        "instruction": str(ours["instruction"]),
        "robobrain_collision_object_ids": sorted(collision_ids),
        "robobrain_collision_object_names": [object_labels[obj_id] for obj_id in sorted(collision_ids)],
        "ours_collision": False,
        "robobrain_box_is_visualization_only": True,
        "robobrain_box_derivation": "2D point + observed source dimensions and yaw",
        "camera_alignment": "image up/front maps up; image right maps right",
        "panels": panels,
        "comparison": comparison,
    }
    return metadata


def load_collision_context(source: dict[str, Path], sample_id: str) -> dict[str, Any]:
    placement_path = source["free_bbox_dir"] / "placements" / f"{sample_id}__placements.json"
    with placement_path.open("r", encoding="utf-8") as handle:
        return build_collision_context(json.load(handle).get("objects", []))


def select_suitable_cases(
    ours_rows: list[dict[str, Any]],
    robobrain_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Path]],
) -> list[dict[str, Any]]:
    """Select task-successful predictions where RoboBrain hits another scene object."""
    robobrain_by_key = {stable_item_key(str(row["item_id"])): row for row in robobrain_rows}
    metrics_by_id = {str(row["item_id"]): row for row in metrics_rows}
    if len(robobrain_by_key) != len(robobrain_rows):
        raise ValueError("RoboBrain results contain duplicate stable label keys")
    if len(metrics_by_id) != len(metrics_rows):
        raise ValueError("Ours benchmark metrics contain duplicate item IDs")

    collision_cache: dict[tuple[str, str], dict[str, Any]] = {}
    selected = []
    for ours in ours_rows:
        item_id = str(ours["item_id"])
        key = stable_item_key(item_id)
        robobrain = robobrain_by_key.get(key)
        metrics = metrics_by_id.get(item_id)
        if robobrain is None or metrics is None:
            raise KeyError(f"Missing aligned result or benchmark metrics for {item_id}")
        if robobrain.get("status") != "ok" or robobrain.get("render_box_world") is None:
            continue
        if not bool(metrics.get("task_success")):
            continue

        scene_key = (str(ours["source_name"]), str(ours["sample_id"]))
        if scene_key not in collision_cache:
            collision_cache[scene_key] = load_collision_context(sources[scene_key[0]], scene_key[1])
        collision = compute_collision_metrics(np.asarray(robobrain["render_box_world"]), collision_cache[scene_key])
        external_collision_ids = sorted(set(collision["collision_object_ids"]) - {str(ours["object_id"])})
        if external_collision_ids:
            selected.append(
                {
                    "stable_item_key": key,
                    "ours": ours,
                    "robobrain": robobrain,
                    "metrics": metrics,
                    "collision_object_ids": external_collision_ids,
                }
            )
    return sorted(
        selected,
        key=lambda row: (
            str(row["ours"]["source_name"]),
            str(row["ours"]["sample_id"]),
            str(row["stable_item_key"]),
        ),
    )


def expected_outputs(output_dir: Path) -> list[Path]:
    return [
        output_dir / f"{stem}.{suffix}"
        for stem in ("robobrain_rgb", "robobrain_topdown", "ours_rgb", "ours_topdown", "comparison")
        for suffix in ("png", "pdf")
    ]


def write_manifest(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            ours = case["ours"]
            metrics = case["metrics"]
            row = {
                "stable_item_key": case["stable_item_key"],
                "ours_item_id": ours["item_id"],
                "robobrain_item_id": case["robobrain"]["item_id"],
                "source_name": ours["source_name"],
                "sample_id": ours["sample_id"],
                "object_id": ours["object_id"],
                "instruction": ours["instruction"],
                "target_relation": metrics["target_relation"],
                "reference_object_id": metrics["reference_object_id"],
                "reference_name": metrics["reference_name"],
                "robobrain_collision_object_ids": case["collision_object_ids"],
                "ours_task_success": True,
                "output_dir": os.fspath(path.parent / case["stable_item_key"]),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(config_path)
    ours_rows = load_json_rows(resolve_path(args.ours_results))
    robobrain_rows = load_jsonl_rows(resolve_path(args.robobrain_results))
    metrics_rows = load_jsonl_rows(resolve_path(args.ours_metrics))
    cases = select_suitable_cases(ours_rows, robobrain_rows, metrics_rows, sources)
    if args.item_id:
        requested = str(args.item_id)
        cases = [
            case
            for case in cases
            if requested in {str(case["stable_item_key"]), str(case["ours"]["item_id"]), str(case["robobrain"]["item_id"])}
        ]
        if not cases:
            raise KeyError(f"Item {requested!r} is not in the strictly selected comparison set")

    write_manifest(output_dir / "selection_manifest.jsonl", cases)
    rendered = skipped = 0
    loaded_scene_key: tuple[str, str] | None = None
    scene: dict[str, Any] | None = None
    for index, case in enumerate(cases, start=1):
        ours = case["ours"]
        sample_output_dir = output_dir / str(case["stable_item_key"])
        if not args.overwrite and all(path.is_file() for path in expected_outputs(sample_output_dir)):
            skipped += 1
            continue

        scene_key = (str(ours["source_name"]), str(ours["sample_id"]))
        if scene_key != loaded_scene_key:
            source = sources[scene_key[0]]
            scene = load_scene(str(source["dataset_dir"]), scene_key[1])
            loaded_scene_key = scene_key
        assert scene is not None
        render_case(
            ours,
            scene,
            case["robobrain"],
            set(case["collision_object_ids"]),
            sample_output_dir,
        )
        rendered += 1
        if index == 1 or index % 25 == 0 or index == len(cases):
            print(f"[{index}/{len(cases)}] rendered={rendered} skipped={skipped}", flush=True)

    print(
        json.dumps(
            {
                "selected": len(cases),
                "rendered": rendered,
                "skipped": skipped,
                "output_dir": os.fspath(output_dir),
                "manifest": os.fspath(output_dir / "selection_manifest.jsonl"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
