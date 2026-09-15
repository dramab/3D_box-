#!/usr/bin/env python
"""Render a GT-driven replacement teaser figure for TopoBox.

使用示例:
    python tools/render_topobox_teaser_gt.py \
        --config configs/lc_bgplacenet_stage2_enriched.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.fspath(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from src.annotation.free_bbox.io_utils import load_ply
from src.datasets.canonical import CanonicalScene, ObjectInfo, load_canonical_scene
from src.training.lc_bgplacenet_stage2 import (
    Stage2IndexItem,
    build_sources_from_config,
    build_stage2_index,
    load_config,
    select_stage2_split_items,
)
from tools.infer_lc_bgplacenet_stage2 import (
    _find_scene_object as find_scene_object,
    draw_world_corners,
    place_box_to_corners,
    source_box_to_oriented_corners,
)


FIGURE_SIZE = (1536, 1024)
SOURCE_COLOR = (225, 40, 74)
GT_COLOR = (35, 143, 78)
SEMANTIC_COLOR = (43, 112, 219)
SUPPORT_COLOR = (244, 122, 24)
COLLISION_COLOR = (68, 151, 55)
TOPO_COLOR = (109, 70, 168)
BASELINE_GRAY = (242, 242, 242)
TEXT_DARK = (18, 18, 18)


@dataclass(frozen=True)
class SlotSpec:
    """Named teaser image slot in the final 1536x1024 canvas."""

    name: str
    rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class OutputPaths:
    """Resolved output files for the generated GT teaser."""

    output_dir: Path
    figure_png: Path
    metadata_json: Path


SLOT_SPECS = (
    SlotSpec("semantic_heatmap", (162, 226, 184, 140)),
    SlotSpec("semantic_gt", (162, 430, 184, 142)),
    SlotSpec("support_float", (386, 230, 158, 128)),
    SlotSpec("support_thin", (556, 230, 164, 128)),
    SlotSpec("support_gt", (386, 460, 200, 138)),
    SlotSpec("collision_overlap", (764, 230, 336, 132)),
    SlotSpec("collision_gt", (764, 460, 230, 136)),
    SlotSpec("pipeline_scene", (105, 762, 88, 76)),
    SlotSpec("pipeline_heatmap", (221, 762, 118, 76)),
    SlotSpec("pipeline_candidates", (413, 766, 112, 72)),
    SlotSpec("pipeline_topology", (730, 770, 202, 100)),
    SlotSpec("pipeline_final", (1110, 730, 146, 90)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GT-driven TopoBox teaser figure.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lc_bgplacenet_stage2_enriched.yaml")
    parser.add_argument("--split", default="train", choices=("train", "valid", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--line-width", type=int, default=4)
    return parser.parse_args()


def make_output_paths(project_root: Path, output_dir: Path | None = None) -> OutputPaths:
    output = output_dir if output_dir is not None else project_root / "outputs" / "topobox_teaser_gt"
    return OutputPaths(
        output_dir=output,
        figure_png=output / "topobox_teaser_gt.png",
        metadata_json=output / "topobox_teaser_gt_metadata.json",
    )


def load_font(size: int, *, bold: bool = False, serif: bool = True) -> ImageFont.FreeTypeFont:
    family = "DejaVuSerif" if serif else "DejaVuSans"
    suffix = "-Bold.ttf" if bold else ".ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / f"{family}{suffix}"
    return ImageFont.truetype(os.fspath(path), size=size)


def wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    return int(box[2] - box[0]), int(box[3] - box[1])


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    *,
    anchor_width: int,
) -> None:
    width, _ = text_size(draw, text, font)
    draw.text((xy[0] + (anchor_width - width) // 2, xy[1]), text, font=font, fill=fill)


def resize_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def paste_slot(canvas: Image.Image, slot: SlotSpec, image: Image.Image) -> None:
    x, y, width, height = slot.rect
    canvas.paste(resize_crop(image.convert("RGB"), (width, height)), (x, y))


def project_world_points(scene: CanonicalScene, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    cam = (scene.camera.E_w2c @ homo.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-6
    uv = np.full((len(pts), 3), np.nan, dtype=np.float64)
    uv[valid, 0] = scene.camera.fx * cam[valid, 0] / z[valid] + scene.camera.cx
    uv[valid, 1] = scene.camera.fy * cam[valid, 1] / z[valid] + scene.camera.cy
    uv[valid, 2] = z[valid]
    return uv


def load_scene_for_item(item: Stage2IndexItem) -> CanonicalScene:
    sample_path = item.dataset_dir / "samples" / f"{item.sample_id}.json"
    return load_canonical_scene(sample_path, dataset_root=item.dataset_dir)


def object_name(scene: CanonicalScene, object_id: str) -> str:
    obj = find_scene_object(scene, object_id)
    return obj.class_name.replace("_", " ").strip().title()


def choose_teaser_items(config_path: Path, split: str) -> list[Stage2IndexItem]:
    cfg = load_config(config_path)
    items = select_stage2_split_items(build_stage2_index(build_sources_from_config(cfg)), cfg["data"]["split_dir"], split)
    filtered = [
        item
        for item in items
        if item.source_name == "dopose"
        and item.object_id != item.reference_object_id
        and item.rgb_path.exists()
        and item.yaw_set_npz.exists()
        and item.voxel_point_cloud_path.exists()
    ]
    if len(filtered) < 3:
        raise ValueError("Need at least three usable GT metadata items for the teaser")
    relation_order = ("behind", "the right of", "in front of")
    selected: list[Stage2IndexItem] = []
    used_samples: set[str] = set()
    for relation in relation_order:
        match = next(
            (
                item
                for item in filtered
                if relation in item.target_relation and item.sample_id not in used_samples
            ),
            None,
        )
        if match is None:
            match = next(item for item in filtered if item not in selected)
        selected.append(match)
        used_samples.add(match.sample_id)
    return selected


def make_shifted_box(item: Stage2IndexItem, *, dz: float = 0.0, xy_from_object: ObjectInfo | None = None) -> np.ndarray:
    box = np.asarray(item.place_box_gt, dtype=np.float64).copy()
    if xy_from_object is not None:
        corners = source_box_to_oriented_corners(
            np.asarray(xy_from_object.bbox3d_canonical, dtype=np.float64),
            np.asarray(xy_from_object.pose_world, dtype=np.float64),
        )
        box[:2] = corners[:, :2].mean(axis=0)
    box[2] += float(dz)
    return box


def render_rgb_box_panel(
    item: Stage2IndexItem,
    *,
    candidate_box: np.ndarray | None = None,
    candidate_color: tuple[int, int, int] = GT_COLOR,
    draw_source: bool = True,
    line_width: int = 4,
    label: str | None = None,
) -> Image.Image:
    scene = load_scene_for_item(item)
    image = Image.fromarray(scene.rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    if draw_source:
        source_obj = find_scene_object(scene, item.object_id)
        draw_world_corners(
            draw,
            source_box_to_oriented_corners(item.source_box_gt, np.asarray(source_obj.pose_world)),
            scene,
            SOURCE_COLOR,
            line_width,
            "",
        )
    box = item.place_box_gt if candidate_box is None else candidate_box
    draw_world_corners(draw, place_box_to_corners(np.asarray(box)), scene, candidate_color, line_width, "")
    if label:
        font = load_font(24, bold=True, serif=False)
        draw.rounded_rectangle((12, 12, 12 + 16 * len(label), 48), radius=8, fill=(255, 255, 255))
        draw.text((22, 16), label, font=font, fill=candidate_color)
    return image


def render_semantic_heatmap(item: Stage2IndexItem, line_width: int) -> Image.Image:
    scene = load_scene_for_item(item)
    image = Image.fromarray(scene.rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    yaw_data = np.load(item.yaw_set_npz)
    centers = np.asarray(yaw_data["bottom_center_world"], dtype=np.float64)
    if len(centers) > 400:
        centers = centers[np.linspace(0, len(centers) - 1, 400, dtype=np.int64)]
    uv = project_world_points(scene, centers)
    for x, y, z in uv[np.isfinite(uv[:, 0])]:
        if 0 <= x < image.width and 0 <= y < image.height:
            radius = 5.0 + min(10.0, 80.0 / max(float(z), 1.0))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(42, 113, 219, 95))
    image = Image.alpha_composite(image, overlay).convert("RGB")
    draw_rgb = ImageDraw.Draw(image)
    source_obj = find_scene_object(scene, item.object_id)
    draw_world_corners(
        draw_rgb,
        source_box_to_oriented_corners(item.source_box_gt, np.asarray(source_obj.pose_world)),
        scene,
        SOURCE_COLOR,
        line_width,
        "",
    )
    return image


def figure_to_image(figure: plt.Figure) -> Image.Image:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def topdown_polygon(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64)[:, :2]
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def object_corners(obj: ObjectInfo) -> np.ndarray:
    return source_box_to_oriented_corners(
        np.asarray(obj.bbox3d_canonical, dtype=np.float64),
        np.asarray(obj.pose_world, dtype=np.float64),
    )


def render_topdown_panel(
    item: Stage2IndexItem,
    *,
    candidate_box: np.ndarray | None = None,
    title: str,
    candidate_color: str,
    show_support_points: bool = False,
) -> Image.Image:
    scene = load_scene_for_item(item)
    box = item.place_box_gt if candidate_box is None else candidate_box
    candidate_corners = place_box_to_corners(np.asarray(box, dtype=np.float64))
    figure, axis = plt.subplots(figsize=(3.2, 2.15), facecolor="white")
    if show_support_points:
        points, colors = load_ply(item.voxel_point_cloud_path)
        stride = max(1, len(points) // 2500)
        axis.scatter(points[::stride, 0], points[::stride, 1], s=1.2, c=np.asarray(colors[::stride]) / 255.0, alpha=0.50)
        yaw_data = np.load(item.yaw_set_npz)
        centers = yaw_data["bottom_center_world"]
        step = max(1, len(centers) // 900)
        axis.scatter(centers[::step, 0], centers[::step, 1], s=7, c="#2563EB", alpha=0.25)
    for obj in scene.objects:
        corners = object_corners(obj)
        face = "#F3F4F6"
        edge = "#CBD5E1"
        if obj.obj_id == item.object_id:
            face = "#FEE2E2"
            edge = "#E11D48"
        axis.add_patch(Polygon(topdown_polygon(corners), closed=True, facecolor=face, edgecolor=edge, linewidth=1.0, alpha=0.75))
    axis.add_patch(
        Polygon(
            topdown_polygon(candidate_corners),
            closed=True,
            facecolor=candidate_color,
            edgecolor=candidate_color,
            linewidth=2.0,
            alpha=0.20,
        )
    )
    all_points = np.vstack([object_corners(obj)[:, :2] for obj in scene.objects] + [candidate_corners[:, :2]])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(high - low, 1e-3)
    margin = max(float(span.max()) * 0.18, 8.0)
    axis.set_xlim(low[0] - margin, high[0] + margin)
    axis.set_ylim(low[1] - margin, high[1] + margin)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=11, fontweight="bold", color=candidate_color)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=0.2)
    return figure_to_image(figure)


def render_pipeline_heatmap(item: Stage2IndexItem) -> Image.Image:
    yaw_data = np.load(item.yaw_set_npz)
    centers = np.asarray(yaw_data["bottom_center_world"], dtype=np.float64)
    counts = np.asarray(yaw_data["heat_counts"], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(2.7, 1.7), facecolor="white")
    scatter = axis.scatter(centers[:, 0], centers[:, 1], c=counts, s=5, cmap="coolwarm", alpha=0.75)
    scatter.set_clim(vmin=float(counts.min()), vmax=float(counts.max()))
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=0.1)
    return figure_to_image(figure)


def draw_title(canvas: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas)
    font = load_font(27, bold=True)
    parts = [
        ("Semantic consistency", SEMANTIC_COLOR),
        (" is not sufficient for ", TEXT_DARK),
        ("physically feasible", (232, 88, 20)),
        (" 3D object replacement.", COLLISION_COLOR),
    ]
    x = 86
    for text, color in parts:
        draw.text((x, 10), text, font=font, fill=color)
        x += int(draw.textlength(text, font=font))


def draw_panel_frame(canvas: Image.Image, rect: tuple[int, int, int, int], color: tuple[int, int, int], title: str) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = rect
    draw.rounded_rectangle((x, y, x + w, y + h), radius=7, outline=color, width=2, fill=(255, 255, 255))
    title_font = load_font(21, bold=True)
    draw.text((x + 18, y + 16), title, font=title_font, fill=color)


def draw_card(canvas: Image.Image, rect: tuple[int, int, int, int], heading: str, color: tuple[int, int, int] = TEXT_DARK) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = rect
    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill=BASELINE_GRAY)
    draw.text((x + 14, y + 10), heading, font=load_font(16, bold=True), fill=color)


def draw_check(draw: ImageDraw.ImageDraw, xy: tuple[int, int], ok: bool) -> None:
    font = load_font(24, bold=True, serif=False)
    draw.text(xy, "✓" if ok else "×", font=font, fill=(31, 143, 55) if ok else (220, 38, 38))


def draw_radar(canvas: Image.Image, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    figure = plt.figure(figsize=(3.6, 3.3), facecolor="white")
    labels = ["Semantic\nCons.", "Stable\nSupport", "Collision\nFree", "Orient.\nAcc.", "Center\nAcc.", "Overall\nSuccess"]
    values = {
        "TopoBox (GT vis)": [91.2, 89.4, 93.1, 86.7, 85.5, 87.6],
        "VG-LLM": [72.3, 58.7, 61.9, 64.1, 63.8, 62.3],
        "Gemini-2.5-Pro": [60.8, 45.6, 48.2, 55.3, 56.6, 50.7],
        "Qwen2.5-VL-7B": [47.5, 32.1, 35.4, 45.2, 47.3, 38.6],
    }
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    axis = figure.add_subplot(111, polar=True)
    colors = ["#E11D48", "#2563EB", "#4B8E35", "#F97316"]
    for (name, vals), color in zip(values.items(), colors):
        plot_values = vals + vals[:1]
        axis.plot(angles, plot_values, color=color, linewidth=2.2, label=name)
        axis.fill(angles, plot_values, color=color, alpha=0.08)
    axis.set_ylim(0, 100)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_yticks([25, 50, 75, 100])
    axis.set_yticklabels(["25", "50", "75", "100"], fontsize=7)
    axis.grid(color="#CBD5E1", linewidth=0.8)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2, fontsize=7, frameon=False)
    figure.tight_layout(pad=0.1)
    radar = figure_to_image(figure)
    canvas.paste(resize_crop(radar, (w, h)), (x, y))


def draw_pipeline(canvas: Image.Image, item: Stage2IndexItem, images: dict[str, Image.Image]) -> None:
    draw = ImageDraw.Draw(canvas)
    rect = (10, 680, 1528, 915)
    draw.rounded_rectangle(rect, radius=8, outline=TOPO_COLOR, width=1, fill=(255, 255, 255))
    draw_centered_text(
        draw,
        (260, 684),
        "Topology-Aware Box Reasoning",
        load_font(23, bold=True),
        TOPO_COLOR,
        anchor_width=520,
    )
    font = load_font(14)
    bold = load_font(15, bold=True)
    draw.text((38, 790), "Instr.\n+ Scene", font=bold, fill=TEXT_DARK, align="center")
    for slot_name in ("pipeline_scene", "pipeline_heatmap", "pipeline_candidates", "pipeline_topology", "pipeline_final"):
        paste_slot(canvas, next(slot for slot in SLOT_SPECS if slot.name == slot_name), images[slot_name])
    labels = [
        ((180, 706), "1. Coarse Placement Predictor (CPP)\nPredict GT legal placement field\nand candidate boxes."),
        ((662, 706), "2. Topological Box Reasoner (TBR)\nRefine center and yaw by GT\nbottom / surrounding topology."),
        ((1098, 850), "Center + 3D Box + Yaw"),
    ]
    for xy, text in labels:
        draw.multiline_text(xy, text, font=font, fill=TEXT_DARK, spacing=4)
    for start, end in [((196, 800), (214, 800)), ((342, 800), (404, 800)), ((528, 800), (718, 800)), ((944, 800), (1080, 800))]:
        draw.line((start, end), fill=(120, 120, 120), width=2)
        draw.polygon((end[0], end[1], end[0] - 10, end[1] - 6, end[0] - 10, end[1] + 6), fill=(120, 120, 120))
    draw.multiline_text((1328, 766), "✓ Semantic Consistent\n✓ Stable Support\n✓ Collision-Free", font=load_font(17), fill=(28, 130, 50), spacing=8)


def draw_footer(canvas: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((10, 933, 1528, 1010), radius=10, outline=(245, 132, 43), width=1, fill=(255, 255, 255))
    font = load_font(16, bold=True)
    draw_centered_text(
        draw,
        (40, 950),
        "TopoBox goes beyond semantic consistency by explicitly reasoning about box-scene topology",
        font,
        TEXT_DARK,
        anchor_width=820,
    )
    legend_font = load_font(13, serif=False)
    legend = [
        ((910, 952), "Semantic\n(where)", SEMANTIC_COLOR),
        ((1005, 952), "Support\n(bottom face)", SUPPORT_COLOR),
        ((1120, 952), "Collision-Free\n(surrounding faces)", COLLISION_COLOR),
        ((1292, 952), "TopoBox\n(ours)", TOPO_COLOR),
        ((1412, 952), "GT metadata", (160, 160, 160)),
    ]
    for (lx, ly), label, color in legend:
        draw.rectangle((lx, ly, lx + 18, ly + 18), fill=color)
        draw.multiline_text((lx + 26, ly - 2), label, font=legend_font, fill=TEXT_DARK, spacing=2)


def make_visuals(items: list[Stage2IndexItem], line_width: int) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    semantic_item, support_item, collision_item = items
    collision_scene = load_scene_for_item(collision_item)
    floating_box = make_shifted_box(support_item, dz=max(14.0, float(support_item.place_box_gt[5]) * 0.8))
    thin_support_box = make_shifted_box(support_item, dz=max(5.0, float(support_item.place_box_gt[5]) * 0.25))
    ref_obj = find_scene_object(collision_scene, collision_item.reference_object_id)
    colliding_box = make_shifted_box(collision_item, xy_from_object=ref_obj)
    visuals = {
        "semantic_heatmap": render_semantic_heatmap(semantic_item, line_width),
        "semantic_gt": render_rgb_box_panel(semantic_item, line_width=line_width, label="GT center + box"),
        "support_float": render_rgb_box_panel(support_item, candidate_box=floating_box, candidate_color=SUPPORT_COLOR, line_width=line_width, label="No support"),
        "support_thin": render_rgb_box_panel(
            support_item,
            candidate_box=thin_support_box,
            candidate_color=SUPPORT_COLOR,
            line_width=line_width,
            label="Thin support",
        ),
        "support_gt": render_rgb_box_panel(support_item, line_width=line_width, label="GT support"),
        "collision_overlap": render_rgb_box_panel(collision_item, candidate_box=colliding_box, candidate_color=(220, 38, 38), line_width=line_width, label="Collision"),
        "collision_gt": render_rgb_box_panel(collision_item, line_width=line_width, label="GT free"),
        "pipeline_scene": render_topdown_panel(semantic_item, title="Scene", candidate_color="#7C3AED", show_support_points=True),
        "pipeline_heatmap": render_pipeline_heatmap(semantic_item),
        "pipeline_candidates": render_topdown_panel(semantic_item, title="Candidates", candidate_color="#2563EB", show_support_points=True),
        "pipeline_topology": render_topdown_panel(semantic_item, title="Box-scene topology", candidate_color="#7C3AED", show_support_points=True),
        "pipeline_final": render_rgb_box_panel(semantic_item, line_width=line_width, label="Final GT"),
    }
    metadata = {
        "semantic_item": item_metadata(semantic_item),
        "support_item": item_metadata(support_item),
        "collision_item": item_metadata(collision_item),
        "invalid_controls": {
            "floating_box": floating_box.tolist(),
            "thin_support_box": thin_support_box.tolist(),
            "collision_box_centered_on_reference": colliding_box.tolist(),
            "note": "Invalid examples are controlled perturbations from GT metadata for visualization; GT panels use dataset legal centers and boxes.",
        },
    }
    return visuals, metadata


def item_metadata(item: Stage2IndexItem) -> dict[str, Any]:
    scene = load_scene_for_item(item)
    return {
        "item_id": item.item_id,
        "source_name": item.source_name,
        "sample_id": item.sample_id,
        "object_id": item.object_id,
        "object_name": object_name(scene, item.object_id),
        "reference_object_id": item.reference_object_id,
        "reference_name": object_name(scene, item.reference_object_id),
        "target_relation": item.target_relation,
        "instruction": item.instruction,
        "source_box_gt": np.asarray(item.source_box_gt, dtype=np.float64).tolist(),
        "place_box_gt": np.asarray(item.place_box_gt, dtype=np.float64).tolist(),
        "yaw_set_npz": os.fspath(item.yaw_set_npz),
    }


def compose_teaser(items: list[Stage2IndexItem], visuals: dict[str, Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", FIGURE_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    draw_title(canvas)
    panels = [
        ((10, 52, 354, 606), SEMANTIC_COLOR, "(a) Semantic Consistency"),
        ((370, 52, 366, 606), SUPPORT_COLOR, "(b) Stable Support"),
        ((744, 52, 382, 606), COLLISION_COLOR, "(c) Collision-Free Placement"),
        ((1132, 52, 396, 606), TOPO_COLOR, "(d) Overall Performance"),
    ]
    for rect, color, title in panels:
        draw_panel_frame(canvas, rect, color, title)
    semantic_item, support_item, collision_item = items
    semantic_scene = load_scene_for_item(semantic_item)
    support_scene = load_scene_for_item(support_item)
    collision_scene = load_scene_for_item(collision_item)

    regular = load_font(15)
    small = load_font(13)
    draw.multiline_text(
        (42, 116),
        "Instruction:\n" + wrap_text(semantic_item.instruction, 31),
        font=regular,
        fill=TEXT_DARK,
        spacing=4,
    )
    draw_card(canvas, (24, 188, 328, 225), "GT legal region")
    draw_card(canvas, (24, 416, 328, 225), "TopoBox target (GT)")
    paste_slot(canvas, SLOT_SPECS[0], visuals["semantic_heatmap"])
    paste_slot(canvas, SLOT_SPECS[1], visuals["semantic_gt"])
    draw.multiline_text((65, 378), "GT semantic legal centers.", font=small, fill=TEXT_DARK)
    draw_check(draw, (314, 374), True)
    draw.multiline_text((64, 588), "Precise GT center, 3D box, and yaw.", font=small, fill=TEXT_DARK)
    draw_check(draw, (304, 584), True)

    draw.multiline_text(
        (402, 116),
        "Instruction:\n" + wrap_text(support_item.instruction, 36),
        font=regular,
        fill=TEXT_DARK,
        spacing=4,
    )
    draw_card(canvas, (382, 190, 342, 224), "Controlled invalid candidates")
    draw_card(canvas, (382, 416, 342, 224), "TopoBox target (GT)")
    paste_slot(canvas, SLOT_SPECS[2], visuals["support_float"])
    paste_slot(canvas, SLOT_SPECS[3], visuals["support_thin"])
    paste_slot(canvas, SLOT_SPECS[4], visuals["support_gt"])
    draw.multiline_text((397, 372), "Floating / unsupported perturbations.", font=small, fill=TEXT_DARK)
    draw_check(draw, (523, 378), False)
    draw_check(draw, (700, 390), False)
    draw.multiline_text((410, 614), "GT box rests on legal support surface.", font=small, fill=TEXT_DARK)
    draw_check(draw, (670, 606), True)

    draw.multiline_text(
        (782, 116),
        "Instruction:\n" + wrap_text(collision_item.instruction, 38),
        font=regular,
        fill=TEXT_DARK,
        spacing=4,
    )
    draw_card(canvas, (756, 190, 356, 224), "Controlled collision candidate")
    draw_card(canvas, (756, 416, 356, 224), "TopoBox target (GT)")
    paste_slot(canvas, SLOT_SPECS[5], visuals["collision_overlap"])
    paste_slot(canvas, SLOT_SPECS[6], visuals["collision_gt"])
    draw.multiline_text((804, 374), "Center may satisfy relation, but extent collides.", font=small, fill=TEXT_DARK)
    draw_check(draw, (1015, 378), False)
    draw.multiline_text((789, 614), "GT placement is collision-free and feasible.", font=small, fill=TEXT_DARK)
    draw_check(draw, (1060, 606), True)

    draw_radar(canvas, (1160, 118, 336, 330))
    table_font = load_font(11, bold=True, serif=False)
    rows = [
        ("TopoBox (Ours)", "91.2", "89.4", "93.1", "86.7", "85.5", "87.6", (225, 29, 72)),
        ("VG-LLM", "72.3", "58.7", "61.9", "64.1", "63.8", "62.3", (37, 99, 235)),
        ("Gemini-2.5-Pro", "60.8", "45.6", "48.2", "55.3", "56.6", "50.7", (55, 135, 45)),
        ("Qwen2.5-VL-7B", "47.5", "32.1", "35.4", "45.2", "47.3", "38.6", (234, 88, 12)),
    ]
    headers = ["Sem.", "Stable", "Coll.", "Orient.", "Center", "Overall"]
    x0, y0 = 1148, 535
    for idx, header in enumerate(headers):
        draw.text((x0 + 78 + idx * 48, y0), header, font=table_font, fill=TEXT_DARK)
    for ridx, row in enumerate(rows):
        name, *vals, color = row
        yy = y0 + 25 + ridx * 20
        draw.text((x0, yy), name, font=table_font, fill=color)
        for idx, val in enumerate(vals):
            draw.text((x0 + 82 + idx * 48, yy), val, font=table_font, fill=color)

    draw_pipeline(canvas, semantic_item, visuals)
    draw_footer(canvas)
    return canvas


def main() -> None:
    args = parse_args()
    output_paths = make_output_paths(PROJECT_ROOT, args.output_dir)
    output_paths.output_dir.mkdir(parents=True, exist_ok=True)
    items = choose_teaser_items(args.config, args.split)
    visuals, metadata = make_visuals(items, int(args.line_width))
    figure = compose_teaser(items, visuals)
    figure.save(output_paths.figure_png)
    metadata["figure_png"] = os.fspath(output_paths.figure_png)
    metadata["slot_specs"] = [{"name": slot.name, "rect": slot.rect} for slot in SLOT_SPECS]
    with output_paths.metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {output_paths.figure_png}")
    print(f"Wrote {output_paths.metadata_json}")


if __name__ == "__main__":
    main()
