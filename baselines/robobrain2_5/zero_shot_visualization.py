"""Shared geometry and rendering utilities for the RoboBrain2.5 baseline."""

from __future__ import annotations

import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
POINT_COLOR = (37, 99, 235)
SOURCE_COLOR = (8, 145, 178)
PLACED_COLOR = (220, 38, 38)
GT_COLOR = (22, 163, 74)
SCENE_COLOR = (156, 163, 175)


@dataclass(frozen=True)
class Camera:
    """Canonical camera parameters; all distances are in the sample unit (cm)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    e_c2w: np.ndarray

    @property
    def e_w2c(self) -> np.ndarray:
        return np.linalg.inv(self.e_c2w)


def parse_point_answer(answer: str) -> tuple[tuple[int, int] | None, str]:
    """Parse RoboBrain's documented ``(x, y)`` normalized pointing format."""
    import re

    matches = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", str(answer))
    if not matches:
        return None, "parse_failed"
    x, y = (int(value) for value in matches[0])
    return (x, y), "ok"


def normalized_to_pixel(point: tuple[int, int], width: int, height: int) -> tuple[tuple[int, int], bool]:
    """Mirror the official renderer's 0--1000 coordinate conversion and clamping."""
    raw_x = int(round(point[0] / 1000.0 * width))
    raw_y = int(round(point[1] / 1000.0 * height))
    pixel = (max(0, min(width - 1, raw_x)), max(0, min(height - 1, raw_y)))
    return pixel, pixel != (raw_x, raw_y)


def local_depth_cm(depth: np.ndarray, pixel: tuple[int, int], radius: int = 2) -> float | None:
    """Return the median positive finite depth in a clipped square pixel neighborhood."""
    x, y = pixel
    height, width = depth.shape[:2]
    window = np.asarray(
        depth[max(0, y - radius) : min(height, y + radius + 1), max(0, x - radius) : min(width, x + radius + 1)],
        dtype=np.float64,
    )
    valid = window[np.isfinite(window) & (window > 0.0)]
    return float(np.median(valid)) if len(valid) else None


def backproject_world(pixel: tuple[int, int], depth_cm: float, camera: Camera) -> np.ndarray:
    """Backproject a depth pixel and transform it into canonical world coordinates."""
    x, y = pixel
    z = float(depth_cm)
    point_cam = np.array(
        [(x - camera.cx) * z / camera.fx, (y - camera.cy) * z / camera.fy, z, 1.0],
        dtype=np.float64,
    )
    return (camera.e_c2w @ point_cam)[:3]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def aabb_corners(bounds: np.ndarray) -> np.ndarray:
    """Create eight corners from ``[xmin, ymin, zmin, xmax, ymax, zmax]``."""
    bounds = np.asarray(bounds, dtype=np.float64)
    low, high = bounds[:3], bounds[3:]
    return np.asarray(
        [[x, y, z] for z in (low[2], high[2]) for y in (low[1], high[1]) for x in (low[0], high[0])],
        dtype=np.float64,
    )


def source_dimensions_and_yaw(source: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Use oracle source dimensions and its observed horizontal orientation for rendering only."""
    bounds = np.asarray(source["bbox3d_canonical"], dtype=np.float64)
    pose = np.asarray(source["pose_world"], dtype=np.float64)
    axes = pose[:3, :3]
    norms = np.linalg.norm(axes, axis=0)
    if np.any(norms < 1e-12):
        raise ValueError("Source pose contains a degenerate axis")
    axes = axes / norms[None, :]
    dimensions = np.maximum(bounds[3:] - bounds[:3], 1e-4)
    up_axis = int(np.argmax(np.abs(axes.T @ np.array([0.0, 0.0, 1.0]))))
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    size = np.array(
        [dimensions[horizontal_axes[0]], dimensions[horizontal_axes[1]], dimensions[up_axis]], dtype=np.float64
    )
    heading = axes[:, horizontal_axes[0]]
    yaw = math.atan2(float(heading[1]), float(heading[0]))
    return size, yaw


def upright_box_corners(center: np.ndarray, dimensions: np.ndarray, yaw: float) -> np.ndarray:
    """Return a canonical-Z-up box's corners."""
    center = np.asarray(center, dtype=np.float64)
    dimensions = np.maximum(np.asarray(dimensions, dtype=np.float64), 1e-4)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    axes = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    half_axes = axes * (dimensions * 0.5)[None, :]
    return np.asarray(
        [center + half_axes @ np.array([sx, sy, sz], dtype=np.float64) for sz in (-1.0, 1.0) for sy in (-1.0, 1.0) for sx in (-1.0, 1.0)],
        dtype=np.float64,
    )


def project_world(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Project canonical-world points to the source RGB image."""
    points_cam = transform_points(points, camera.e_w2c)
    z = points_cam[:, 2]
    uv = np.empty((len(points_cam), 2), dtype=np.float64)
    uv[:, 0] = camera.fx * points_cam[:, 0] / z + camera.cx
    uv[:, 1] = camera.fy * points_cam[:, 1] / z + camera.cy
    return uv, z


def draw_projected_corners(
    draw: ImageDraw.ImageDraw,
    corners: np.ndarray,
    camera: Camera,
    color: tuple[int, int, int],
    label: str,
    width: int = 3,
) -> None:
    """Draw a world-space box while skipping edges behind the camera."""
    uv, depth = project_world(corners, camera)
    visible = depth > 0.0
    for start, end in BOX_EDGES:
        if visible[start] and visible[end]:
            draw.line([tuple(uv[start]), tuple(uv[end])], fill=color, width=width)
    if np.any(visible):
        xy = uv[visible].mean(axis=0)
        font = ImageFont.load_default()
        bbox = draw.textbbox(tuple(xy), label, font=font)
        draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=(0, 0, 0))
        draw.text(tuple(xy), label, fill=color, font=font)


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Compute a deterministic 2D monotone-chain convex hull without extra dependencies."""
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) <= 2:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return float(np.cross(first - origin, second - origin))

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def render_topdown(
    scene_objects: list[dict[str, Any]],
    source_id: str,
    candidate_corners: np.ndarray | None,
    gt_corners: np.ndarray | None,
    size: int = 640,
) -> Image.Image:
    """Render raw scene objects, the moved source, and the visual-only candidate box."""
    polygons: dict[str, np.ndarray] = {}
    for obj in scene_objects:
        corners = transform_points(np.asarray(aabb_corners(obj["bbox3d_canonical"])), np.asarray(obj["pose_world"]))
        polygons[str(obj["obj_id"])] = convex_hull_xy(corners[:, :2])
    if candidate_corners is not None:
        polygons["__candidate__"] = convex_hull_xy(candidate_corners[:, :2])
    if gt_corners is not None:
        polygons["__gt__"] = convex_hull_xy(gt_corners[:, :2])
    all_points = np.vstack([poly for poly in polygons.values() if len(poly)])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    margin = max(float((high - low).max()) * 0.08, 1.0)
    low, high = low - margin, high + margin
    span = np.maximum(high - low, 1e-6)

    def to_pixel(poly: np.ndarray) -> list[tuple[float, float]]:
        x = (poly[:, 0] - low[0]) / span[0] * (size - 1)
        y = (1.0 - (poly[:, 1] - low[1]) / span[1]) * (size - 1)
        return [(float(px), float(py)) for px, py in zip(x, y)]

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    for obj_id, poly in polygons.items():
        if obj_id in {"__candidate__", "__gt__"}:
            continue
        fill, outline = ((8, 145, 178, 70), (8, 145, 178, 255)) if obj_id == source_id else ((229, 231, 235, 90), (156, 163, 175, 200))
        points = to_pixel(poly)
        draw.polygon(points, fill=fill, outline=outline)
        center = np.asarray(points, dtype=np.float64).mean(axis=0)
        draw.text((float(center[0] + 3), float(center[1] + 3)), obj_id, fill=outline, font=font)
    if candidate_corners is not None:
        points = to_pixel(polygons["__candidate__"])
        draw.polygon(points, fill=(220, 38, 38, 80), outline=(185, 28, 28, 255), width=3)
        center = np.asarray(points, dtype=np.float64).mean(axis=0)
        draw.text((float(center[0] + 3), float(center[1] + 3)), "RoboBrain point + source box", fill=(185, 28, 28, 255), font=font)
    if gt_corners is not None:
        points = to_pixel(polygons["__gt__"])
        draw.polygon(points, fill=(22, 163, 74, 70), outline=(*GT_COLOR, 255), width=3)
        center = np.asarray(points, dtype=np.float64).mean(axis=0)
        draw.text((float(center[0] + 3), float(center[1] + 3)), "GT placement", fill=(*GT_COLOR, 255), font=font)
    return image


def render_composite(
    rgb: np.ndarray,
    camera: Camera,
    scene_objects: list[dict[str, Any]],
    source_id: str,
    pixel: tuple[int, int] | None,
    candidate_corners: np.ndarray | None,
    gt_bottom_center: np.ndarray,
    gt_corners: np.ndarray,
    answer: str,
    status: str,
    max_rgb_width: int = 960,
) -> Image.Image:
    """Create one compact RGB-plus-top-down qualitative visualization."""
    rgb_image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(rgb_image)
    source = next(obj for obj in scene_objects if str(obj["obj_id"]) == source_id)
    source_corners = transform_points(aabb_corners(np.asarray(source["bbox3d_canonical"])), np.asarray(source["pose_world"]))
    draw_projected_corners(draw, source_corners, camera, SOURCE_COLOR, "source")
    if pixel is not None:
        x, y = pixel
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=POINT_COLOR, outline=(255, 255, 255), width=2)
    if candidate_corners is not None:
        draw_projected_corners(draw, candidate_corners, camera, PLACED_COLOR, "RoboBrain placed source")
    gt_uv, gt_depth = project_world(np.asarray([gt_bottom_center]), camera)
    if gt_depth[0] > 0.0:
        x, y = gt_uv[0]
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=GT_COLOR, outline=(255, 255, 255), width=2)
    draw_projected_corners(draw, gt_corners, camera, GT_COLOR, "GT placement")
    if rgb_image.width > max_rgb_width:
        scale = max_rgb_width / rgb_image.width
        rgb_image = rgb_image.resize((max_rgb_width, max(1, round(rgb_image.height * scale))), Image.Resampling.LANCZOS)

    topdown = render_topdown(scene_objects, source_id, candidate_corners, gt_corners, size=max(480, rgb_image.height))
    panel_height = max(rgb_image.height, topdown.height)
    if rgb_image.height != panel_height:
        rgb_image = rgb_image.resize((rgb_image.width, panel_height), Image.Resampling.LANCZOS)
    if topdown.height != panel_height:
        topdown = topdown.resize((topdown.width, panel_height), Image.Resampling.LANCZOS)
    header_height = 52
    composite = Image.new("RGB", (rgb_image.width + topdown.width, header_height + panel_height), "white")
    composite.paste(rgb_image, (0, header_height))
    composite.paste(topdown, (rgb_image.width, header_height))
    header = ImageDraw.Draw(composite)
    font = ImageFont.load_default()
    header.text((8, 7), f"status: {status}", fill=(17, 24, 39), font=font)
    safe_answer = " ".join(str(answer).split())[:260]
    header.text((8, 26), f"RoboBrain answer: {safe_answer}", fill=(55, 65, 81), font=font)
    return composite


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one durable result row so a long GPU job can be resumed."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_web_gallery(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a self-contained paginated gallery that works by opening the HTML file directly."""
    display_rows = [
        {
            "item_id": row["item_id"],
            "source": row["source_name"],
            "instruction": row["instruction"],
            "status": row["status"],
            "image": row.get("visualization_path", ""),
        }
        for row in rows
    ]
    payload = json.dumps(display_rows, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html><meta charset=\"utf-8\"><title>RoboBrain2.5 zero-shot test gallery</title>
<style>body{font-family:system-ui;margin:20px;background:#f8fafc;color:#111827}.controls{position:sticky;top:0;background:#f8fafc;padding:10px 0}.card{background:white;margin:16px 0;padding:12px;border-radius:8px;box-shadow:0 1px 3px #0002}.card img{max-width:100%;height:auto}.meta{font-size:14px;white-space:pre-wrap}button{margin-right:8px}</style>
<h1>RoboBrain2.5 zero-shot point visualization</h1><div class=\"controls\"><button id=\"prev\">Previous</button><button id=\"next\">Next</button><span id=\"page\"></span></div><main id=\"rows\"></main>
<script>const rows=""" + payload + """;let page=0;const size=24;const root=document.querySelector('#rows');function render(){const start=page*size;const selected=rows.slice(start,start+size);root.replaceChildren(...selected.map(r=>{const c=document.createElement('article');c.className='card';const m=document.createElement('div');m.className='meta';m.textContent=`${r.item_id} | ${r.source} | ${r.status}\\n${r.instruction}`;c.append(m);if(r.image){const i=document.createElement('img');i.loading='lazy';i.src='../'+r.image;i.alt=r.item_id;c.append(i)}return c}));document.querySelector('#page').textContent=`${start+1}-${Math.min(start+size,rows.length)} / ${rows.length}`};document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);render()};document.querySelector('#next').onclick=()=>{page=Math.min(Math.ceil(rows.length/size)-1,page+1);render()};render()</script>"""
    output_path.write_text(page, encoding="utf-8")


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Summarize only run health, not physical feasibility or task performance."""
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    statuses = Counter()
    for row in rows:
        statuses[str(row["status"])] += 1
        by_source[str(row["source_name"])][str(row["status"])] += 1
    payload = {
        "purpose": "zero-shot qualitative visualization only; no physical-feasibility metrics are computed",
        "item_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "by_source": {key: dict(sorted(value.items())) for key, value in sorted(by_source.items())},
        "oracle_source_geometry_for_visualization": True,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
