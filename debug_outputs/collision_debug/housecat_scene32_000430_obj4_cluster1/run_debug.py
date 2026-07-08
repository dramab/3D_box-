import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/limengfei/xingqunqi/3D_box-")
DATASET_ROOT = ROOT / "data/housecat"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
sys.path.insert(0, str(ROOT))

from src.annotation.auto_label import convex_hull_xy, polygon_area_xy, polygon_intersection_area_xy
from src.annotation.free_bbox.geometry import get_bbox_corners, transform_points
from src.datasets.canonical import load_sample_record


ITEM_ID = "housecat__label_009672__32fe6df71c86cb19"
SAMPLE_ID = "housecat__scene32__000430"
SHOE_ID = "obj_6_shoe_crocs_yellow_sandal_right"
OUT = ROOT / "debug_outputs/collision_debug/housecat_scene32_000430_obj4_cluster1"


def load_target_rows():
    preds = json.loads((ROOT / "outputs/lc_bgplacenet_stage2/inference_stage2_test/predictions.json").read_text())
    pred = next(row for row in preds if row["item_id"] == ITEM_ID)
    metrics_path = ROOT / "outputs/lc_bgplacenet_stage2/benchmark_stage2_test/per_sample_metrics.jsonl"
    metric = None
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        if row["item_id"] == ITEM_ID:
            metric = row
            break
    if metric is None:
        raise RuntimeError(f"metric row not found: {ITEM_ID}")
    return pred, metric


def place_box_to_bbox_transform(place_box):
    box = np.asarray(place_box, dtype=np.float64)
    dims = np.maximum(box[3:6], 1e-4)
    yaw = float(box[6])
    c, s = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform[:3, 3] = box[:3]
    return np.concatenate([-dims * 0.5, dims * 0.5]), transform


def obb_from_bbox_transform(object_id, bbox3d, transform):
    bbox = np.asarray(bbox3d, dtype=np.float64)
    pose = np.asarray(transform, dtype=np.float64)
    local_center = (bbox[:3] + bbox[3:]) * 0.5
    center = pose[:3, :3] @ local_center + pose[:3, 3]
    half_extents = np.maximum((bbox[3:] - bbox[:3]) * 0.5, 1e-4)
    axes = pose[:3, :3]
    axis_scales = np.linalg.norm(axes, axis=0)
    if np.any(axis_scales < 1e-12):
        raise ValueError(f"degenerate OBB transform: {object_id}")
    axes = axes / axis_scales[None, :]
    return {
        "object_id": object_id,
        "center": center.astype(np.float64),
        "axes": axes.astype(np.float64),
        "half_extents": (half_extents * axis_scales).astype(np.float64),
    }


def corners_from_obb(obb):
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                signs = np.array([xi, yi, zi], dtype=np.float64) * 2.0 - 1.0
                corners.append(obb["center"] + obb["axes"] @ (obb["half_extents"] * signs))
    return np.asarray(corners, dtype=np.float64)


def project(points_world, K, E_w2c):
    pts = transform_points(np.asarray(points_world, dtype=np.float64), E_w2c)
    z = pts[:, 2]
    uv = np.empty((len(pts), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * pts[:, 0] / z + K[0, 2]
    uv[:, 1] = K[1, 1] * pts[:, 1] / z + K[1, 2]
    return uv, z


def sat_details(a, b):
    a_axes = np.asarray(a["axes"], dtype=np.float64)
    b_axes = np.asarray(b["axes"], dtype=np.float64)
    axes = []
    labels = []
    for i in range(3):
        axes.append(a_axes[:, i])
        labels.append(f"pred_axis_{i}")
    for i in range(3):
        axes.append(b_axes[:, i])
        labels.append(f"shoe_axis_{i}")
    for i in range(3):
        for j in range(3):
            axes.append(np.cross(a_axes[:, i], b_axes[:, j]))
            labels.append(f"cross_{i}_{j}")

    rows = []
    delta = np.asarray(b["center"]) - np.asarray(a["center"])
    for label, axis in zip(labels, axes):
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-8:
            continue
        axis = axis / norm
        center_distance = abs(float(axis @ delta))
        radius_a = float(np.abs(axis @ a_axes) @ np.asarray(a["half_extents"]))
        radius_b = float(np.abs(axis @ b_axes) @ np.asarray(b["half_extents"]))
        overlap = radius_a + radius_b - center_distance
        rows.append(
            {
                "axis": label,
                "center_distance_cm": center_distance,
                "radius_sum_cm": radius_a + radius_b,
                "overlap_cm": overlap,
                "axis_vector": axis.tolist(),
            }
        )
    return sorted(rows, key=lambda row: row["overlap_cm"])


def draw_projected_box(draw, corners, K, E_w2c, color, label):
    uv, z = project(corners, K, E_w2c)
    edges = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7))
    for i, j in edges:
        if z[i] > 0.0 and z[j] > 0.0:
            draw.line([(float(uv[i, 0]), float(uv[i, 1])), (float(uv[j, 0]), float(uv[j, 1]))], fill=color, width=4)
    center = uv[z > 0.0].mean(axis=0)
    font = ImageFont.load_default()
    draw.rectangle([float(center[0]), float(center[1]), float(center[0] + 170), float(center[1] + 14)], fill=(0, 0, 0))
    draw.text((float(center[0] + 2), float(center[1] + 1)), label, fill=color, font=font)


def save_topdown(pred_poly, shoe_poly, xy_intersection, z_overlap, sat_min):
    size = 720
    all_xy = np.vstack([pred_poly, shoe_poly])
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    pad = max(float(span.max()) * 0.18, 2.0)
    min_xy -= pad
    max_xy += pad
    span = np.maximum(max_xy - min_xy, 1e-6)

    def to_px(poly):
        x = (poly[:, 0] - min_xy[0]) / span[0] * (size - 1)
        y = (1.0 - (poly[:, 1] - min_xy[1]) / span[1]) * (size - 1)
        return [(float(a), float(b)) for a, b in zip(x, y)]

    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    draw.polygon(to_px(shoe_poly), fill=(245, 158, 11, 90), outline=(180, 83, 9, 255))
    draw.polygon(to_px(pred_poly), fill=(220, 38, 38, 90), outline=(185, 28, 28, 255))
    draw.text(
        (12, 12),
        f"XY intersection={xy_intersection:.3f} cm^2, Z overlap={z_overlap:.3f} cm, SAT min overlap={sat_min:.3f} cm",
        fill=(17, 24, 39),
        font=font,
    )
    draw.text((12, 32), "No footprint dilation/margin in this debug image", fill=(17, 24, 39), font=font)
    image.save(OUT / "topdown_pred_vs_shoe_exact_footprints.png")


def main():
    pred, metric = load_target_rows()
    record = load_sample_record(DATASET_ROOT / f"samples/{SAMPLE_ID}.json")
    placements = json.loads((ROOT / f"outputs/free_bbox_housecat/placements/{SAMPLE_ID}__placements.json").read_text())
    shoe = {obj["object_id"]: obj for obj in placements["objects"]}[SHOE_ID]

    camera = record["camera"]
    K = np.array([[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
    E_c2w = np.asarray(camera["E_c2w"], dtype=np.float64)
    E_w2c = np.linalg.inv(E_c2w)

    pred_bbox, pred_transform = place_box_to_bbox_transform(pred["place_box"])
    pred_obb = obb_from_bbox_transform("prediction", pred_bbox, pred_transform)
    shoe_obb = obb_from_bbox_transform(SHOE_ID, shoe["canonical_aabb_object"], shoe["original_pose_world"])
    pred_corners = corners_from_obb(pred_obb)
    shoe_corners = corners_from_obb(shoe_obb)

    sat = sat_details(pred_obb, shoe_obb)
    pred_poly = convex_hull_xy(pred_corners)
    shoe_poly = convex_hull_xy(shoe_corners)
    xy_intersection = polygon_intersection_area_xy(pred_poly, shoe_poly)
    pred_area = polygon_area_xy(pred_poly)
    shoe_area = polygon_area_xy(shoe_poly)
    pred_z = [float(pred_corners[:, 2].min()), float(pred_corners[:, 2].max())]
    shoe_z = [float(shoe_corners[:, 2].min()), float(shoe_corners[:, 2].max())]
    z_overlap = min(pred_z[1], shoe_z[1]) - max(pred_z[0], shoe_z[0])
    pred_uv, pred_zcam = project(pred_corners, K, E_w2c)
    shoe_uv, shoe_zcam = project(shoe_corners, K, E_w2c)

    report = {
        "item_id": ITEM_ID,
        "metric_row": metric,
        "prediction_place_box_xyzdxdydzyaw": pred["place_box"],
        "collision_code": {
            "type": "3D OBB SAT",
            "eps_cm": 1e-6,
            "explicit_margin_or_padding_cm": 0.0,
            "source_file": "tools/benchmark_lc_bgplacenet_stage2.py",
        },
        "coordinate_normalization": record.get("preprocess", {}).get("coordinate_normalization"),
        "camera_projection": {
            "uses_sample_E_c2w_after_coordinate_normalization": True,
            "E_w2c_is_inverse_of_E_c2w": True,
            "pred_projected_uv_minmax": [pred_uv.min(axis=0).tolist(), pred_uv.max(axis=0).tolist()],
            "shoe_projected_uv_minmax": [shoe_uv.min(axis=0).tolist(), shoe_uv.max(axis=0).tolist()],
            "pred_zcam_minmax": [float(pred_zcam.min()), float(pred_zcam.max())],
            "shoe_zcam_minmax": [float(shoe_zcam.min()), float(shoe_zcam.max())],
        },
        "pred_obb": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pred_obb.items()},
        "shoe_obb": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in shoe_obb.items()},
        "xy_footprint": {
            "pred_area_cm2": float(pred_area),
            "shoe_area_cm2": float(shoe_area),
            "intersection_area_cm2": float(xy_intersection),
            "intersection_over_pred": float(xy_intersection / pred_area),
            "intersection_over_shoe": float(xy_intersection / shoe_area),
            "pred_polygon_xy": pred_poly.tolist(),
            "shoe_polygon_xy": shoe_poly.tolist(),
        },
        "z_aabb_overlap": {
            "pred_z_minmax_cm": pred_z,
            "shoe_z_minmax_cm": shoe_z,
            "overlap_cm": float(z_overlap),
        },
        "sat_min_overlap": sat[0],
        "sat_smallest_overlaps": sat[:8],
        "all_sat_axes_have_positive_overlap": bool(all(row["overlap_cm"] > 1e-6 for row in sat)),
    }
    (OUT / "collision_debug_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    image = Image.open(DATASET_ROOT / record["rgb_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw_projected_box(draw, shoe_corners, K, E_w2c, (245, 158, 11), "shoe OBB")
    draw_projected_box(draw, pred_corners, K, E_w2c, (220, 38, 38), "pred spoon box")
    image.save(OUT / "rgb_pred_vs_shoe_projection.png")
    save_topdown(pred_poly, shoe_poly, xy_intersection, z_overlap, sat[0]["overlap_cm"])

    print(
        json.dumps(
            {
                "out_dir": str(OUT),
                "metric_collision": metric["collision"],
                "metric_collision_ids": metric["collision_object_ids"],
                "xy_intersection_cm2": report["xy_footprint"]["intersection_area_cm2"],
                "z_overlap_cm": report["z_aabb_overlap"]["overlap_cm"],
                "sat_min_overlap_cm": report["sat_min_overlap"]["overlap_cm"],
                "sat_min_axis": report["sat_min_overlap"]["axis"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
