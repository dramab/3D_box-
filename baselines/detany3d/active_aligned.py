"""Pure data and geometry helpers for the DetAny3D active_aligned baseline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml


SOURCE_PROMPT_PATTERN = re.compile(r"^\s*move\s+(.+?)\s+located\s+at\b", re.IGNORECASE)


@dataclass(frozen=True)
class ActiveAlignedRecord:
    """One unique source object from the fixed active_aligned split."""

    item_id: str
    source_name: str
    sample_id: str
    object_id: str
    class_name: str
    instruction: str
    prompt: str
    rgb_path: Path
    camera_k: np.ndarray
    camera_e_w2c: np.ndarray
    canonical_aabb_object_cm: np.ndarray
    pose_world_cm: np.ndarray


def source_prompt_from_instruction(instruction: str) -> str:
    """Convert a placement instruction to the exact source noun phrase prompt."""
    match = SOURCE_PROMPT_PATTERN.search(str(instruction))
    if match is None:
        raise ValueError(f"Cannot extract source object from instruction: {instruction}")
    phrase = " ".join(match.group(1).strip().split()).lower()
    if not phrase:
        raise ValueError(f"Empty source object in instruction: {instruction}")
    return phrase.rstrip(".") + "."


def load_active_aligned_records(
    project_root: Path,
    split: str = "test",
    stage1_config: Path = Path("configs/lc_bgplacenet_stage1.yaml"),
) -> List[ActiveAlignedRecord]:
    """Load and deduplicate source objects without importing the training stack."""
    project_root = Path(project_root).resolve()
    config_path = _resolve(project_root, stage1_config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    split_dir = _resolve(project_root, Path(config["data"]["split_dir"]))
    split_path = split_dir / f"{split}.json"
    with split_path.open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)
    if split_payload.get("split") != split:
        raise ValueError(f"Split file declares {split_payload.get('split')}, expected {split}")

    sources = {
        str(source["name"]): {
            "dataset_dir": _resolve(project_root, Path(source["dataset_dir"])),
            "free_bbox_dir": _resolve(project_root, Path(source["free_bbox_dir"])),
            "labels_path": _resolve(project_root, Path(source["labels_path"])),
        }
        for source in config["data"]["sources"]
    }
    labels_by_source = {
        name: _read_json(paths["labels_path"]) for name, paths in sources.items()
    }

    records: List[ActiveAlignedRecord] = []
    seen = set()
    for split_item in split_payload.get("items", []):
        source_name = str(split_item["source_name"])
        sample_id = str(split_item["sample_id"])
        object_id = str(split_item["object_id"])
        key = (source_name, sample_id, object_id)
        if key in seen:
            continue
        seen.add(key)

        source = sources[source_name]
        label_index = int(split_item["label_index"])
        label_record = labels_by_source[source_name][label_index]
        instruction = str(split_item["instruction"])
        if str(label_record["sample_id"]) != sample_id or str(label_record["object_id"]) != object_id:
            raise ValueError(f"Label index mismatch for {source_name}/{sample_id}/{object_id}")

        sample_path = source["dataset_dir"] / "samples" / f"{sample_id}.json"
        sample = _read_json(sample_path)
        if sample.get("unit") != "cm":
            raise ValueError(f"Expected canonical centimetres in {sample_path}")
        camera = sample["camera"]
        camera_k = np.asarray(
            [
                [float(camera["fx"]), 0.0, float(camera["cx"])],
                [0.0, float(camera["fy"]), float(camera["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        camera_e_w2c = np.linalg.inv(np.asarray(camera["E_c2w"], dtype=np.float64)).astype(
            np.float32
        )

        placement_path = source["free_bbox_dir"] / "placements" / f"{sample_id}__placements.json"
        placement = _read_json(placement_path)
        object_record = next(
            (
                obj
                for obj in placement.get("objects", [])
                if str(obj.get("object_id")) == object_id
            ),
            None,
        )
        if object_record is None:
            raise ValueError(f"Object {object_id} is missing from {placement_path}")

        records.append(
            ActiveAlignedRecord(
                item_id=str(split_item["item_id"]),
                source_name=source_name,
                sample_id=sample_id,
                object_id=object_id,
                class_name=str(label_record["class_name"]),
                instruction=instruction,
                prompt=source_prompt_from_instruction(instruction),
                rgb_path=source["dataset_dir"] / str(sample["rgb_path"]),
                camera_k=camera_k,
                camera_e_w2c=camera_e_w2c,
                canonical_aabb_object_cm=np.asarray(
                    object_record["canonical_aabb_object"], dtype=np.float32
                ),
                pose_world_cm=np.asarray(object_record["original_pose_world"], dtype=np.float32),
            )
        )
    return records


def select_balanced_records(
    records: Sequence[ActiveAlignedRecord], max_samples: int, seed: int
) -> List[ActiveAlignedRecord]:
    """Select a deterministic, approximately source-balanced qualitative subset."""
    if max_samples <= 0:
        return []
    grouped: Dict[str, List[ActiveAlignedRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_name, []).append(record)
    rng = np.random.default_rng(int(seed))
    for values in grouped.values():
        rng.shuffle(values)

    selected: List[ActiveAlignedRecord] = []
    source_names = sorted(grouped)
    offset = 0
    while len(selected) < min(max_samples, len(records)):
        added = False
        for source_name in source_names:
            values = grouped[source_name]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        offset += 1
    return selected


def obb_corners_camera_m(record: ActiveAlignedRecord) -> np.ndarray:
    """Return the GT source OBB corners in camera coordinates and metres."""
    bbox = record.canonical_aabb_object_cm
    lower, upper = bbox[:3], bbox[3:]
    corners_object = np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ],
        dtype=np.float32,
    )
    corners_world_cm = transform_points(corners_object, record.pose_world_cm)
    return transform_points(corners_world_cm, record.camera_e_w2c) / 100.0


def gt_center_camera_m(record: ActiveAlignedRecord) -> np.ndarray:
    """Return the GT source OBB center in camera coordinates and metres."""
    bbox = record.canonical_aabb_object_cm
    center_object = ((bbox[:3] + bbox[3:]) * 0.5)[None, :]
    center_world_cm = transform_points(center_object, record.pose_world_cm)
    return (transform_points(center_world_cm, record.camera_e_w2c)[0] / 100.0).astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to Nx3 row-vector points."""
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(points_camera: np.ndarray, camera_k: np.ndarray) -> np.ndarray:
    """Project positive-depth camera points to pixels."""
    points_camera = np.asarray(points_camera, dtype=np.float32)
    pixels_h = points_camera @ np.asarray(camera_k, dtype=np.float32).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def processed_pixels_to_original(
    points: np.ndarray,
    original_hw: Tuple[int, int],
    resized_hw: Tuple[int, int],
    crop_xy: Tuple[int, int],
) -> np.ndarray:
    """Map DetAny3D processed-image pixels back to the original RGB frame."""
    original_h, original_w = original_hw
    resized_h, resized_w = resized_hw
    crop_x, crop_y = crop_xy
    mapped = np.asarray(points, dtype=np.float32).copy()
    mapped[:, 0] = (mapped[:, 0] + crop_x) * (original_w / float(resized_w))
    mapped[:, 1] = (mapped[:, 1] + crop_y) * (original_h / float(resized_h))
    return mapped


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
