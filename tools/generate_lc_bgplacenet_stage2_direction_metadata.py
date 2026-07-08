#!/usr/bin/env python
"""
Generate language direction metadata for LC-BGPlaceNet Stage 2 benchmark.

使用示例:
    python tools/generate_lc_bgplacenet_stage2_direction_metadata.py \
        --config configs/lc_bgplacenet_stage2.yaml \
        --split test \
        --output outputs/lc_bgplacenet_stage2/direction_metadata_test.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(MPL_CONFIG_DIR))

from src.annotation.auto_label import center_distance, describe_spatial_relation, get_mapping, get_world_aabb
from src.training.lc_bgplacenet_stage2 import (
    build_sources_from_config,
    build_stage2_index,
    load_config,
    normalize_stage1_split,
    select_stage2_split_items,
)
from tools.benchmark_lc_bgplacenet_stage2 import (
    DIRECTION_METADATA_SCHEMA_VERSION,
    load_direction_scene_context,
    object_corners_world,
    parse_target_direction,
    place_box_to_corners,
)

DEFAULT_MAPPING = PROJECT_ROOT / "configs/annotation/mapping.json"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage 2 direction benchmark metadata.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lc_bgplacenet_stage2.yaml")
    parser.add_argument("--split", choices=("train", "valid", "val", "test"), default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/lc_bgplacenet_stage2/direction_metadata_test.json",
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    return parser.parse_args()


def _resolve_project_path(path_like: str | os.PathLike[str]) -> Path:
    """Resolve config paths relative to the project root."""
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_config_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with data paths resolved against PROJECT_ROOT."""
    resolved = copy.deepcopy(cfg)
    data_cfg = resolved.get("data", {})
    if data_cfg.get("split_dir") is not None:
        data_cfg["split_dir"] = os.fspath(_resolve_project_path(data_cfg["split_dir"]))
    for source in data_cfg.get("sources", []):
        for key in ("dataset_dir", "free_bbox_dir", "labels_path"):
            if source.get(key) is not None:
                source[key] = os.fspath(_resolve_project_path(source[key]))
    return resolved


def _normalize_name(name: str) -> str:
    """Normalize object display names for instruction matching."""
    return " ".join(str(name).strip().lower().split())


def _display_name(obj: Any, mapping_data: dict[str, str]) -> str:
    """Return the same display name used by auto-label instructions."""
    return str(mapping_data.get(obj.class_name, obj.class_name))


def _candidate_distance_cm(target_corners: np.ndarray, ref_corners: np.ndarray) -> float:
    """Use camera-independent 3D AABB center distance for tie-breaking."""
    target_min, target_max = get_world_aabb(target_corners)
    ref_min, ref_max = get_world_aabb(ref_corners)
    return center_distance(target_min, target_max, ref_min, ref_max)


def build_direction_metadata_item(item: Any, mapping_data: dict[str, str]) -> dict[str, Any]:
    """Build one metadata row by parsing the instruction and resolving its reference object."""
    target_relation, reference_name = parse_target_direction(item.instruction)
    scene_context = load_direction_scene_context(item)
    target_corners = place_box_to_corners(item.place_box_gt)
    reference_key = _normalize_name(reference_name)
    candidates = []

    for ref_obj in scene_context["object_by_id"].values():
        ref_names = {
            _normalize_name(_display_name(ref_obj, mapping_data)),
            _normalize_name(ref_obj.class_name),
        }
        if reference_key not in ref_names:
            continue
        ref_corners = object_corners_world(ref_obj)
        relation = describe_spatial_relation(
            target_corners,
            ref_corners,
            scene_context["camera"].E_w2c,
            scene_context["camera"].K,
        )
        if relation != target_relation:
            continue
        candidates.append((_candidate_distance_cm(target_corners, ref_corners), ref_obj))

    if not candidates:
        raise ValueError(
            f"Cannot resolve target reference for {item.item_id}: "
            f"relation={target_relation!r}, reference_name={reference_name!r}"
        )

    candidates.sort(key=lambda pair: pair[0])
    distance_cm, ref_obj = candidates[0]
    return {
        "item_id": item.item_id,
        "source_name": item.source_name,
        "sample_id": item.sample_id,
        "object_id": item.object_id,
        "cluster_id": int(item.cluster_id),
        "instruction": item.instruction,
        "target_relation": target_relation,
        "reference_object_id": str(ref_obj.obj_id),
        "reference_class_name": str(ref_obj.class_name),
        "reference_name": reference_name,
        "reference_distance_cm": float(distance_cm),
        "reference_candidate_count": int(len(candidates)),
    }


def main() -> None:
    """Generate metadata JSON for the requested fixed split."""
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    sources = build_sources_from_config(cfg)
    items = build_stage2_index(sources)
    split = normalize_stage1_split(args.split)
    split_items = select_stage2_split_items(items, cfg["data"]["split_dir"], split)
    mapping_data = get_mapping(os.fspath(args.mapping))

    metadata_items = [build_direction_metadata_item(item, mapping_data) for item in split_items]
    payload = {
        "schema_version": DIRECTION_METADATA_SCHEMA_VERSION,
        "split": split,
        "item_count": len(metadata_items),
        "items": metadata_items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(metadata_items)} direction metadata rows to {args.output}")


if __name__ == "__main__":
    main()
