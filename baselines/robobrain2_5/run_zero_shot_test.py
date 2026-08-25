#!/usr/bin/env python
"""Run RoboBrain2.5 zero-shot point prediction and render every fixed test item.

Usage:
    conda activate vlm_qwen
    python baselines/robobrain2_5/run_zero_shot_test.py \
        --model-dir baselines/robobrain2_5/hf_cache/RoboBrain2.5-8B-NV
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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.robobrain2_5.zero_shot_visualization import (
    Camera,
    backproject_world,
    load_jsonl,
    local_depth_cm,
    normalized_to_pixel,
    parse_point_answer,
    parse_point_depth_answer,
    render_composite,
    source_dimensions_and_yaw,
    transform_points,
    upright_box_corners,
    write_jsonl,
    write_summary,
    write_web_gallery,
)
from src.datasets.canonical import load_sample_record


DEFAULT_CONFIG = PROJECT_ROOT / "configs/lc_bgplacenet_stage2.yaml"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "baselines/robobrain2_5/hf_cache/RoboBrain2.5-8B-NV"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/robobrain2_5_front_behind_swapped"
ENRICHED_BOTTOM_CENTER_XYD_PROMPT_VARIANT = "enriched_label_bottom_center_xyd"


OFFICIAL_POINTING_SUFFIX = (
    " Please provide its 2D coordinates. Your answer should be formatted as a tuple, "
    "i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above."
)

BOTTOM_CENTER_XYD_SUFFIX = (
    "\nYour answer should be formatted as a single-element list containing one tuple, "
    "i.e., [(x, y, d)], where x and y are normalized image coordinates in the range "
    "[0, 1000], and d is the absolute camera depth in centimeters."
)

FRONT_BEHIND_RELATION_SWAPS = (
    ("in front of ", "behind "),
    ("behind ", "in front of "),
    ("the front left of ", "the back left of "),
    ("the back left of ", "the front left of "),
    ("the front right of ", "the back right of "),
    ("the back right of ", "the front right of "),
)


def swap_front_behind_relation(destination_clause: str) -> str:
    """Swap the benchmark's front/back direction while preserving the reference object."""
    for source_prefix, target_prefix in FRONT_BEHIND_RELATION_SWAPS:
        if destination_clause.startswith(source_prefix):
            return target_prefix + destination_clause[len(source_prefix) :]
    return destination_clause


def pointing_prompt_from_instruction(instruction: str) -> str:
    """Rewrite a move label and align its front/back direction with RoboBrain."""
    text = str(instruction).strip().rstrip(".")
    source_clause, separator, destination_clause = text.rpartition(" to ")
    if not separator or not source_clause.startswith("Move ") or not destination_clause:
        raise ValueError(f"Expected a templated move instruction, got: {instruction!r}")
    destination_clause = swap_front_behind_relation(destination_clause)
    return f"Identify spot within the vacant space that's {destination_clause}."


def model_prompt_from_instruction(instruction: str) -> str:
    """Return the exact prompt passed to the model, including the official suffix."""
    return pointing_prompt_from_instruction(instruction) + OFFICIAL_POINTING_SUFFIX



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoboBrain2.5 point prediction on the fixed test split.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Optional local PEFT/LoRA adapter directory.")
    parser.add_argument(
        "--prompt-variant",
        choices=("official_vacant_space_front_behind_swapped", ENRICHED_BOTTOM_CENTER_XYD_PROMPT_VARIANT),
        default="official_vacant_space_front_behind_swapped",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-samples", type=int, default=None, help="Only for a smoke run; omit for all fixed test items.")
    parser.add_argument("--no-resume", action="store_true", help="Fail if result rows already exist instead of resuming.")
    parser.add_argument("--finalize-only", action="store_true", help="Regenerate summary and gallery from predictions.jsonl without inference.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_sources(config_path: Path) -> dict[str, dict[str, Path]]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return {
        str(source["name"]): {
            "dataset_dir": resolve_path(source["dataset_dir"]),
            "free_bbox_dir": resolve_path(source["free_bbox_dir"]),
            "labels_path": resolve_path(source["labels_path"]),
        }
        for source in config["data"]["sources"]
    }


def load_test_items(config_path: Path) -> list[dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    split_dir = resolve_path(config["data"]["split_dir"])
    with (split_dir / "test.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = list(payload["items"])
    if payload.get("split") != "test":
        raise ValueError(f"Expected test split, got {payload.get('split')!r}")
    return items


@lru_cache(maxsize=16)
def load_json_payload(path_text: str) -> Any:
    with Path(path_text).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_label_record(item: dict[str, Any], source: dict[str, Path]) -> dict[str, Any]:
    """Resolve and validate the config-specific label referenced by one split item."""
    labels = load_json_payload(str(source["labels_path"]))
    label = labels[int(item["label_index"])]
    if str(label["sample_id"]) != str(item["sample_id"]) or str(label["object_id"]) != str(item["object_id"]):
        raise ValueError(f"Label index does not match split item: {item['item_id']}")
    return label


def model_prompt_from_item(item: dict[str, Any], source: dict[str, Path]) -> str:
    """Build the official pointing prompt from structured enriched metadata when available."""
    label = load_label_record(item, source)
    placement_relation = label.get("spatial_relation", {}).get("placement", {})
    relation = placement_relation.get("relation")
    reference_name = placement_relation.get("reference_name")
    if relation and reference_name:
        destination_clause = swap_front_behind_relation(f"{relation} {reference_name}")
        return f"Identify spot within the vacant space that's {destination_clause}." + OFFICIAL_POINTING_SUFFIX
    return model_prompt_from_instruction(str(item["instruction"]))


def bottom_center_xyd_prompt_from_item(item: dict[str, Any], source: dict[str, Path]) -> str:
    """Keep the enriched label verbatim and request one bottom-center ``(x, y, d)`` point."""
    label = load_label_record(item, source)
    enriched_label = str(label["label"]).strip()
    if enriched_label != str(item["instruction"]).strip():
        raise ValueError(f"Enriched label does not match split instruction: {item['item_id']}")
    return (
        "Please predict the bottom-center placement point of the object to successfully complete the task. "
        f'The task is: "{enriched_label}"'
        + BOTTOM_CENTER_XYD_SUFFIX
    )


def load_ground_truth_placement(item: dict[str, Any], source: dict[str, Path]) -> dict[str, Any]:
    """Read the held-out placement box for visualization only; it is never sent to RoboBrain."""
    label = load_label_record(item, source)
    placement_path = source["free_bbox_dir"] / "placements" / f"{item['sample_id']}__placements.json"
    payload = load_json_payload(str(placement_path))
    object_record = next(record for record in payload["objects"] if str(record["object_id"]) == str(item["object_id"]))
    placement_id = str(label["placement_sample_id"])
    placement = next(record for record in object_record["placements"] if str(record["sample_id"]) == placement_id)
    center = np.asarray(placement["center_world"], dtype=np.float64)
    dimensions = np.asarray(placement["yaw_only_dimensions"], dtype=np.float64)
    yaw = float(np.deg2rad(float(placement["yaw_degrees"])))
    bottom_center = np.asarray(placement.get("bottom_center_world"), dtype=np.float64)
    return {
        "bottom_center_world": bottom_center,
        "box": [*center.tolist(), *dimensions.tolist(), yaw],
        "corners": upright_box_corners(center, dimensions, yaw),
    }


@lru_cache(maxsize=8)
def load_scene(dataset_dir_text: str, sample_id: str) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir_text)
    record = load_sample_record(dataset_dir / "samples" / f"{sample_id}.json")
    camera_record = record["camera"]
    camera = Camera(
        fx=float(camera_record["fx"]), fy=float(camera_record["fy"]),
        cx=float(camera_record["cx"]), cy=float(camera_record["cy"]),
        width=int(camera_record["img_w"]), height=int(camera_record["img_h"]),
        e_c2w=np.asarray(camera_record["E_c2w"], dtype=np.float64),
    )
    rgb = np.asarray(Image.open(dataset_dir / str(record["rgb_path"])).convert("RGB"), dtype=np.uint8)
    depth = np.load(dataset_dir / str(record["depth_path"])).astype(np.float32)
    return {
        "rgb": rgb,
        "rgb_path": dataset_dir / str(record["rgb_path"]),
        "depth": depth,
        "camera": camera,
        "objects": list(record["objects"]),
    }


@lru_cache(maxsize=4096)
def load_rgb_path(dataset_dir_text: str, sample_id: str) -> Path:
    """Resolve the sole visual model input without loading depth or oracle geometry."""
    dataset_dir = Path(dataset_dir_text)
    record = load_sample_record(dataset_dir / "samples" / f"{sample_id}.json")
    return dataset_dir / str(record["rgb_path"])


class RoboBrainPointModel:
    """Minimal wrapper around the official single-image pointing inference path."""

    def __init__(self, model_dir: Path, max_new_tokens: int, adapter_dir: Path | None = None) -> None:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError(
                "RoboBrain2.5-8B-NV inference requires a CUDA-visible GPU; refusing an impractical CPU fallback."
            )
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype="auto", device_map="auto", local_files_only=True
        )
        if adapter_dir is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir, is_trainable=False)
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.input_device = next(self.model.parameters()).device

    def predict(self, model_prompt: str, rgb_path: Path) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": f"file://{rgb_path}"}, {"type": "text", "text": model_prompt}]}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[chat], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.input_device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def output_record(
    item: dict[str, Any], scene: dict[str, Any], ground_truth: dict[str, Any], model_prompt: str,
    answer: str, output_dir: Path, model_dir: Path | None = None, adapter_dir: Path | None = None,
    prediction_format: str = "xy",
) -> dict[str, Any]:
    predicted_depth_cm: float | None = None
    depth_source: str | None = None
    if prediction_format == "xyd":
        point_depth, status = parse_point_depth_answer(answer)
        point = point_depth[:2] if point_depth is not None else None
        predicted_depth_cm = point_depth[2] if point_depth is not None else None
        depth_source = "model_absolute_camera_depth_cm" if point_depth is not None else None
    elif prediction_format == "xy":
        point, status = parse_point_answer(answer)
    else:
        raise ValueError(f"Unknown prediction format: {prediction_format}")
    pixel: tuple[int, int] | None = None
    world_point: np.ndarray | None = None
    candidate_corners: np.ndarray | None = None
    candidate_box: list[float] | None = None
    if point is not None:
        pixel, clamped = normalized_to_pixel(point, scene["camera"].width, scene["camera"].height)
        if clamped:
            status = "point_clamped"
        if prediction_format == "xyd":
            depth_cm = predicted_depth_cm
            if depth_cm is None or not np.isfinite(depth_cm) or depth_cm <= 0.0:
                depth_cm = None
                status = "depth_invalid"
        else:
            depth_cm = local_depth_cm(scene["depth"], pixel)
            depth_source = "rgbd_local_median_cm" if depth_cm is not None else None
        if depth_cm is None:
            if status != "depth_invalid":
                status = "depth_failed"
        else:
            world_point = backproject_world(pixel, depth_cm, scene["camera"])
            source = next(obj for obj in scene["objects"] if str(obj["obj_id"]) == str(item["object_id"]))
            dimensions, yaw = source_dimensions_and_yaw(source)
            center = world_point.copy()
            center[2] += dimensions[2] * 0.5
            candidate_corners = upright_box_corners(center, dimensions, yaw)
            candidate_box = [*center.tolist(), *dimensions.tolist(), float(yaw)]

    visual_rel = Path("visualizations") / f"{item['item_id']}.jpg"
    visual_path = output_dir / visual_rel
    image = render_composite(
        scene["rgb"], scene["camera"], scene["objects"], str(item["object_id"]), pixel, candidate_corners,
        np.asarray(ground_truth["bottom_center_world"]), np.asarray(ground_truth["corners"]), answer, status,
    )
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(visual_path, quality=88, optimize=True)
    cluster_id = item.get("cluster_id")
    return {
        "item_id": str(item["item_id"]), "source_name": str(item["source_name"]), "sample_id": str(item["sample_id"]),
        "object_id": str(item["object_id"]),
        "cluster_id": int(cluster_id) if cluster_id is not None else None,
        "instruction": str(item["instruction"]),
        "model_prompt": model_prompt,
        "model_checkpoint": str(model_dir) if model_dir is not None else None,
        "adapter_checkpoint": str(adapter_dir) if adapter_dir is not None else None,
        "raw_answer": answer, "normalized_point": list(point) if point is not None else None,
        "predicted_depth_cm": predicted_depth_cm,
        "depth_source": depth_source,
        "prediction_format": prediction_format,
        "pixel": list(pixel) if pixel is not None else None, "world_point_cm": world_point.tolist() if world_point is not None else None,
        "render_box_world": candidate_box, "status": status, "visualization_path": str(visual_rel),
        "oracle_source_geometry_for_visualization": True,
        "oracle_source_geometry_for_candidate_construction": True,
        "gt_bottom_center_world": np.asarray(ground_truth["bottom_center_world"]).tolist(),
        "gt_place_box_world": ground_truth["box"],
        "gt_for_visualization_only": True,
    }


def finalize(output_dir: Path) -> None:
    rows = load_jsonl(output_dir / "predictions.jsonl")
    write_summary(rows, output_dir / "summary.json")
    web_dir = output_dir / "web_vis"
    web_dir.mkdir(parents=True, exist_ok=True)
    write_web_gallery(rows, web_dir / "index.html")
    print(f"Wrote {len(rows)} records, summary.json, and {web_dir / 'index.html'}")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    if args.finalize_only:
        finalize(output_dir)
        return
    if predictions_path.exists() and args.no_resume:
        raise FileExistsError(f"Result file already exists: {predictions_path}")
    sources = load_sources(config_path)
    items = load_test_items(config_path)
    if args.max_samples is not None:
        items = items[: int(args.max_samples)]
    completed = {str(row["item_id"]) for row in load_jsonl(predictions_path)}
    model_dir = resolve_path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"RoboBrain checkpoint is incomplete: {model_dir}")
    adapter_dir = resolve_path(args.adapter_dir) if args.adapter_dir is not None else None
    if adapter_dir is not None:
        if output_dir == DEFAULT_OUTPUT_DIR:
            raise ValueError("A fine-tuned adapter requires a distinct --output-dir; refusing to mix zero-shot results.")
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"RoboBrain LoRA adapter is incomplete: {adapter_dir}")
    run_config = {
        "config": str(config_path),
        "model_dir": str(model_dir),
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "split": args.split,
        "prompt_variant": str(args.prompt_variant),
        "prediction_format": "xyd" if args.prompt_variant == ENRICHED_BOTTOM_CENTER_XYD_PROMPT_VARIANT else "xy",
        "max_new_tokens": int(args.max_new_tokens),
    }
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        with run_config_path.open("r", encoding="utf-8") as handle:
            previous_run_config = json.load(handle)
        if previous_run_config != run_config:
            raise ValueError(f"Existing output directory belongs to a different run: {run_config_path}")
    else:
        with run_config_path.open("w", encoding="utf-8") as handle:
            json.dump(run_config, handle, indent=2, ensure_ascii=False)
    model = RoboBrainPointModel(model_dir, args.max_new_tokens, adapter_dir=adapter_dir)
    for index, item in enumerate(items, start=1):
        item_id = str(item["item_id"])
        if item_id in completed:
            continue
        source = sources[str(item["source_name"])]
        dataset_dir = source["dataset_dir"]
        if args.prompt_variant == ENRICHED_BOTTOM_CENTER_XYD_PROMPT_VARIANT:
            model_prompt = bottom_center_xyd_prompt_from_item(item, source)
            prediction_format = "xyd"
        else:
            model_prompt = model_prompt_from_item(item, source)
            prediction_format = "xy"
        rgb_path = load_rgb_path(str(dataset_dir), str(item["sample_id"]))
        inference_error: Exception | None = None
        try:
            answer = model.predict(model_prompt, rgb_path)
        except Exception as error:
            answer = f"INFERENCE_ERROR: {type(error).__name__}: {error}"
            inference_error = error
        # Depth, scene objects, and GT are deliberately read only after model inference.
        scene = load_scene(str(dataset_dir), str(item["sample_id"]))
        ground_truth = load_ground_truth_placement(item, source)
        row = output_record(
            item, scene, ground_truth, model_prompt, answer, output_dir, model_dir, adapter_dir, prediction_format
        )
        if inference_error is not None:
            row["status"] = "inference_failed"
        write_jsonl(predictions_path, row)
        print(f"[{index}/{len(items)}] {item_id}: {row['status']}", flush=True)
    finalize(output_dir)


if __name__ == "__main__":
    main()
