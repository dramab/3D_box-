#!/usr/bin/env python3
"""Run DetAny3D zero-shot source-object localization on active_aligned."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The official repositories track some bytecode files. Keep inference read-only.
sys.dont_write_bytecode = True

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from box import Box
from PIL import Image
from torchvision.ops import box_convert

from active_aligned import (
    ActiveAlignedRecord,
    gt_center_camera_m,
    load_active_aligned_records,
    obb_corners_camera_m,
    processed_pixels_to_original,
    project_points,
    select_balanced_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PROJECT_ROOT / "baselines" / "detany3d"
DETANY_ROOT = BASELINE_ROOT / "external" / "DetAny3D"
GROUNDING_DINO_ROOT = BASELINE_ROOT / "external" / "GroundingDINO"
CHECKPOINT_ROOT = BASELINE_ROOT / "checkpoints"
BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DetAny3D zero-shot source localization on the fixed active_aligned split."
    )
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.37)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "detany3d_source_localization_exact_name",
    )
    return parser.parse_args()


def configure_external_imports() -> None:
    """Use project-local external repositories without installing them globally."""
    for path in (DETANY_ROOT, GROUNDING_DINO_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def validate_runtime(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda"):
        raise ValueError("DetAny3D qualitative inference requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Run this script on the GPU host, not the restricted sandbox.")
    required = (
        CHECKPOINT_ROOT / "detany3d_ckpts" / "zero_shot_category_ckpt.pth",
        CHECKPOINT_ROOT / "unidepth_ckpts" / "model.pth",
        CHECKPOINT_ROOT / "dino_ckpts" / "dinov2_vitl14_pretrain.pth",
        CHECKPOINT_ROOT / "sam_ckpts" / "sam_vit_h_4b8939.pth",
        CHECKPOINT_ROOT / "groundingdino_ckpts" / "groundingdino_swinb_cogcoor.pth",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))


def select_records(args: argparse.Namespace) -> List[ActiveAlignedRecord]:
    records = load_active_aligned_records(PROJECT_ROOT, split=args.split)
    if args.sample_id is not None or args.object_id is not None:
        selected = [
            record
            for record in records
            if (args.sample_id is None or record.sample_id == args.sample_id)
            and (args.object_id is None or record.object_id == args.object_id)
        ]
        if not selected:
            raise ValueError("No unique source object matches --sample-id/--object-id")
        return selected[: args.max_samples]
    return select_balanced_records(records, args.max_samples, args.seed)


def load_grounding_dino(device: str) -> Any:
    from groundingdino.util.inference import load_model

    config_path = GROUNDING_DINO_ROOT / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py"
    checkpoint_path = (
        CHECKPOINT_ROOT / "groundingdino_ckpts" / "groundingdino_swinb_cogcoor.pth"
    )
    return load_model(str(config_path), str(checkpoint_path), device=device)


def run_grounding_dino(
    model: Any,
    records: Sequence[ActiveAlignedRecord],
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Run exact-name prompts first so GroundingDINO can be released before DetAny3D."""
    import groundingdino.datasets.transforms as transforms
    from groundingdino.util.inference import predict

    transform = transforms.Compose(
        [
            transforms.RandomResize([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    detections: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        image = Image.open(record.rgb_path).convert("RGB")
        image_tensor, _ = transform(image, None)
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=record.prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            remove_combined=False,
        )
        key = (record.source_name, record.sample_id, record.object_id)
        if len(boxes) == 0:
            detections[key] = {"status": "no_grounding_detection"}
        else:
            best = int(torch.argmax(logits).item())
            height, width = np.asarray(image).shape[:2]
            box_xyxy = box_convert(boxes[best : best + 1], "cxcywh", "xyxy")[0]
            box_xyxy = box_xyxy * torch.tensor([width, height, width, height])
            detections[key] = {
                "status": "ok",
                "box_xyxy": box_xyxy.numpy().astype(np.float32),
                "score": float(logits[best].item()),
                "phrase": str(phrases[best]),
            }
        print(f"[GroundingDINO {index}/{len(records)}] {record.prompt} -> {detections[key]['status']}")
    return detections


def load_detany3d(device: str) -> Tuple[Any, Any]:
    """Build the official novel-class model and load shape-compatible checkpoint entries."""
    from wrap_model import WrapModel

    config_path = (
        DETANY_ROOT
        / "detect_anything"
        / "configs"
        / "inference_novel_cls_gdino_prompt_previous_metric.yaml"
    )
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = Box(yaml.safe_load(handle))
    cfg.resume = str(CHECKPOINT_ROOT / "detany3d_ckpts" / "zero_shot_category_ckpt.pth")
    cfg.unidepth_path = str(CHECKPOINT_ROOT / "unidepth_ckpts" / "model.pth")
    cfg.dino_path = str(CHECKPOINT_ROOT / "dino_ckpts" / "dinov2_vitl14_pretrain.pth")
    cfg.model.checkpoint = str(CHECKPOINT_ROOT / "sam_ckpts" / "sam_vit_h_4b8939.pth")

    model = WrapModel(cfg)
    checkpoint = torch.load(cfg.resume, map_location="cpu")
    checkpoint_state = checkpoint["state_dict"]
    model_state = model.state_dict()
    loaded = 0
    for key, value in model_state.items():
        candidate = checkpoint_state.get(key)
        if candidate is not None and candidate.shape == value.shape:
            model_state[key] = candidate.detach()
            loaded += 1
    if loaded == 0:
        raise RuntimeError(f"No compatible parameters found in {cfg.resume}")
    model.load_state_dict(model_state)
    del checkpoint, checkpoint_state, model_state
    gc.collect()

    model.to(device)
    model.setup()
    model.eval()
    print(f"Loaded {loaded} DetAny3D checkpoint tensors")
    return model, cfg


def prepare_detany_input(
    image_rgb: np.ndarray, box_xyxy: np.ndarray, cfg: Any, device: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Apply the official deploy.py resize, center crop, normalization, and padding."""
    from detect_anything.utils.transforms import ResizeLongestSide

    transform = ResizeLongestSide(int(cfg.model.pad))
    original_hw = tuple(int(value) for value in image_rgb.shape[:2])
    image_tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0)
    resized = transform.apply_image_torch(image_tensor)
    resized_h, resized_w = (int(resized.shape[-2]), int(resized.shape[-1]))
    if max(resized_h, resized_w) % 112 != 0:
        raise ValueError("Official DetAny3D resize expects the long side to be divisible by 112")
    crop_h, crop_w = (resized_h // 14) * 14, (resized_w // 14) * 14
    crop_y = resized_h // 2 - crop_h // 2
    crop_x = resized_w // 2 - crop_w // 2
    cropped = resized[:, :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

    mean = torch.tensor(cfg.dataset.pixel_mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.dataset.pixel_std).view(1, 3, 1, 1)
    image_for_sam = (cropped - mean) / std
    image_for_sam = F.pad(
        image_for_sam,
        (0, int(cfg.model.pad) - crop_w, 0, int(cfg.model.pad) - crop_h),
    ).to(device)
    dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_for_dino = ((cropped / 255.0 - dino_mean) / dino_std).to(device)

    if cfg.model.vit_pad_mask:
        vit_pad_size = (crop_h // cfg.model.image_encoder.patch_size, crop_w // cfg.model.image_encoder.patch_size)
    else:
        size = cfg.model.pad // cfg.model.image_encoder.patch_size
        vit_pad_size = (size, size)
    official_box = transform.apply_boxes_torch(
        torch.from_numpy(np.asarray(box_xyxy, dtype=np.float32)[None, :]), original_hw
    ).to(torch.int32)
    input_dict = {
        "images": image_for_sam,
        "vit_pad_size": torch.tensor(vit_pad_size, device=device).unsqueeze(0),
        "images_shape": torch.tensor((crop_h, crop_w), dtype=torch.float32, device=device).unsqueeze(0),
        "image_for_dino": image_for_dino,
        "boxes_coords": official_box.to(device),
    }
    geometry = {
        "original_hw": original_hw,
        "resized_hw": (resized_h, resized_w),
        "crop_xy": (crop_x, crop_y),
    }
    return input_dict, geometry


def infer_detany3d(
    model: Any,
    cfg: Any,
    record: ActiveAlignedRecord,
    detection: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    from detect_anything.datasets.utils import compute_3d_bbox_vertices, rotation_6d_to_matrix
    from train_utils import decode_bboxes

    image_rgb = np.asarray(Image.open(record.rgb_path).convert("RGB"), dtype=np.uint8)
    input_dict, geometry = prepare_detany_input(image_rgb, detection["box_xyxy"], cfg, device)
    with torch.no_grad():
        output = model(input_dict)
        predicted_k = output["pred_K"]
        _, predicted_boxes_3d = decode_bboxes(output, cfg, predicted_k)
        predicted_rotation = rotation_6d_to_matrix(output["pred_pose_6d"])

    box_3d = predicted_boxes_3d[0].detach().cpu().numpy().astype(np.float32)
    rotation = predicted_rotation[0].detach().cpu().numpy().astype(np.float32)
    predicted_k_np = predicted_k[0].detach().cpu().numpy().astype(np.float32)
    corners_camera_m, _ = compute_3d_bbox_vertices(*box_3d, rotation_matrix=rotation)
    corners_processed = project_points(corners_camera_m, predicted_k_np)
    corners_original = processed_pixels_to_original(corners_processed, **geometry)

    gt_corners_camera = obb_corners_camera_m(record)
    gt_corners_original = project_points(gt_corners_camera, record.camera_k)
    gt_center = gt_center_camera_m(record)
    center_error_m = float(np.linalg.norm(box_3d[:3] - gt_center))
    grounding_iou = box_iou_xyxy(
        detection["box_xyxy"],
        np.concatenate([gt_corners_original.min(axis=0), gt_corners_original.max(axis=0)]),
    )
    return {
        "image_rgb": image_rgb,
        "predicted_box_3d": box_3d,
        "predicted_rotation": rotation,
        "predicted_k": predicted_k_np,
        "predicted_corners_original": corners_original,
        "gt_center_camera_m": gt_center,
        "gt_corners_original": gt_corners_original,
        "center_error_m": center_error_m,
        "grounding_iou_2d": grounding_iou,
    }


def box_iou_xyxy(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    intersection_min = np.maximum(first[:2], second[:2])
    intersection_max = np.minimum(first[2:], second[2:])
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = float(intersection_size[0] * intersection_size[1])
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def render_result(
    record: ActiveAlignedRecord,
    detection: Dict[str, Any],
    inference: Optional[Dict[str, Any]],
    output_path: Path,
) -> None:
    image_rgb = np.asarray(Image.open(record.rgb_path).convert("RGB"), dtype=np.uint8)
    left = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    right = left.copy()
    if detection["status"] == "ok":
        x1, y1, x2, y2 = np.rint(detection["box_xyxy"]).astype(int)
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 210, 255), 5)
        draw_label(left, f"GroundingDINO {detection['score']:.3f}", (x1, max(y1 - 12, 30)))
    else:
        draw_label(left, "GroundingDINO: no detection", (20, 45), color=(0, 0, 255))

    if inference is not None:
        draw_cuboid(right, inference["gt_corners_original"], (80, 220, 80), 5)
        draw_cuboid(right, inference["predicted_corners_original"], (255, 180, 0), 5)
        draw_label(right, "GT 3D", (20, 42), color=(80, 220, 80))
        draw_label(right, "DetAny3D", (20, 82), color=(255, 180, 0))
        draw_label(
            right,
            f"center error: {inference['center_error_m']:.3f} m",
            (20, 122),
            color=(255, 255, 255),
        )

    panel = np.concatenate([left, right], axis=1)
    banner_height = 92
    canvas = np.full((panel.shape[0] + banner_height, panel.shape[1], 3), 25, dtype=np.uint8)
    canvas[banner_height:] = panel
    draw_label(canvas, f"prompt: {record.prompt}", (20, 34), color=(255, 255, 255), scale=0.9)
    draw_label(
        canvas,
        f"{record.source_name} | {record.sample_id} | {record.object_id}",
        (20, 72),
        color=(190, 190, 190),
        scale=0.7,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def draw_cuboid(image: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int], width: int) -> None:
    if not np.all(np.isfinite(corners)):
        return
    rounded = np.rint(corners).astype(int)
    for start, end in BOX_EDGES:
        cv2.line(image, tuple(rounded[start]), tuple(rounded[end]), color, width, cv2.LINE_AA)


def draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.75,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def write_contact_sheet(paths: Sequence[Path], output_path: Path) -> None:
    images = [cv2.imread(str(path)) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    target_width = 1200
    thumbnails = [
        cv2.resize(image, (target_width, int(image.shape[0] * target_width / image.shape[1])))
        for image in images
    ]
    columns = 2
    rows = (len(thumbnails) + columns - 1) // columns
    cell_height = max(image.shape[0] for image in thumbnails)
    sheet = np.full((rows * cell_height, columns * target_width, 3), 30, dtype=np.uint8)
    for index, image in enumerate(thumbnails):
        row, column = divmod(index, columns)
        sheet[row * cell_height : row * cell_height + image.shape[0], column * target_width : (column + 1) * target_width] = image
    cv2.imwrite(str(output_path), sheet)


def result_record(
    record: ActiveAlignedRecord,
    detection: Dict[str, Any],
    inference: Optional[Dict[str, Any]],
    visualization_path: Path,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "item_id": record.item_id,
        "source_name": record.source_name,
        "sample_id": record.sample_id,
        "object_id": record.object_id,
        "class_name": record.class_name,
        "instruction": record.instruction,
        "model_prompt": record.prompt,
        "rgb_path": str(record.rgb_path),
        "status": detection["status"],
        "visualization_path": str(visualization_path),
        "input_contract": {
            "grounding_dino": ["rgb", "model_prompt"],
            "detany3d": ["rgb", "predicted_2d_box"],
            "gt_used_by_model": False,
        },
    }
    if detection["status"] == "ok":
        result["grounding_dino"] = {
            "box_xyxy": detection["box_xyxy"].tolist(),
            "score": detection["score"],
            "phrase": detection["phrase"],
        }
    if inference is not None:
        result["detany3d"] = {
            "box_camera_m": inference["predicted_box_3d"].tolist(),
            "rotation_matrix": inference["predicted_rotation"].tolist(),
            "predicted_intrinsics": inference["predicted_k"].tolist(),
        }
        result["gt_evaluation_only"] = {
            "center_camera_m": inference["gt_center_camera_m"].tolist(),
            "center_error_m": inference["center_error_m"],
            "grounding_iou_2d": inference["grounding_iou_2d"],
        }
    return result


def main() -> None:
    args = parse_args()
    configure_external_imports()
    validate_runtime(args)
    records = select_records(args)
    output_dir = args.output_dir.resolve()
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(BASELINE_ROOT / "cache" / "huggingface"))

    grounding_model = load_grounding_dino(args.device)
    detections = run_grounding_dino(
        grounding_model,
        records,
        args.device,
        args.box_threshold,
        args.text_threshold,
    )
    del grounding_model
    gc.collect()
    torch.cuda.empty_cache()

    detany_model, cfg = load_detany3d(args.device)
    results = []
    visualization_paths = []
    for index, record in enumerate(records, start=1):
        key = (record.source_name, record.sample_id, record.object_id)
        detection = detections[key]
        inference = None
        if detection["status"] == "ok":
            inference = infer_detany3d(detany_model, cfg, record, detection, args.device)
        stem = f"{record.source_name}__{record.sample_id}__{record.object_id}"
        visualization_path = visualization_dir / f"{stem}.png"
        render_result(record, detection, inference, visualization_path)
        visualization_paths.append(visualization_path)
        results.append(result_record(record, detection, inference, visualization_path))
        print(f"[DetAny3D {index}/{len(records)}] {stem} -> {detection['status']}")

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    write_contact_sheet(visualization_paths, output_dir / "contact_sheet.png")
    print(f"Results: {results_path}")
    print(f"Contact sheet: {output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
