"""Stage 2 decoder inference export and web compatibility tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import tools.infer_lc_bgplacenet_stage2 as inference
import tools.render_lc_bgplacenet_stage2_point_mask as point_mask_vis
from src.annotation.free_bbox.io_utils import save_ply
from src.models.lc_bgplacenet.stage2 import NUM_YAW_BINS
from tools.benchmark_lc_bgplacenet_stage2 import load_predictions
from tools.export_lc_bgplacenet_stage2_inference_web import collect_rows, write_html


def test_continuous_mask_maps_all_scores_and_preserves_zero_response_color() -> None:
    scene_points = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    heatmap_points = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    heatmap_scores = np.asarray([0.2, 0.8], dtype=np.float32)
    colors = np.asarray([[100, 120, 140], [40, 60, 80], [10, 20, 30]], dtype=np.uint8)

    point_scores = point_mask_vis.map_continuous_scores(
        scene_points, heatmap_points, heatmap_scores, p3_stride_cm=4.0
    )
    blended = point_mask_vis.blend_continuous_mask(colors, point_scores)
    point_sizes = point_mask_vis.continuous_mask_point_sizes(point_scores)

    np.testing.assert_allclose(point_scores, [0.2, 0.8, 0.0])
    np.testing.assert_allclose(blended[2], colors[2] / 255.0)
    assert blended[1, 0] > colors[1, 0] / 255.0
    assert point_sizes[1] > point_sizes[0] > point_sizes[2]
    assert point_sizes[2] == point_mask_vis.LOCAL_POINT_SIZE


def test_local_crop_keeps_semantic_core_and_removes_unrelated_points() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    scores = np.asarray([0.0, 0.8, 0.0], dtype=np.float32)
    source_corners = np.asarray([[0.0, 0.0, 0.0]])
    prediction_corners = np.asarray([[2.0, 0.0, 0.0]])

    keep = point_mask_vis.local_crop_mask(
        points, scores, source_corners, prediction_corners
    )

    assert keep.tolist() == [True, True, False]


def test_prediction_only_item_uses_canonical_source_metadata(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "hope"
    sample_dir = dataset_dir / "samples"
    sample_dir.mkdir(parents=True)
    sample_id = "hope__scene_0000__0005"
    sample_record = {
        "rgb_path": "rgb/sample.jpg",
        "voxel_point_cloud_path": "point_clouds_voxel_1cm/sample.ply",
        "camera": {
            "fx": 100.0,
            "fy": 101.0,
            "cx": 50.0,
            "cy": 40.0,
            "E_c2w": np.eye(4).tolist(),
        },
        "objects": [
            {
                "obj_id": "obj_3",
                "bbox3d_canonical": [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0],
                "pose_world": np.eye(4).tolist(),
            }
        ],
    }
    (sample_dir / f"{sample_id}.json").write_text(json.dumps(sample_record), encoding="utf-8")
    source = SimpleNamespace(name="hope", dataset_dir=dataset_dir, free_bbox_dir=tmp_path / "free_bbox")
    monkeypatch.setattr(inference, "build_sources_from_config", lambda _cfg: [source])

    item = inference.build_prediction_only_item({}, sample_id, "obj_3", "Place the object behind the can.")

    assert item.sample_id == sample_id
    assert item.object_id == "obj_3"
    assert item.instruction == "Place the object behind the can."
    assert item.source_box_gt.tolist() == [0.0, 0.0, 0.0, 2.0, 4.0, 6.0]
    assert item.place_box_gt.shape == (7,)


def _decoder_prediction(center_offset: float) -> dict[str, torch.Tensor]:
    centers = torch.tensor([[[center_offset, 0.0, 0.0], [center_offset + 3.0, 0.0, 0.0]]])
    yaw_logits = torch.zeros(1, 2, NUM_YAW_BINS)
    return {
        "pred_bottom_centers": centers,
        "pred_yaw_indices": torch.zeros(1, 2, dtype=torch.long),
        "pred_yaw_logits": yaw_logits,
        "pred_logits": torch.tensor([[2.0, 1.0]]),
    }


def test_decoder_stage_export_reuses_final_benchmark_prediction(monkeypatch) -> None:
    final_boxes = torch.full((1, 16, 7), 9.0)
    final_scores = torch.full((1, 16), 0.75)
    final_yaw_bins = torch.full((1, 16), 2, dtype=torch.long)
    final_valid = torch.zeros(1, 16, dtype=torch.bool)
    final_valid[:, 0] = True
    final_box = final_boxes[:, 0]
    outputs = {
        "source_box": torch.tensor([[0.0, 0.0, 0.0, 2.0, 4.0, 3.0]]),
        "query_valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "decoder_aux_outputs": [_decoder_prediction(float(index)) for index in range(3)],
        "place_boxes": final_boxes,
        "place_scores": final_scores,
        "place_yaw_bins": final_yaw_bins,
        "place_valid_mask": final_valid,
        "place_box": final_box,
    }
    original_pose_nms = inference.pose_nms
    call_count = 0

    def counted_pose_nms(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_pose_nms(*args, **kwargs)

    monkeypatch.setattr(inference, "pose_nms", counted_pose_nms)
    stages = inference.build_decoder_stage_predictions(outputs)

    assert len(stages) == 4
    assert call_count == 3
    assert stages[-1]["place_boxes"] is final_boxes
    assert stages[-1]["place_box"] is final_box


def test_decoder_stage_record_contains_only_web_additive_fields(tmp_path) -> None:
    stage = {
        "place_boxes": np.zeros((16, 7), dtype=np.float32),
        "place_scores": np.linspace(0.9, 0.1, 16, dtype=np.float32),
        "place_yaw_bins": np.arange(16, dtype=np.int64),
        "place_valid_mask": np.array([True, True] + [False] * 14),
    }
    record = inference.decoder_stage_to_record(2, stage, tmp_path / "decoder_2.png")

    assert record["stage"] == 2
    assert len(record["placements"]) == 2
    assert record["placements"][0]["yaw_bin"] == 0
    assert set(record) == {"stage", "best_place_score", "placements", "visualization_png"}


def test_web_collects_four_stages_and_benchmark_ignores_extra_fields(tmp_path) -> None:
    input_dir = tmp_path / "inference"
    output_dir = input_dir / "web_vis"
    input_dir.mkdir()
    stage_paths = []
    for stage_index in range(1, 5):
        stage_path = input_dir / f"decoder_{stage_index}.png"
        Image.new("RGB", (32, 24), "white").save(stage_path)
        stage_paths.append(stage_path)
    prediction_png = input_dir / "prediction.png"
    Image.new("RGB", (32, 24), "white").save(prediction_png)
    heatmap_ply = input_dir / "heatmap.ply"
    save_ply(
        heatmap_ply,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        np.asarray([[0, 0, 255], [255, 0, 0]], dtype=np.uint8),
    )
    prediction = {
        "item_id": "sample__obj__cluster",
        "sample_id": "sample",
        "object_id": "obj",
        "cluster_id": 0,
        "instruction": "Place the object to the right.",
        "place_box": [0.0] * 7,
        "place_box_gt": [1.0] * 7,
        "visualization_png": str(prediction_png),
        "pred_heatmap_ply": str(heatmap_ply),
        "decoder_stages": [
            {
                "stage": stage_index,
                "best_place_score": 0.1 * stage_index,
                "placements": [{"box": [0.0] * 7, "score": 0.1, "yaw_bin": 0}],
                "visualization_png": str(stage_path),
            }
            for stage_index, stage_path in enumerate(stage_paths, start=1)
        ],
    }
    predictions_path = input_dir / "predictions.json"
    predictions_path.write_text(json.dumps([prediction]), encoding="utf-8")

    rows = collect_rows(input_dir, output_dir, {}, 64, 1, None)
    write_html(output_dir / "index.html", rows, input_dir)
    benchmark_rows = load_predictions(predictions_path)
    html_text = (output_dir / "index.html").read_text(encoding="utf-8")

    assert len(rows[0]["stages"]) == 4
    assert "SPACE-Former Decoder 四阶段可视化" in html_text
    assert '"stage": 4' in html_text
    assert "renderPage()" in html_text
    assert 'src="../decoder_1.png"' not in html_text
    assert "data-stage-images" not in html_text
    assert benchmark_rows[0]["place_box"] == prediction["place_box"]
    assert benchmark_rows[0]["place_box_gt"] == prediction["place_box_gt"]
