"""Direct-Box 1Q target matching, loss and configuration tests."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.models.lc_bgplacenet.stage2 import NUM_YAW_BINS, SPACEFormerDecoder
from src.training.lc_bgplacenet_stage2 import (
    compute_stage2_loss,
    select_nearest_direct_box_targets,
)


def _direct_box_batch() -> dict[str, torch.Tensor]:
    yaw_masks = torch.zeros(2, 3, NUM_YAW_BINS, dtype=torch.bool)
    yaw_masks[0, 0, 1] = True
    yaw_masks[0, 1, [2, 3]] = True
    yaw_masks[1, 0, 4] = True
    return {
        "gt_bottom_centers": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
                [[2.0, 3.0, 4.0], [99.0, 99.0, 99.0], [99.0, 99.0, 99.0]],
            ]
        ),
        "gt_yaw_masks": yaw_masks,
        "gt_valid_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "source_box_gt": torch.tensor(
            [[0.0, 0.0, 0.0, 2.0, 4.0, 2.0], [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]
        ),
    }


def test_direct_box_selects_nearest_valid_target() -> None:
    batch = _direct_box_batch()
    predictions = torch.tensor([[7.5, 0.0, 0.0], [2.1, 3.0, 4.0]])

    indices, centers, yaw_masks = select_nearest_direct_box_targets(predictions, batch)

    assert indices.tolist() == [1, 0]
    assert torch.equal(centers, torch.tensor([[8.0, 0.0, 0.0], [2.0, 3.0, 4.0]]))
    assert yaw_masks[0, 2]
    assert yaw_masks[0, 3]
    assert yaw_masks[1, 4]


def test_direct_box_loss_backpropagates_center_yaw_and_source() -> None:
    batch = _direct_box_batch()
    pred_centers = torch.tensor(
        [[[7.5, 0.0, 0.0]], [[2.1, 3.0, 4.0]]], requires_grad=True
    )
    yaw_logits = torch.zeros(2, 1, NUM_YAW_BINS, requires_grad=True)
    source_box = torch.tensor(
        [[0.0, 0.0, 0.0, 2.1, 3.9, 2.0], [0.0, 0.0, 0.0, 2.0, 2.1, 2.0]],
        requires_grad=True,
    )
    outputs = {
        "pred_bottom_centers": pred_centers,
        "raw_yaw_logits": yaw_logits,
        "source_box": source_box,
    }
    cfg = {
        "model": {"type": "direct_box_1q"},
        "loss": {
            "lambda_center": 5.0,
            "lambda_yaw": 0.5,
            "lambda_corner": 0.5,
            "lambda_src": 0.5,
            "source": {"lambda_center": 4.0, "lambda_size": 2.0, "lambda_iou": 0.2},
        },
    }

    losses = compute_stage2_loss(outputs, batch, cfg)
    losses["loss"].backward()

    assert losses["matched_query_count"].item() == 2
    assert losses["background_query_count"].item() == 0
    assert pred_centers.grad is not None and torch.isfinite(pred_centers.grad).all()
    assert yaw_logits.grad is not None and torch.isfinite(yaw_logits.grad).all()
    assert source_box.grad is not None and torch.isfinite(source_box.grad).all()


def test_direct_box_enriched_config_uses_mapped_split_and_labels() -> None:
    config_path = Path("configs/lc_bgplacenet_stage2_direct_box_1q_enriched.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model"]["type"] == "direct_box_1q"
    assert config["data"]["split_dir"] == "data/splits/active_aligned_enriched"
    assert all("enriched" in source["labels_path"] for source in config["data"]["sources"])


def test_direct_box_pabr_auxiliary_loss_backpropagates_to_coarse_pose() -> None:
    batch = _direct_box_batch()
    final_centers = torch.tensor(
        [[[7.5, 0.0, 0.0]], [[2.1, 3.0, 4.0]]], requires_grad=True
    )
    final_yaw = torch.zeros(2, 1, NUM_YAW_BINS, requires_grad=True)
    coarse_centers = torch.tensor(
        [[[6.0, 0.0, 0.0]], [[1.0, 3.0, 4.0]]], requires_grad=True
    )
    coarse_yaw = torch.zeros(2, 1, NUM_YAW_BINS, requires_grad=True)
    source_box = torch.tensor(
        [[0.0, 0.0, 0.0, 2.1, 3.9, 2.0], [0.0, 0.0, 0.0, 2.0, 2.1, 2.0]],
        requires_grad=True,
    )
    outputs = {
        "pred_bottom_centers": final_centers,
        "raw_yaw_logits": final_yaw,
        "source_box": source_box,
        "decoder_aux_outputs": [
            {
                "pred_bottom_centers": coarse_centers,
                "pred_yaw_logits": coarse_yaw,
            }
        ],
    }
    cfg = {
        "model": {"type": "direct_box_1q_pabr"},
        "loss": {
            "lambda_center": 5.0,
            "lambda_yaw": 0.5,
            "lambda_corner": 0.5,
            "lambda_src": 0.5,
            "lambda_aux": 0.1,
            "source": {"lambda_center": 4.0, "lambda_size": 2.0, "lambda_iou": 0.2},
        },
    }

    losses = compute_stage2_loss(outputs, batch, cfg)
    losses["loss"].backward()

    assert losses["loss_aux"].item() > 0.0
    assert coarse_centers.grad is not None and torch.isfinite(coarse_centers.grad).all()
    assert coarse_yaw.grad is not None and torch.isfinite(coarse_yaw.grad).all()


def test_pabr_single_query_decoder_omits_untrainable_score_head() -> None:
    decoder = SPACEFormerDecoder(
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        num_layers=2,
        predict_placement_score=False,
    )

    assert all(layer.placement_head is None for layer in decoder.layers)


def test_pabr_single_query_decoder_refines_from_initial_yaw() -> None:
    torch.manual_seed(7)
    decoder = SPACEFormerDecoder(
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        num_layers=2,
        predict_placement_score=False,
    )
    query = torch.randn(1, 1, 8, requires_grad=True)
    bottom_centers = torch.tensor([[[2.0, 2.0, 1.0]]], requires_grad=True)
    initial_yaw_logits = torch.zeros(1, 1, NUM_YAW_BINS, requires_grad=True)
    source_feature = torch.randn(1, 8)
    source_size = torch.tensor([[2.0, 1.0, 1.0]])
    text_tokens = torch.randn(1, 3, 8)
    text_attention_mask = torch.ones(1, 3, dtype=torch.bool)
    pyramid = [
        {
            "features": torch.randn(1, 8),
            "coords": torch.tensor([[0, 2, 2, 1]]),
            "spatial_shape": [8, 8, 8],
            "stride": stride,
        }
        for stride in (1, 2, 4)
    ]

    layer_outputs, _ = decoder(
        query=query,
        bottom_centers=bottom_centers,
        query_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        pyramid=pyramid,
        voxel_origins=torch.zeros(1, 3),
        voxel_size_cm=1.0,
        source_feature=source_feature,
        source_size=source_size,
        text_tokens=text_tokens,
        text_attention_mask=text_attention_mask,
        collect_diagnostics=False,
        initial_yaw_logits=initial_yaw_logits,
    )
    loss = layer_outputs[-1]["pred_bottom_centers"].square().mean()
    loss = loss + layer_outputs[-1]["pred_yaw_logits"].square().mean()
    loss.backward()

    assert len(layer_outputs) == 2
    assert layer_outputs[-1]["pred_bottom_centers"].shape == (1, 1, 3)
    assert torch.equal(layer_outputs[-1]["pred_logits"], torch.full((1, 1), 20.0))
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert bottom_centers.grad is not None and torch.isfinite(bottom_centers.grad).all()
    assert initial_yaw_logits.grad is not None and torch.isfinite(initial_yaw_logits.grad).all()


def test_direct_box_pabr_enriched_config_is_isolated_from_baseline() -> None:
    config_path = Path("configs/lc_bgplacenet_stage2_direct_box_1q_pabr_enriched.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model"]["type"] == "direct_box_1q_pabr"
    assert config["model"]["pabr"]["num_layers"] == 4
    assert config["loss"]["lambda_aux"] == 0.1
    assert config["training"]["output_dir"].endswith("direct_box_1q_pabr_enriched")
