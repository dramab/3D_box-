"""Direct-Box 1Q target matching, loss and configuration tests."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.models.lc_bgplacenet.stage2 import NUM_YAW_BINS
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
