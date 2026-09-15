"""Single-query Direct-Box model with pose-aligned boundary refinement."""

from __future__ import annotations

from typing import Any

import torch

from src.models.lc_bgplacenet.direct_box import DirectBox1QStage2
from src.models.lc_bgplacenet.stage2 import (
    SPACEFormerDecoder,
    SparseFeaturePyramid,
    prepare_sparse_lookup_index,
)


class DirectBox1QPABRStage2(DirectBox1QStage2):
    """Use the Direct-Box prediction as the initial pose for four-layer PABR."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        hidden_dim = int(cfg["hidden_dim"])
        pabr_cfg = cfg["pabr"]
        num_layers = int(pabr_cfg.get("num_layers", 4))
        if num_layers <= 0:
            raise ValueError("PABR num_layers must be positive")
        self.voxel_size_cm = float(pabr_cfg.get("voxel_size_cm", 1.0))
        self.pyramid = SparseFeaturePyramid(hidden_dim)
        self.pabr_decoder = SPACEFormerDecoder(
            hidden_dim=hidden_dim,
            num_heads=int(pabr_cfg.get("num_heads", 8)),
            dropout=float(pabr_cfg.get("dropout", 0.1)),
            num_layers=num_layers,
            predict_placement_score=False,
        )

    def _refine_placement(
        self,
        *,
        batch: dict[str, Any],
        fused_features: torch.Tensor,
        source_out: dict[str, torch.Tensor],
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        placement_feature: torch.Tensor,
        bottom_centers: torch.Tensor,
        yaw_logits: torch.Tensor,
        query_valid_mask: torch.Tensor,
        collect_diagnostics: bool,
    ) -> tuple[
        dict[str, torch.Tensor],
        list[dict[str, torch.Tensor]],
        dict[str, torch.Tensor],
    ]:
        """Sample pose-aligned box boundaries and iteratively refine center/yaw."""
        batch_size = int(batch["batch_size"])
        source_size = source_out["source_box"][:, 3:6].clamp_min(1e-4)
        pyramid = self.pyramid(
            fused_features,
            batch["sparse_coords"],
            batch["spatial_shape"],
            batch_size,
        )
        voxel_origins = torch.floor(batch["scene_min"] / self.voxel_size_cm) * self.voxel_size_cm
        for level in pyramid:
            prepare_sparse_lookup_index(level)

        layer_outputs, sampling_logs = self.pabr_decoder(
            placement_feature,
            bottom_centers,
            query_valid_mask,
            pyramid,
            voxel_origins,
            self.voxel_size_cm,
            source_out["source_feature"],
            source_size,
            text_tokens,
            text_attention_mask,
            collect_diagnostics=collect_diagnostics,
            initial_yaw_logits=yaw_logits,
        )
        initial_output = {
            "pred_bottom_centers": bottom_centers,
            "pred_yaw_logits": yaw_logits,
            "pred_yaw_indices": torch.argmax(yaw_logits, dim=-1),
            "pred_logits": bottom_centers.new_full(query_valid_mask.shape, 20.0),
        }
        return layer_outputs[-1], [initial_output, *layer_outputs[:-1]], sampling_logs
