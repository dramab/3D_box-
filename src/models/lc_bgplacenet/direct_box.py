"""Single-query direct placement-box baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.lc_bgplacenet.stage1 import LCBGPlaceNetStage1
from src.models.lc_bgplacenet.stage2 import (
    NUM_YAW_BINS,
    boxes_from_bottom_centers,
    pose_nms,
)


class DirectBox1QStage2(LCBGPlaceNetStage1):
    """Predict one placement center and yaw from a global placement query."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        hidden_dim = int(cfg["hidden_dim"])
        direct_cfg = cfg["direct_box"]
        dropout = float(direct_cfg.get("dropout", 0.1))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=int(direct_cfg.get("num_heads", 8)),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.placement_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(direct_cfg.get("num_layers", 4)),
        )
        self.placement_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.placement_query, std=0.02)
        self.source_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.center_head = nn.Linear(hidden_dim, 3)
        self.yaw_head = nn.Linear(hidden_dim, NUM_YAW_BINS)

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
        """Return the unrefined Direct-Box pose; subclasses may add a refiner."""
        del (
            batch,
            fused_features,
            source_out,
            text_tokens,
            text_attention_mask,
            placement_feature,
            collect_diagnostics,
        )
        return (
            {
                "pred_bottom_centers": bottom_centers,
                "pred_yaw_logits": yaw_logits,
                "pred_yaw_indices": torch.argmax(yaw_logits, dim=-1),
                # A single always-valid query has no meaningful ranking target.
                "pred_logits": bottom_centers.new_full(query_valid_mask.shape, 20.0),
            },
            [],
            {},
        )

    def forward(
        self,
        batch: dict[str, Any],
        collect_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run shared source grounding followed by one direct placement query."""
        batch_size = int(batch["batch_size"])
        backbone_batch = self._prepare_backbone_batch(batch)
        voxel_features = self.backbone(backbone_batch)
        text_tokens, text_global, text_attention_mask = self.text_encoder(
            batch["instructions"], device=backbone_batch["features"].device
        )
        fused_features = self.fusion(
            voxel_features=voxel_features,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            batch_indices=batch["batch_indices"],
            batch_size=batch_size,
        )
        voxel_tokens, pos_tokens, padding_mask = self._pack_voxels(
            fused_features,
            batch["coords_norm"],
            batch["batch_indices"],
            batch_size,
        )
        source_out = self.source_grounding(
            voxel_features=voxel_tokens,
            voxel_pos_embed=pos_tokens,
            text_global=text_global,
            voxel_padding_mask=padding_mask,
            scene_min=batch["scene_min"],
            scene_max=batch["scene_max"],
        )

        query = self.placement_query.expand(batch_size, -1, -1)
        query = query + self.source_proj(source_out["source_feature"]).unsqueeze(1)
        query = query + self.text_proj(text_global).unsqueeze(1)
        placement_feature = self.placement_decoder(
            tgt=query,
            memory=voxel_tokens + pos_tokens,
            memory_key_padding_mask=padding_mask,
        )
        center_raw = self.center_head(placement_feature)
        bottom_centers = batch["scene_min"][:, None] + torch.sigmoid(center_raw) * (
            batch["scene_max"] - batch["scene_min"]
        ).clamp_min(1e-4)[:, None]
        yaw_logits = self.yaw_head(placement_feature)
        source_box = source_out["source_box"]
        query_valid_mask = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=bottom_centers.device
        )
        final, auxiliary_outputs, sampling_logs = self._refine_placement(
            batch=batch,
            fused_features=fused_features,
            source_out=source_out,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            placement_feature=placement_feature,
            bottom_centers=bottom_centers,
            yaw_logits=yaw_logits,
            query_valid_mask=query_valid_mask,
            collect_diagnostics=collect_diagnostics,
        )
        final_bottom_centers = final["pred_bottom_centers"]
        final_yaw_logits = final["pred_yaw_logits"]
        final_yaw_bins = final["pred_yaw_indices"]
        placement_logits = final["pred_logits"]
        raw_boxes = boxes_from_bottom_centers(
            final_bottom_centers,
            source_box[:, 3:6].clamp_min(1e-4),
            final_yaw_bins,
        )
        predictions = pose_nms(
            raw_boxes,
            placement_logits,
            final_yaw_logits,
            query_valid_mask,
            max_outputs=1,
        )
        return {
            "source_box": source_box,
            "source_feature": source_out["source_feature"],
            "query_valid_mask": query_valid_mask,
            "raw_place_boxes": raw_boxes,
            "raw_place_logits": placement_logits,
            "raw_yaw_logits": final_yaw_logits,
            "raw_yaw_bins": final_yaw_bins,
            "pred_bottom_centers": final_bottom_centers,
            "initial_bottom_centers": bottom_centers,
            "initial_yaw_logits": yaw_logits,
            "decoder_aux_outputs": auxiliary_outputs,
            "sampling_logs": sampling_logs,
            **predictions,
        }
