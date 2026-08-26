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

    def forward(
        self,
        batch: dict[str, Any],
        collect_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run shared source grounding followed by one direct placement query."""
        del collect_diagnostics
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
        yaw_bins = torch.argmax(yaw_logits, dim=-1)
        source_box = source_out["source_box"]
        raw_boxes = boxes_from_bottom_centers(
            bottom_centers,
            source_box[:, 3:6].clamp_min(1e-4),
            yaw_bins,
        )
        query_valid_mask = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=raw_boxes.device
        )
        # A single query is always emitted; yaw confidence is retained as its score.
        placement_logits = raw_boxes.new_full((batch_size, 1), 20.0)
        predictions = pose_nms(
            raw_boxes,
            placement_logits,
            yaw_logits,
            query_valid_mask,
            max_outputs=1,
        )
        return {
            "source_box": source_box,
            "source_feature": source_out["source_feature"],
            "query_valid_mask": query_valid_mask,
            "raw_place_boxes": raw_boxes,
            "raw_place_logits": placement_logits,
            "raw_yaw_logits": yaw_logits,
            "raw_yaw_bins": yaw_bins,
            "pred_bottom_centers": bottom_centers,
            "decoder_aux_outputs": [],
            **predictions,
        }
