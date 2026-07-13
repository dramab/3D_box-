"""
LC-BGPlaceNet Stage 2 dense placement model.

Stage 2 keeps the Stage 1 source grounding path and adds a source-conditioned
dense placement field over all active voxels.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from src.models.lc_bgplacenet.stage1 import (
    CLIPTextEncoder,
    ProjectedCLIPVoxelFeatureEncoder,
    SingleQuerySourceGroundingHead,
    SparseBackboneSpconv,
    VoxelLanguageFusion,
)


def argmax_per_batch(values: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return one global argmax index for each batch item."""
    best_indices = []
    for batch_idx in range(int(batch_size)):
        idx = torch.nonzero(batch_indices == batch_idx, as_tuple=False).flatten()
        if len(idx) == 0:
            raise ValueError(f"batch item {batch_idx} has no active voxels")
        best_indices.append(idx[torch.argmax(values[idx])])
    return torch.stack(best_indices, dim=0)


def decode_place_box(
    heatmap_logits: torch.Tensor,
    bottom_offset: torch.Tensor,
    yaw_sincos: torch.Tensor,
    size_residual: torch.Tensor,
    source_size: torch.Tensor,
    voxel_centers: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    random_yaw_for_near_square: bool = False,
    yaw_sensitive_ratio: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Decode dense placement predictions into one place box per sample."""
    best_indices = argmax_per_batch(heatmap_logits, batch_indices, batch_size)
    bottom_center = voxel_centers[best_indices] + bottom_offset[best_indices]
    size_pred = source_size.clamp_min(1e-4) * torch.exp(size_residual)

    center = bottom_center.clone()
    center[:, 2] = bottom_center[:, 2] + size_pred[:, 2] * 0.5

    yaw_vec = F.normalize(yaw_sincos[best_indices], dim=-1, eps=1e-6)
    yaw = torch.atan2(yaw_vec[:, 0], yaw_vec[:, 1])
    if random_yaw_for_near_square:
        ratio = torch.abs(size_pred[:, 0] - size_pred[:, 1]) / torch.minimum(
            size_pred[:, 0],
            size_pred[:, 1],
        ).clamp_min(1e-6)
        near_square = ratio < float(yaw_sensitive_ratio)
        if torch.any(near_square):
            yaw[near_square] = torch.empty_like(yaw[near_square]).uniform_(-torch.pi, torch.pi)

    return {
        "best_indices": best_indices,
        "bottom_center": bottom_center,
        "size_pred": size_pred,
        "yaw": yaw,
        "place_box": torch.cat([center, size_pred, yaw[:, None]], dim=-1),
    }


class SparsePlacementResidualBlock(nn.Module):
    """Submanifold sparse-conv residual block that preserves active voxels."""

    def __init__(self, hidden_dim: int, indice_key: str) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Any) -> Any:
        out = self.net(x)
        return out.replace_feature(self.act(out.features + x.features))


class SparsePlacementUNetNeck(nn.Module):
    """Sparse convolution neck for local voxel-context aggregation."""

    def __init__(self, hidden_dim: int, num_blocks: int = 2) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        blocks = [
            SparsePlacementResidualBlock(hidden_dim, indice_key=f"placement_neck_{idx}")
            for idx in range(max(int(num_blocks), 1))
        ]
        self.spconv = spconv
        self.net = nn.ModuleList(blocks)

    def forward(
        self,
        field_features: torch.Tensor,
        sparse_coords: torch.Tensor,
        spatial_shape: list[int],
        batch_size: int,
    ) -> torch.Tensor:
        # The collate function already aligns sparse_coords with field_features row order.
        x = self.spconv.SparseConvTensor(
            features=field_features,
            indices=sparse_coords.int(),
            spatial_shape=spatial_shape,
            batch_size=int(batch_size),
        )
        for block in self.net:
            x = block(x)
        return x.features


class SourceConditionedDensePlacementField(nn.Module):
    """Predict dense placement heatmap, offset, yaw and size residual."""

    def __init__(self, hidden_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        dropout = float(cfg.get("dropout", 0.1))
        self.size_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.source_condition_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.placement_fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.sparse_neck = SparsePlacementUNetNeck(
            hidden_dim=hidden_dim,
            num_blocks=int(cfg.get("neck_num_blocks", 2)),
        )
        self.heatmap_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.offset_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.yaw_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.size_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        voxel_features: torch.Tensor,
        coords_norm: torch.Tensor,
        sparse_coords: torch.Tensor,
        spatial_shape: list[int],
        batch_indices: torch.Tensor,
        batch_size: int,
        source_feature: torch.Tensor,
        source_size_stage1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        size_embed = self.size_mlp(source_size_stage1.clamp_min(1e-4))
        source_condition = self.source_condition_mlp(torch.cat([source_feature, size_embed], dim=-1))
        source_per_voxel = source_condition[batch_indices]
        pos_embed = self.coord_mlp(coords_norm)
        field_features = self.placement_fusion_mlp(torch.cat([voxel_features, pos_embed, source_per_voxel], dim=-1))
        field_features = self.sparse_neck(
            field_features=field_features,
            sparse_coords=sparse_coords,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        return {
            "source_condition": source_condition,
            "placement_features": field_features,
            "placement_heatmap_logits": self.heatmap_head(field_features).squeeze(-1),
            "bottom_offset": self.offset_head(field_features),
            "yaw_sincos": self.yaw_head(field_features),
            "size_residual": self.size_head(source_condition),
        }


class LCBGPlaceNetDensePlacement(nn.Module):
    """LC-BGPlaceNet Stage 2 model with dense placement field."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        backbone_cfg = cfg["backbone"]
        if str(backbone_cfg.get("type", "spconv")).lower() != "spconv":
            raise ValueError("Only spconv backbone is implemented for Stage 2.")
        clip_cfg = cfg.get("clip", {})
        clip_model = str(clip_cfg.get("model_name_or_path", clip_cfg.get("model_name", "openai/clip-vit-base-patch16")))
        clip_local_only = bool(clip_cfg.get("local_files_only", True))
        clip_freeze = bool(clip_cfg.get("freeze", True))
        voxel_clip_dim = int(clip_cfg.get("voxel_feature_dim", 0))
        self.voxel_image_encoder = (
            ProjectedCLIPVoxelFeatureEncoder(
                model_name_or_path=clip_model,
                voxel_feature_dim=voxel_clip_dim,
                freeze=clip_freeze,
                local_files_only=clip_local_only,
            )
            if voxel_clip_dim > 0
            else None
        )
        self.backbone = SparseBackboneSpconv(
            in_channels=int(backbone_cfg.get("in_channels", 6)),
            hidden_dim=hidden_dim,
            num_blocks=int(backbone_cfg.get("num_blocks", 3)),
        )
        self.text_encoder = CLIPTextEncoder(
            model_name_or_path=clip_model,
            hidden_dim=hidden_dim,
            freeze=clip_freeze,
            max_length=int(clip_cfg.get("max_length", 77)),
            local_files_only=clip_local_only,
        )
        self.fusion = VoxelLanguageFusion(
            hidden_dim=hidden_dim,
            num_heads=int(cfg["fusion"].get("num_heads", 4)),
            dropout=float(cfg["fusion"].get("dropout", 0.1)),
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.source_grounding = SingleQuerySourceGroundingHead(
            hidden_dim=hidden_dim,
            num_heads=int(cfg["source_grounding"].get("num_heads", 4)),
            num_layers=int(cfg["source_grounding"].get("num_layers", 3)),
            dropout=float(cfg["source_grounding"].get("dropout", 0.1)),
        )
        self.dense_placement_field = SourceConditionedDensePlacementField(
            hidden_dim=hidden_dim,
            cfg=cfg.get("placement", {}),
        )
        self.yaw_sensitive_ratio = float(cfg.get("placement", {}).get("yaw_sensitive_ratio", 0.25))

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch_size = int(batch["batch_size"])
        backbone_batch = self._prepare_backbone_batch(batch)
        f_3d = self.backbone(backbone_batch)
        text_tokens, text_global, text_attention_mask = self.text_encoder(
            batch["instructions"],
            device=backbone_batch["features"].device,
        )
        f_vl = self.fusion(
            voxel_features=f_3d,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            batch_indices=batch["batch_indices"],
            batch_size=batch_size,
        )
        voxel_tokens, pos_tokens, padding_mask = self._pack_voxels(
            f_vl,
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
        source_box = source_out["source_box"]
        placement_out = self.dense_placement_field(
            voxel_features=f_vl,
            coords_norm=batch["coords_norm"],
            sparse_coords=batch["sparse_coords"],
            spatial_shape=batch["spatial_shape"],
            batch_indices=batch["batch_indices"],
            batch_size=batch_size,
            source_feature=source_out["source_feature"],
            source_size_stage1=source_box[:, 3:6],
        )
        decoded = decode_place_box(
            heatmap_logits=placement_out["placement_heatmap_logits"],
            bottom_offset=placement_out["bottom_offset"],
            yaw_sincos=placement_out["yaw_sincos"],
            size_residual=placement_out["size_residual"],
            source_size=source_box[:, 3:6],
            voxel_centers=batch["world_coords"],
            batch_indices=batch["batch_indices"],
            batch_size=batch_size,
            random_yaw_for_near_square=not self.training,
            yaw_sensitive_ratio=self.yaw_sensitive_ratio,
        )
        return {
            "source_box": source_box,
            "source_feature": source_out["source_feature"],
            **placement_out,
            **decoded,
        }

    def _prepare_backbone_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.voxel_image_encoder is None:
            return batch
        image_features = self.voxel_image_encoder(
            images=batch["images"],
            world_coords=batch["world_coords"],
            batch_indices=batch["batch_indices"],
            camera_k=batch["camera_K"],
            camera_e_w2c=batch["camera_E_w2c"],
            image_hw=batch["image_hw"],
        )
        out = dict(batch)
        out["features"] = torch.cat([batch["features"], image_features], dim=-1)
        return out

    def _pack_voxels(
        self,
        voxel_features: torch.Tensor,
        coords_norm: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = torch.bincount(batch_indices, minlength=batch_size)
        max_voxels = int(counts.max().item())
        hidden_dim = voxel_features.shape[-1]
        tokens = voxel_features.new_zeros((batch_size, max_voxels, hidden_dim))
        pos = voxel_features.new_zeros((batch_size, max_voxels, hidden_dim))
        padding_mask = torch.ones((batch_size, max_voxels), dtype=torch.bool, device=voxel_features.device)
        pos_embed = self.pos_mlp(coords_norm)
        for batch_idx in range(batch_size):
            idx = torch.nonzero(batch_indices == batch_idx, as_tuple=False).flatten()
            count = len(idx)
            if count == 0:
                continue
            tokens[batch_idx, :count] = voxel_features[idx]
            pos[batch_idx, :count] = pos_embed[idx]
            padding_mask[batch_idx, :count] = False
        return tokens, pos, padding_mask
