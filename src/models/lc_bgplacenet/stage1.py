"""
LC-BGPlaceNet Stage 1 model.

Implements source grounding and target support region prediction from active
1cm voxel point clouds and language instructions.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def aabb_iou_3d(box_a: torch.Tensor, box_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Axis-aligned 3D IoU for boxes encoded as (cx, cy, cz, l, w, h)."""
    a_min = box_a[:, :3] - box_a[:, 3:6] * 0.5
    a_max = box_a[:, :3] + box_a[:, 3:6] * 0.5
    b_min = box_b[:, :3] - box_b[:, 3:6] * 0.5
    b_max = box_b[:, :3] + box_b[:, 3:6] * 0.5

    inter_min = torch.maximum(a_min, b_min)
    inter_max = torch.minimum(a_max, b_max)
    inter_size = (inter_max - inter_min).clamp_min(0.0)
    inter_vol = inter_size.prod(dim=-1)
    a_vol = box_a[:, 3:6].clamp_min(0.0).prod(dim=-1)
    b_vol = box_b[:, 3:6].clamp_min(0.0).prod(dim=-1)
    return inter_vol / (a_vol + b_vol - inter_vol + eps)


class SparseBackboneSpconv(nn.Module):
    """Small high-resolution sparse-conv backbone over active voxels."""

    def __init__(self, in_channels: int, hidden_dim: int, num_blocks: int = 3) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        layers: list[nn.Module] = [
            spconv.SubMConv3d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(int(num_blocks) - 1):
            layers.extend(
                [
                    spconv.SubMConv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
        self.spconv = spconv
        self.net = spconv.SparseSequential(*layers)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        x = self.spconv.SparseConvTensor(
            features=batch["features"],
            indices=batch["sparse_coords"].int(),
            spatial_shape=batch["spatial_shape"],
            batch_size=int(batch["batch_size"]),
        )
        return self.net(x).features


class TransformerTextEncoder(nn.Module):
    """Frozen HuggingFace text encoder with projection to model hidden size."""

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        freeze: bool = True,
        max_length: int = 48,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.freeze = bool(freeze)
        self.max_length = int(max_length)
        encoder_dim = int(self.encoder.config.hidden_size)
        self.proj = nn.Linear(encoder_dim, hidden_dim)

        if self.freeze:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    def forward(self, instructions: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            encoded = self.encoder(**tokens)
        token_features = self.proj(encoded.last_hidden_state)
        return token_features, tokens["attention_mask"].bool()


class W3LanguageRouting(nn.Module):
    """W³ 文本路由:三个可学习 role query 将文本 token 特征池化为三路角色表征。

    what  -> 源物体身份与几何信息,注入 source grounding head 的 query
    where -> 支撑面方位信息,FiLM 调制 support head
    whole -> 指令整体语义,透传给 Stage 2
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        # 三路 role query,顺序固定为 what / where / whole
        self.role_queries = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = text_tokens.shape[0]
        query = self.role_queries.repeat(batch_size, 1, 1)
        pooled, _ = self.attn(
            query=query,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=~text_attention_mask,
            need_weights=False,
        )
        pooled = self.norm(pooled)
        return pooled[:, 0], pooled[:, 1], pooled[:, 2]


class VoxelLanguageFusion(nn.Module):
    """Voxel-to-language cross-attention over sparse active voxel features."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        voxel_features: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        fused = torch.empty_like(voxel_features)
        key_padding_mask = ~text_attention_mask
        for batch_idx in range(batch_size):
            voxel_mask = batch_indices == batch_idx
            if not torch.any(voxel_mask):
                continue
            query = voxel_features[voxel_mask].unsqueeze(0)
            attn_out, _ = self.attn(
                query=query,
                key=text_tokens[batch_idx : batch_idx + 1],
                value=text_tokens[batch_idx : batch_idx + 1],
                key_padding_mask=key_padding_mask[batch_idx : batch_idx + 1],
                need_weights=False,
            )
            fused_one = self.norm1(query + attn_out)
            fused_one = self.norm2(fused_one + self.ffn(fused_one))
            fused[voxel_mask] = fused_one.squeeze(0)
        return fused


class SingleQuerySourceGroundingHead(nn.Module):
    """Single-query source grounding head for one movable object per instruction."""

    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.source_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.box_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_pos_embed: torch.Tensor,
        text_what: torch.Tensor,
        voxel_padding_mask: torch.Tensor,
        scene_min: torch.Tensor,
        scene_max: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = voxel_features.shape[0]
        memory = voxel_features + voxel_pos_embed
        query = self.source_query.repeat(batch_size, 1, 1)
        query = query + self.text_proj(text_what).unsqueeze(1)
        source_feature = self.decoder(
            tgt=query,
            memory=memory,
            memory_key_padding_mask=voxel_padding_mask,
        ).squeeze(1)
        raw_box = self.box_head(source_feature)
        center_raw, size_raw = torch.split(raw_box, [3, 3], dim=-1)
        center = scene_min + torch.sigmoid(center_raw) * (scene_max - scene_min).clamp_min(1e-4)
        size = F.softplus(size_raw) + 1e-4
        return {
            "source_box": torch.cat([center, size], dim=-1),
            "source_feature": source_feature,
        }


class SupportHead(nn.Module):
    """逐 voxel 支撑区域分类,叠加 CamPE 相机方位编码,并用 W³ where 向量做 FiLM 调制。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.pre = nn.Linear(hidden_dim, hidden_dim)
        # where 向量生成逐通道的 (gamma, beta) 调制系数
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        voxel_features: torch.Tensor,
        pos_embed_cam: torch.Tensor,
        where_embed: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.pre(voxel_features + pos_embed_cam)
        gamma, beta = self.film(where_embed).chunk(2, dim=-1)
        # 方位向量按 voxel 所属样本广播;(1 + gamma) 保证恒等初始化下的稳定性
        hidden = (1.0 + gamma[batch_indices]) * hidden + beta[batch_indices]
        return self.head(self.act(hidden)).squeeze(-1)


class LCBGPlaceNetStage1(nn.Module):
    """LC-BGPlaceNet Stage 1 model."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        backbone_cfg = cfg["backbone"]
        if str(backbone_cfg.get("type", "spconv")).lower() != "spconv":
            raise ValueError("Only spconv backbone is implemented for Stage 1.")
        self.backbone = SparseBackboneSpconv(
            in_channels=int(backbone_cfg.get("in_channels", 6)),
            hidden_dim=hidden_dim,
            num_blocks=int(backbone_cfg.get("num_blocks", 3)),
        )
        self.text_encoder = TransformerTextEncoder(
            model_name=str(cfg["text"]["model_name"]),
            hidden_dim=hidden_dim,
            freeze=bool(cfg["text"].get("freeze", True)),
            max_length=int(cfg["text"].get("max_length", 48)),
            local_files_only=bool(cfg["text"].get("local_files_only", True)),
        )
        self.fusion = VoxelLanguageFusion(
            hidden_dim=hidden_dim,
            num_heads=int(cfg["fusion"].get("num_heads", 4)),
            dropout=float(cfg["fusion"].get("dropout", 0.1)),
        )
        # w3 段可缺省;旧配置未显式配置时退回默认超参,保证向后兼容加载。
        w3_cfg = cfg.get("w3", {})
        self.w3 = W3LanguageRouting(
            hidden_dim=hidden_dim,
            num_heads=int(w3_cfg.get("num_heads", 8)),
            dropout=float(w3_cfg.get("dropout", 0.1)),
        )
        # CamPE:相机相关方位编码,替代原先的世界系 pos_mlp，Source Grounding 和
        # Support Head 共用同一份编码(见 forward 中的 pos_embed_cam)。
        self.pos_mlp_cam = nn.Sequential(
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
        self.support_head = SupportHead(hidden_dim)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch_size = int(batch["batch_size"])
        f_3d = self.backbone(batch)
        text_tokens, text_attention_mask = self.text_encoder(
            batch["instructions"],
            device=batch["features"].device,
        )
        # W³ 三路路由:what/where/whole 分别服务 grounding / support / Stage 2
        text_what, text_where, text_whole = self.w3(text_tokens, text_attention_mask)
        f_vl = self.fusion(
            voxel_features=f_3d,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            batch_indices=batch["batch_indices"],
            batch_size=batch_size,
        )
        # CamPE:每个 active voxel 相对该帧相机的方位编码,Source Grounding 和
        # Support Head 共用同一份,与语言指令中方位词的参照系保持一致。
        pos_embed_cam = self.pos_mlp_cam(batch["coords_cam_norm"])
        voxel_tokens, pos_tokens, padding_mask = self._pack_voxels(
            f_vl,
            pos_embed_cam,
            batch["batch_indices"],
            batch_size,
        )
        source_out = self.source_grounding(
            voxel_features=voxel_tokens,
            voxel_pos_embed=pos_tokens,
            text_what=text_what,
            voxel_padding_mask=padding_mask,
            scene_min=batch["scene_min"],
            scene_max=batch["scene_max"],
        )
        support_logits = self.support_head(f_vl, pos_embed_cam, text_where, batch["batch_indices"])
        return {
            "source_box": source_out["source_box"],
            "source_feature": source_out["source_feature"],
            "support_logits": support_logits,
            "voxel_language_features": f_vl,
            "text_whole": text_whole,
        }

    def _pack_voxels(
        self,
        voxel_features: torch.Tensor,
        pos_embed: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = torch.bincount(batch_indices, minlength=batch_size)
        max_voxels = int(counts.max().item())
        hidden_dim = voxel_features.shape[-1]
        tokens = voxel_features.new_zeros((batch_size, max_voxels, hidden_dim))
        pos = voxel_features.new_zeros((batch_size, max_voxels, hidden_dim))
        padding_mask = torch.ones((batch_size, max_voxels), dtype=torch.bool, device=voxel_features.device)
        for batch_idx in range(batch_size):
            idx = torch.nonzero(batch_indices == batch_idx, as_tuple=False).flatten()
            count = len(idx)
            if count == 0:
                continue
            tokens[batch_idx, :count] = voxel_features[idx]
            pos[batch_idx, :count] = pos_embed[idx]
            padding_mask[batch_idx, :count] = False
        return tokens, pos, padding_mask
