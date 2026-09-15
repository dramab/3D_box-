"""SPACE-Former Stage 2 set-prediction model.

SPACE-Former keeps the Stage 1 language/source grounding path and replaces the
dense placement field with a bounded, size-prompted set decoder.  Internally a
query tracks a placement bottom center; exported boxes use geometric centers.
"""

from __future__ import annotations

import math
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


NUM_QUERIES = 48
NUM_SAMPLES = 64
NUM_YAW_BINS = 12
MAX_OUTPUT_BOXES = 16
PYRAMID_STRIDES = (1, 2, 4)


def yaw_bin_angles(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the 12 physical yaw-bin centers in radians over [0, pi)."""
    return torch.arange(NUM_YAW_BINS, device=device, dtype=dtype) * (math.pi / NUM_YAW_BINS)


def fold_yaw_mask_24_to_12(mask: torch.Tensor) -> torch.Tensor:
    """Merge yaw and yaw+pi labels into 12 physically equivalent bins."""
    if mask.shape[-1] != 24:
        raise ValueError(f"expected a 24-bin yaw mask, got shape={tuple(mask.shape)}")
    return mask[..., :12].bool() | mask[..., 12:].bool()


def farthest_point_sample(points: torch.Tensor, count: int, seed_index: int = 0) -> torch.Tensor:
    """Deterministically sample spatially diverse point indices."""
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape (N, 3)")
    count = min(int(count), len(points))
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    selected[0] = int(seed_index)
    min_distance = torch.full((len(points),), float("inf"), device=points.device, dtype=points.dtype)
    for output_index in range(1, count):
        distance = (points - points[selected[output_index - 1]]).square().sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        selected[output_index] = torch.argmax(min_distance)
    return selected


def build_cylinder_template(
    source_size: torch.Tensor,
    voxel_size_cm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 64 rotation-invariant cylinder samples for decoder layer zero."""
    batch_size = source_size.shape[0]
    device, dtype = source_size.device, source_size.dtype
    angles16 = torch.arange(16, device=device, dtype=dtype) * (2.0 * math.pi / 16.0)
    circle16 = torch.stack([torch.cos(angles16), torch.sin(angles16)], dim=-1)
    radius = torch.linalg.vector_norm(source_size[:, :2], dim=-1) * 0.5
    dz = source_size[:, 2]
    bottom_z = -dz * 0.5 - 0.5 * float(voxel_size_cm)

    side_xy = radius[:, None, None] * circle16[None, :, :]
    side_parts = []
    for z in (bottom_z, torch.zeros_like(dz), dz * 0.5):
        side_parts.append(torch.cat([side_xy, z[:, None, None].expand(-1, 16, 1)], dim=-1))
    side = torch.cat(side_parts, dim=1)

    # One center, five inner-ring and ten outer-ring samples form the bottom disk.
    inner_angles = torch.arange(5, device=device, dtype=dtype) * (2.0 * math.pi / 5.0)
    outer_angles = torch.arange(10, device=device, dtype=dtype) * (2.0 * math.pi / 10.0)
    disk_xy = torch.cat(
        [
            torch.zeros((batch_size, 1, 2), device=device, dtype=dtype),
            radius[:, None, None] * 0.5 * torch.stack([torch.cos(inner_angles), torch.sin(inner_angles)], -1)[None],
            radius[:, None, None] * torch.stack([torch.cos(outer_angles), torch.sin(outer_angles)], -1)[None],
        ],
        dim=1,
    )
    disk = torch.cat([disk_xy, bottom_z[:, None, None].expand(-1, 16, 1)], dim=-1)
    offsets = torch.cat([side, disk], dim=1)
    if offsets.shape != (batch_size, NUM_SAMPLES, 3):
        raise RuntimeError(f"invalid cylinder template shape: {tuple(offsets.shape)}")

    is_bottom = torch.zeros((NUM_SAMPLES,), device=device, dtype=dtype)
    is_bottom[:16] = 1.0
    is_bottom[48:] = 1.0
    normalized = offsets / source_size[:, None, :].clamp_min(1e-4)
    return offsets, normalized, is_bottom


def build_box_surface_template(
    source_size: torch.Tensor,
    voxel_size_cm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 64 local Box-surface samples for oriented decoder layers."""
    batch_size = source_size.shape[0]
    device, dtype = source_size.device, source_size.dtype
    grid4 = torch.linspace(-0.5, 0.5, 4, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(grid4, grid4, indexing="ij")
    face_xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    bottom_norm = torch.cat(
        [face_xy, torch.full((16, 1), -0.5, device=device, dtype=dtype)], dim=-1
    )
    top_norm = torch.cat([face_xy, torch.full((16, 1), 0.5, device=device, dtype=dtype)], dim=-1)

    face_axis = torch.tensor([-1.0 / 6.0, 1.0 / 6.0], device=device, dtype=dtype)
    face_z = torch.tensor([-0.3, -0.1, 0.1, 0.3], device=device, dtype=dtype)
    aa, zz = torch.meshgrid(face_axis, face_z, indexing="ij")
    aa, zz = aa.flatten(), zz.flatten()
    side_norm = torch.cat(
        [
            torch.stack([torch.full_like(aa, -0.5), aa, zz], dim=-1),
            torch.stack([torch.full_like(aa, 0.5), aa, zz], dim=-1),
            torch.stack([aa, torch.full_like(aa, -0.5), zz], dim=-1),
            torch.stack([aa, torch.full_like(aa, 0.5), zz], dim=-1),
        ],
        dim=0,
    )
    normalized = torch.cat([bottom_norm, top_norm, side_norm], dim=0)[None].expand(batch_size, -1, -1).clone()
    normalized[:, :16, 2] -= 0.5 * float(voxel_size_cm) / source_size[:, None, 2].clamp_min(1e-4)
    offsets = normalized * source_size[:, None, :]
    if offsets.shape != (batch_size, NUM_SAMPLES, 3):
        raise RuntimeError(f"invalid Box template shape: {tuple(offsets.shape)}")
    is_bottom = torch.zeros((NUM_SAMPLES,), device=device, dtype=dtype)
    is_bottom[:16] = 1.0
    return offsets, normalized, is_bottom


def yaw_rotation_matrix(yaw: torch.Tensor) -> torch.Tensor:
    """Return world-Z rotation matrices for arbitrary leading yaw dimensions."""
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    zeros, ones = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        [
            cosine,
            -sine,
            zeros,
            sine,
            cosine,
            zeros,
            zeros,
            zeros,
            ones,
        ],
        dim=-1,
    ).reshape(*yaw.shape, 3, 3)


def straight_through_yaw(yaw_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode discrete yaw with hard forward values and soft backward gradients."""
    probabilities = torch.softmax(yaw_logits, dim=-1)
    indices = torch.argmax(probabilities, dim=-1)
    hard = F.one_hot(indices, NUM_YAW_BINS).to(probabilities.dtype)
    weights = hard + probabilities - probabilities.detach()
    angles = yaw_bin_angles(yaw_logits.device, yaw_logits.dtype)
    rotations = yaw_rotation_matrix(angles)
    matrix = torch.einsum("...k,kij->...ij", weights, rotations)
    return matrix, indices


def boxes_from_bottom_centers(
    bottom_centers: torch.Tensor,
    source_size: torch.Tensor,
    yaw_indices: torch.Tensor,
) -> torch.Tensor:
    """Construct `(cx, cy, cz, dx, dy, dz, yaw)` boxes from decoder state."""
    size = source_size[:, None, :].expand(-1, bottom_centers.shape[1], -1)
    centers = bottom_centers.clone()
    centers[..., 2] += size[..., 2] * 0.5
    angles = yaw_bin_angles(bottom_centers.device, bottom_centers.dtype)[yaw_indices]
    return torch.cat([centers, size, angles[..., None]], dim=-1)


class SparseResidualBlock(nn.Module):
    """Submanifold sparse residual block preserving sparse indices."""

    def __init__(self, hidden_dim: int, indice_key: str) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(hidden_dim, hidden_dim, 3, padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(hidden_dim, hidden_dim, 3, padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Any) -> Any:
        out = self.net(x)
        return out.replace_feature(self.act(out.features + x.features))


class SparseFeaturePyramid(nn.Module):
    """Build P1/P2/P3 sparse features at strides 1/2/4."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self.spconv = spconv
        self.p1_block = SparseResidualBlock(hidden_dim, "space_p1")
        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(hidden_dim, hidden_dim, 2, stride=2, bias=False, indice_key="space_down2"),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.p2_block = SparseResidualBlock(hidden_dim, "space_p2")
        self.down3 = spconv.SparseSequential(
            spconv.SparseConv3d(hidden_dim, hidden_dim, 2, stride=2, bias=False, indice_key="space_down3"),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.p3_block = SparseResidualBlock(hidden_dim, "space_p3")

    def forward(
        self,
        features: torch.Tensor,
        sparse_coords: torch.Tensor,
        spatial_shape: list[int],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        x = self.spconv.SparseConvTensor(
            features=features,
            indices=sparse_coords.int(),
            spatial_shape=spatial_shape,
            batch_size=int(batch_size),
        )
        p1 = self.p1_block(x)
        p2 = self.p2_block(self.down2(p1))
        p3 = self.p3_block(self.down3(p2))
        return [
            {"features": p1.features, "coords": p1.indices.long(), "spatial_shape": list(p1.spatial_shape), "stride": 1},
            {"features": p2.features, "coords": p2.indices.long(), "spatial_shape": list(p2.spatial_shape), "stride": 2},
            {"features": p3.features, "coords": p3.indices.long(), "spatial_shape": list(p3.spatial_shape), "stride": 4},
        ]


class RegionCrossBlock(nn.Module):
    """Update text-token queries by cross-attending to P3 memory."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(query)
        attended, _ = self.attn(normalized, memory, memory, key_padding_mask=padding_mask, need_weights=False)
        query = self.attention_norm(query + attended)
        return self.ffn_norm(query + self.ffn(query))


class TextGuidedRegionPredictor(nn.Module):
    """Predict coarse regions from P3 and all valid text tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.blocks = nn.ModuleList([RegionCrossBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.compat_query = nn.Linear(hidden_dim, hidden_dim)
        self.compat_key = nn.Linear(hidden_dim, hidden_dim)
        self.logit_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        p3_features: torch.Tensor,
        p3_batch_indices: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.bincount(p3_batch_indices, minlength=batch_size)
        max_count = int(counts.max().item())
        memory = p3_features.new_zeros((batch_size, max_count, self.hidden_dim))
        padding_mask = torch.ones((batch_size, max_count), dtype=torch.bool, device=p3_features.device)
        for batch_index in range(batch_size):
            indices = torch.nonzero(p3_batch_indices == batch_index, as_tuple=False).flatten()
            memory[batch_index, : len(indices)] = p3_features[indices]
            padding_mask[batch_index, : len(indices)] = False

        text_query = text_tokens
        for block in self.blocks:
            text_query = block(text_query, memory, padding_mask)

        q = self.compat_query(text_query).reshape(
            batch_size, text_tokens.shape[1], self.num_heads, self.head_dim
        )
        k = self.compat_key(p3_features).reshape(-1, self.num_heads, self.head_dim)
        token_compatibility = (
            (q[p3_batch_indices] * k[:, None]).sum(dim=-1).mean(dim=-1) / math.sqrt(self.head_dim)
        )
        valid_tokens = text_attention_mask[p3_batch_indices]
        token_compatibility = token_compatibility.masked_fill(
            ~valid_tokens, torch.finfo(token_compatibility.dtype).min
        )
        compatibility = torch.logsumexp(token_compatibility, dim=-1) - valid_tokens.sum(dim=-1).log()
        logits = compatibility + self.logit_mlp(torch.cat([p3_features, compatibility[:, None]], dim=-1)).squeeze(-1)
        return logits, text_query


class BoundedAnchorGenerator(nn.Module):
    """Select top P3 cells and FPS at most 32 underlying P1 anchors."""

    def __init__(self, hidden_dim: int, num_region_cells: int) -> None:
        super().__init__()
        self.num_region_cells = int(num_region_cells)
        if self.num_region_cells <= 0:
            raise ValueError("num_region_cells must be positive")
        self.pos_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        p1: dict[str, Any],
        p3: dict[str, Any],
        region_logits: torch.Tensor,
        voxel_origins: torch.Tensor,
        voxel_size_cm: float,
        scene_min: torch.Tensor,
        scene_max: torch.Tensor,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        device = p1["features"].device
        hidden_dim = p1["features"].shape[-1]
        query_features = p1["features"].new_zeros((batch_size, NUM_QUERIES, hidden_dim))
        anchors = p1["features"].new_zeros((batch_size, NUM_QUERIES, 3))
        valid_mask = torch.zeros((batch_size, NUM_QUERIES), dtype=torch.bool, device=device)
        expanded_counts = torch.zeros((batch_size,), dtype=torch.long, device=device)
        selected_cell_counts = torch.zeros((batch_size,), dtype=torch.long, device=device)

        p1_coords = p1["coords"]
        p3_coords = p3["coords"]
        p1_world = voxel_origins[p1_coords[:, 0]] + (
            p1_coords[:, 1:].to(p1["features"].dtype) + 0.5
        ) * float(voxel_size_cm)
        p1_to_p3 = torch.div(p1_coords[:, 1:], 4, rounding_mode="floor")

        for batch_index in range(batch_size):
            p3_indices = torch.nonzero(p3_coords[:, 0] == batch_index, as_tuple=False).flatten()
            selected_count = min(self.num_region_cells, len(p3_indices))
            if selected_count == 0:
                continue
            local_top = torch.topk(region_logits[p3_indices], k=selected_count).indices
            selected_p3 = p3_coords[p3_indices[local_top], 1:]
            selected_cell_counts[batch_index] = selected_count

            p1_indices = torch.nonzero(p1_coords[:, 0] == batch_index, as_tuple=False).flatten()
            membership = (p1_to_p3[p1_indices, None, :] == selected_p3[None, :, :]).all(dim=-1).any(dim=-1)
            candidates = p1_indices[membership]
            expanded_counts[batch_index] = len(candidates)
            if len(candidates) == 0:
                continue

            candidate_points = p1_world[candidates]
            fps_indices = farthest_point_sample(candidate_points, NUM_QUERIES)
            chosen = candidates[fps_indices]
            count = len(chosen)
            chosen_world = p1_world[chosen]
            extent = (scene_max[batch_index] - scene_min[batch_index]).clamp_min(1e-4)
            chosen_norm = (chosen_world - scene_min[batch_index]) / extent
            query_features[batch_index, :count] = self.query_norm(
                p1["features"][chosen] + self.pos_mlp(chosen_norm)
            )
            anchors[batch_index, :count] = chosen_world
            valid_mask[batch_index, :count] = True

        return {
            "query_features": query_features,
            "anchor_bottom_centers": anchors,
            "query_valid_mask": valid_mask,
            "expanded_p1_candidate_count": expanded_counts,
            "region_selected_cell_count": selected_cell_counts,
        }


def aggregate_sparse_mask(
    source_coords: torch.Tensor,
    source_mask: torch.Tensor,
    target_coords: torch.Tensor,
    target_spatial_shape: list[int],
    stride: int,
) -> torch.Tensor:
    """将 P1 active support 按整数 sparse key 聚合到目标尺度。"""
    if source_mask.shape != (len(source_coords),):
        raise ValueError("source_mask must align with source sparse coordinates")
    shape = torch.as_tensor(target_spatial_shape, device=target_coords.device, dtype=torch.long)
    source_coarse = torch.cat(
        [
            source_coords[:, :1].long(),
            torch.div(source_coords[:, 1:].long(), int(stride), rounding_mode="floor"),
        ],
        dim=1,
    )

    def linear_key(coords: torch.Tensor) -> torch.Tensor:
        return (
            ((coords[:, 0] * shape[0] + coords[:, 1]) * shape[1] + coords[:, 2])
            * shape[2]
            + coords[:, 3]
        )

    support_keys = linear_key(source_coarse[source_mask])
    return torch.isin(linear_key(target_coords.long()), support_keys)


def prepare_sparse_lookup_index(level: dict[str, Any]) -> None:
    """Cache sorted sparse voxel keys once for repeated decoder lookups."""
    if "lookup_sorted_keys" in level:
        return
    coords = level["coords"]
    spatial_shape = torch.as_tensor(level["spatial_shape"], device=coords.device, dtype=torch.long)
    active_keys = (
        (coords[:, 0] * spatial_shape[0] + coords[:, 1]) * spatial_shape[1] + coords[:, 2]
    ) * spatial_shape[2] + coords[:, 3]
    sorted_keys, order = torch.sort(active_keys)
    level["lookup_spatial_shape"] = spatial_shape
    level["lookup_sorted_keys"] = sorted_keys
    level["lookup_order"] = order


def sparse_nearest_lookup(
    level: dict[str, Any],
    sample_points: torch.Tensor,
    voxel_origins: torch.Tensor,
    voxel_size_cm: float,
    query_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lookup the nearest active sparse voxel for each world-space sample."""
    batch_size, num_queries, num_samples, _ = sample_points.shape
    coords = level["coords"]
    features = level["features"]
    stride = int(level["stride"])
    prepare_sparse_lookup_index(level)
    spatial_shape = level["lookup_spatial_shape"]
    batch_ids = torch.arange(batch_size, device=coords.device)[:, None, None].expand(-1, num_queries, num_samples)
    origin = voxel_origins[:, None, None, :]
    query_coords = torch.floor(
        (sample_points - origin) / (float(voxel_size_cm) * stride)
    ).long()
    in_bounds = ((query_coords >= 0) & (query_coords < spatial_shape)).all(dim=-1)
    in_bounds &= query_valid_mask[:, :, None]

    def linear_key(batch: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return ((batch * spatial_shape[0] + xyz[..., 0]) * spatial_shape[1] + xyz[..., 1]) * spatial_shape[2] + xyz[..., 2]

    sorted_keys = level["lookup_sorted_keys"]
    order = level["lookup_order"]
    flat_keys = linear_key(batch_ids, query_coords).flatten()
    positions = torch.searchsorted(sorted_keys, flat_keys)
    safe_positions = positions.clamp(max=max(len(sorted_keys) - 1, 0))
    matched = (positions < len(sorted_keys)) & (sorted_keys[safe_positions] == flat_keys)
    matched &= in_bounds.flatten()

    sampled = features.new_zeros((flat_keys.numel(), features.shape[-1]))
    # Empty boolean indexing is valid; avoiding a tensor-to-bool check removes one GPU sync per lookup.
    sampled[matched] = features[order[safe_positions[matched]]]
    return (
        sampled.reshape(batch_size, num_queries, num_samples, -1),
        matched.reshape(batch_size, num_queries, num_samples),
    )


class QuerySelfAttention(nn.Module):
    """Self-attention among the bounded placement queries."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, query: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(query)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        query = query + attended
        query = query + self.ffn(self.norm2(query))
        return query * valid_mask[..., None]


class GeometricRouting(nn.Module):
    """Pool size-aligned samples into a geometry-only query feature."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.source_condition = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.sample_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + 5, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in PYRAMID_STRIDES
            ]
        )
        self.point_attentions = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True) for _ in PYRAMID_STRIDES]
        )
        self.scale_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(
        self,
        query: torch.Tensor,
        sampled_features: list[torch.Tensor],
        sample_valid_masks: list[torch.Tensor],
        is_bottom: torch.Tensor,
        normalized_offsets: torch.Tensor,
        source_feature: torch.Tensor,
        source_size: torch.Tensor,
        query_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        condition = self.source_condition(torch.cat([source_feature, torch.log(source_size.clamp_min(1e-4))], dim=-1))
        q_geo = self.condition_norm(query + condition[:, None, :])
        q_flat = q_geo.reshape(batch_size * num_queries, 1, -1)
        scale_tokens = []
        bottom = is_bottom[None, None, :, None].expand(batch_size, num_queries, -1, -1)
        for feature, valid, projector, attention in zip(
            sampled_features, sample_valid_masks, self.sample_projectors, self.point_attentions
        ):
            metadata = torch.cat(
                [valid[..., None].to(feature.dtype), bottom, normalized_offsets], dim=-1
            )
            tokens = projector(torch.cat([feature, metadata], dim=-1)).reshape(batch_size * num_queries, NUM_SAMPLES, -1)
            pooled, _ = attention(q_flat, tokens, tokens, need_weights=False)
            scale_tokens.append(pooled.squeeze(1).reshape(batch_size, num_queries, -1))
        scales = torch.stack(scale_tokens, dim=2).reshape(batch_size * num_queries, len(PYRAMID_STRIDES), -1)
        fused, _ = self.scale_attention(q_flat, scales, scales, need_weights=False)
        geometry = fused.squeeze(1).reshape(batch_size, num_queries, -1)
        geometry = self.norm(geometry + self.ffn(geometry)) * query_valid_mask[..., None]
        return geometry


class SemanticRouting(nn.Module):
    """Cross-attend placement queries into a semantics-only feature."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        query_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attn(
            query,
            text_tokens,
            text_tokens,
            key_padding_mask=~text_attention_mask,
            need_weights=False,
        )
        semantic = self.attention_norm(query + attended)
        semantic = self.ffn_norm(semantic + self.ffn(semantic))
        semantic = semantic * query_valid_mask[..., None]
        return semantic


class FactorizedFusion(nn.Module):
    """Fuse geometry and semantics while keeping their routes explicit."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        geometry: torch.Tensor,
        semantic: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([geometry, semantic], dim=-1)))
        query = self.norm(query + gate * geometry + (1.0 - gate) * semantic)
        return query * valid_mask[..., None]


class SPACEFormerDecoderLayer(nn.Module):
    """One iterative size-prompted placement refinement layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        predict_placement_score: bool = True,
    ) -> None:
        super().__init__()
        self.self_attention = QuerySelfAttention(hidden_dim, num_heads, dropout)
        self.geometry = GeometricRouting(hidden_dim, num_heads, dropout)
        self.semantics = SemanticRouting(hidden_dim, num_heads, dropout)
        self.fusion = FactorizedFusion(hidden_dim)
        self.center_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)
        )
        self.yaw_head = nn.Linear(hidden_dim, NUM_YAW_BINS)
        self.placement_head = nn.Linear(hidden_dim, 1) if predict_placement_score else None


class SPACEFormerDecoder(nn.Module):
    """Four-layer iterative rotated multi-scale decoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int = 4,
        predict_placement_score: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SPACEFormerDecoderLayer(
                    hidden_dim,
                    num_heads,
                    dropout,
                    predict_placement_score=predict_placement_score,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        bottom_centers: torch.Tensor,
        query_valid_mask: torch.Tensor,
        pyramid: list[dict[str, Any]],
        voxel_origins: torch.Tensor,
        voxel_size_cm: float,
        source_feature: torch.Tensor,
        source_size: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        collect_diagnostics: bool = True,
        initial_yaw_logits: torch.Tensor | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        batch_size, num_queries, _ = query.shape
        if initial_yaw_logits is None:
            current_rotation = torch.eye(3, device=query.device, dtype=query.dtype)[None, None].expand(
                batch_size, num_queries, -1, -1
            )
        else:
            current_rotation, _ = straight_through_yaw(initial_yaw_logits)
        layer_outputs = []
        log_totals: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.layers):
            query = layer.self_attention(query, query_valid_mask)
            if layer_index == 0 and initial_yaw_logits is None:
                local_offsets_single, normalized_single, is_bottom = build_cylinder_template(source_size, voxel_size_cm)
            else:
                local_offsets_single, normalized_single, is_bottom = build_box_surface_template(source_size, voxel_size_cm)
            local_offsets = local_offsets_single[:, None].expand(-1, num_queries, -1, -1)
            normalized_offsets = normalized_single[:, None].expand(-1, num_queries, -1, -1)
            world_offsets = torch.einsum("bnij,bnkj->bnki", current_rotation, local_offsets)
            box_centers = bottom_centers.clone()
            box_centers[..., 2] += source_size[:, None, 2] * 0.5
            sample_points = box_centers[:, :, None, :] + world_offsets

            sampled_features, sample_valid_masks = [], []
            for scale_index, level in enumerate(pyramid):
                feature, valid = sparse_nearest_lookup(
                    level,
                    sample_points,
                    voxel_origins,
                    voxel_size_cm,
                    query_valid_mask,
                )
                sampled_features.append(feature)
                sample_valid_masks.append(valid)
                if collect_diagnostics:
                    prefix = f"layer{layer_index}_p{scale_index + 1}"
                    log_totals[f"{prefix}_sample_total_count"] = query_valid_mask.sum() * NUM_SAMPLES
                    log_totals[f"{prefix}_sample_active_count"] = valid.sum()
                    log_totals[f"{prefix}_sample_feature_nonzero_count"] = (
                        feature.abs().sum(dim=-1) > 0
                    ).sum()

            if collect_diagnostics:
                stacked_valid = torch.stack(sample_valid_masks, dim=2)
                bottom_mask = is_bottom.bool()[None, None, None, :]
                bottom_count = int(is_bottom.sum())
                log_totals[f"layer{layer_index}_bottom_sample_count"] = (
                    query_valid_mask.sum() * bottom_count * len(PYRAMID_STRIDES)
                )
                log_totals[f"layer{layer_index}_bottom_active_count"] = (
                    stacked_valid & bottom_mask
                ).sum()
                log_totals[f"layer{layer_index}_nonbottom_sample_count"] = (
                    query_valid_mask.sum() * (NUM_SAMPLES - bottom_count) * len(PYRAMID_STRIDES)
                )
                log_totals[f"layer{layer_index}_nonbottom_active_count"] = (
                    stacked_valid & ~bottom_mask
                ).sum()
                all_zero = ~stacked_valid.any(dim=(2, 3)) & query_valid_mask
                log_totals[f"layer{layer_index}_all_zero_sample_query_count"] = all_zero.sum()

            geometry = layer.geometry(
                query,
                sampled_features,
                sample_valid_masks,
                is_bottom,
                normalized_offsets,
                source_feature,
                source_size,
                query_valid_mask,
            )
            semantic = layer.semantics(
                query, text_tokens, text_attention_mask, query_valid_mask
            )
            query = layer.fusion(query, geometry, semantic, query_valid_mask)
            local_residual = torch.tanh(layer.center_head(query)) * source_size[:, None, :]
            world_residual = torch.einsum("bnij,bnj->bni", current_rotation, local_residual)
            bottom_centers = (bottom_centers + world_residual) * query_valid_mask[..., None]
            yaw_logits = layer.yaw_head(query)
            if layer.placement_head is None:
                placement_logits = query.new_full(query_valid_mask.shape, 20.0)
            else:
                placement_logits = layer.placement_head(query).squeeze(-1)
                placement_logits = placement_logits.masked_fill(~query_valid_mask, -20.0)
            current_rotation, yaw_indices = straight_through_yaw(yaw_logits)
            layer_outputs.append(
                {
                    "pred_bottom_centers": bottom_centers,
                    "pred_yaw_logits": yaw_logits,
                    "pred_yaw_indices": yaw_indices,
                    "pred_logits": placement_logits,
                }
            )
        return layer_outputs, log_totals


def pose_nms(
    raw_boxes: torch.Tensor,
    raw_logits: torch.Tensor,
    raw_yaw_logits: torch.Tensor,
    query_valid_mask: torch.Tensor,
    max_outputs: int = MAX_OUTPUT_BOXES,
    score_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Greedily remove near-duplicate center/yaw poses and keep at most 16."""
    batch_size = raw_boxes.shape[0]
    device = raw_boxes.device
    boxes = raw_boxes.new_zeros((batch_size, max_outputs, 7))
    scores_out = raw_logits.new_zeros((batch_size, max_outputs))
    yaw_bins = torch.zeros((batch_size, max_outputs), dtype=torch.long, device=device)
    valid_out = torch.zeros((batch_size, max_outputs), dtype=torch.bool, device=device)
    yaw_confidence, raw_yaw_bins = torch.sigmoid(raw_yaw_logits).max(dim=-1)
    scores = torch.sigmoid(raw_logits) * yaw_confidence
    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(query_valid_mask[batch_index], as_tuple=False).flatten()
        if len(valid_indices) == 0:
            continue
        sorted_indices = valid_indices[torch.argsort(scores[batch_index, valid_indices], descending=True)]
        above = sorted_indices[scores[batch_index, sorted_indices] >= float(score_threshold)]
        if len(above) == 0:
            above = sorted_indices[:1]
        kept: list[torch.Tensor] = []
        for candidate in above:
            duplicate = False
            for previous in kept:
                size = raw_boxes[batch_index, candidate, 3:6].clamp_min(1e-4)
                center_distance = torch.linalg.vector_norm(
                    (raw_boxes[batch_index, candidate, :3] - raw_boxes[batch_index, previous, :3]) / size
                )
                yaw_distance = torch.abs(raw_yaw_bins[batch_index, candidate] - raw_yaw_bins[batch_index, previous])
                yaw_distance = torch.minimum(yaw_distance, NUM_YAW_BINS - yaw_distance)
                if bool((center_distance < 0.25) & (yaw_distance <= 1)):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
            if len(kept) == max_outputs:
                break
        selected = torch.stack(kept) if kept else sorted_indices[:1]
        count = len(selected)
        boxes[batch_index, :count] = raw_boxes[batch_index, selected]
        scores_out[batch_index, :count] = scores[batch_index, selected]
        yaw_bins[batch_index, :count] = raw_yaw_bins[batch_index, selected]
        valid_out[batch_index, :count] = True
    return {
        "place_boxes": boxes,
        "place_scores": scores_out,
        "place_yaw_bins": yaw_bins,
        "place_valid_mask": valid_out,
        "place_box": boxes[:, 0],
    }


def postprocess_pose_predictions(
    raw_boxes: torch.Tensor,
    placement_logits: torch.Tensor,
    yaw_logits: torch.Tensor,
    query_valid_mask: torch.Tensor,
    training: bool,
) -> dict[str, torch.Tensor]:
    """Skip validation-only pose NMS during training."""
    if training:
        return {}
    return pose_nms(raw_boxes, placement_logits, yaw_logits, query_valid_mask)


class SPACEFormerStage2(nn.Module):
    """Size-Prompted Affordance and Collision Explorer."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        if hidden_dim != 256:
            raise ValueError("SPACE-Former fixed feature_dim is 256")
        backbone_cfg = cfg["backbone"]
        if str(backbone_cfg.get("type", "spconv")).lower() != "spconv":
            raise ValueError("Only spconv backbone is implemented for SPACE-Former")
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
        fusion_cfg = cfg["fusion"]
        self.fusion = VoxelLanguageFusion(
            hidden_dim=hidden_dim,
            num_heads=int(fusion_cfg.get("num_heads", 8)),
            dropout=float(fusion_cfg.get("dropout", 0.1)),
        )
        self.pos_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        source_cfg = cfg["source_grounding"]
        self.source_grounding = SingleQuerySourceGroundingHead(
            hidden_dim=hidden_dim,
            num_heads=int(source_cfg.get("num_heads", 8)),
            num_layers=int(source_cfg.get("num_layers", 4)),
            dropout=float(source_cfg.get("dropout", 0.1)),
        )
        space_cfg = cfg.get("space_former", cfg.get("placement", {}))
        num_heads = int(space_cfg.get("num_heads", 8))
        dropout = float(space_cfg.get("dropout", 0.1))
        self.voxel_size_cm = float(space_cfg.get("voxel_size_cm", 1.0))
        self.pyramid = SparseFeaturePyramid(hidden_dim)
        self.region_predictor = TextGuidedRegionPredictor(
            hidden_dim,
            num_heads,
            dropout,
            num_layers=int(space_cfg["region_cross_num_layers"]),
        )
        self.anchor_generator = BoundedAnchorGenerator(
            hidden_dim,
            num_region_cells=int(space_cfg.get("num_region_cells", 8)),
        )
        self.decoder = SPACEFormerDecoder(hidden_dim, num_heads, dropout, num_layers=4)

    def forward(
        self,
        batch: dict[str, Any],
        collect_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(batch["batch_size"])
        backbone_batch = self._prepare_backbone_batch(batch)
        f_3d = self.backbone(backbone_batch)
        text_tokens, text_global, text_attention_mask = self.text_encoder(
            batch["instructions"], device=backbone_batch["features"].device
        )
        f_vl = self.fusion(
            voxel_features=f_3d,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            batch_indices=batch["batch_indices"],
            batch_size=batch_size,
        )
        voxel_tokens, pos_tokens, padding_mask = self._pack_voxels(
            f_vl, batch["coords_norm"], batch["batch_indices"], batch_size
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
        source_size = source_box[:, 3:6].clamp_min(1e-4)
        pyramid = self.pyramid(
            f_vl,
            batch["sparse_coords"],
            batch["spatial_shape"],
            batch_size,
        )
        voxel_origins = self._voxel_origins(batch, batch_size)
        for level in pyramid:
            coords = level["coords"]
            level["world_coords"] = voxel_origins[coords[:, 0]] + (
                coords[:, 1:].to(f_vl.dtype) + 0.5
            ) * (self.voxel_size_cm * int(level["stride"]))
            prepare_sparse_lookup_index(level)
        p3_batch_indices = pyramid[2]["coords"][:, 0]
        region_logits, _ = self.region_predictor(
            pyramid[2]["features"],
            p3_batch_indices,
            text_tokens,
            text_attention_mask,
            batch_size,
        )
        anchors = self.anchor_generator(
            pyramid[0],
            pyramid[2],
            region_logits,
            voxel_origins,
            self.voxel_size_cm,
            batch["scene_min"],
            batch["scene_max"],
            batch_size,
        )
        layer_outputs, sampling_logs = self.decoder(
            anchors["query_features"],
            anchors["anchor_bottom_centers"],
            anchors["query_valid_mask"],
            pyramid,
            voxel_origins,
            self.voxel_size_cm,
            source_out["source_feature"],
            source_size,
            text_tokens,
            text_attention_mask,
            collect_diagnostics=collect_diagnostics,
        )
        final = layer_outputs[-1]
        raw_boxes = boxes_from_bottom_centers(
            final["pred_bottom_centers"], source_size, final["pred_yaw_indices"]
        )
        postprocessed = postprocess_pose_predictions(
            raw_boxes,
            final["pred_logits"],
            final["pred_yaw_logits"],
            anchors["query_valid_mask"],
            training=self.training,
        )
        return {
            "source_box": source_box,
            "source_feature": source_out["source_feature"],
            "region_logits": region_logits,
            "region_sparse_coords": pyramid[2]["coords"],
            "region_spatial_shape": pyramid[2]["spatial_shape"],
            "region_world_coords": pyramid[2]["world_coords"],
            "region_batch_indices": p3_batch_indices,
            "voxel_origins": voxel_origins,
            "anchor_bottom_centers": anchors["anchor_bottom_centers"],
            "query_valid_mask": anchors["query_valid_mask"],
            "expanded_p1_candidate_count": anchors["expanded_p1_candidate_count"],
            "region_selected_cell_count": anchors["region_selected_cell_count"],
            "raw_place_boxes": raw_boxes,
            "raw_place_logits": final["pred_logits"],
            "raw_yaw_logits": final["pred_yaw_logits"],
            "raw_yaw_bins": final["pred_yaw_indices"],
            "pred_bottom_centers": final["pred_bottom_centers"],
            "decoder_aux_outputs": layer_outputs[:-1],
            "sampling_logs": sampling_logs,
            **postprocessed,
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
        for batch_index in range(batch_size):
            indices = torch.nonzero(batch_indices == batch_index, as_tuple=False).flatten()
            tokens[batch_index, : len(indices)] = voxel_features[indices]
            pos[batch_index, : len(indices)] = pos_embed[indices]
            padding_mask[batch_index, : len(indices)] = False
        return tokens, pos, padding_mask

    def _voxel_origins(self, batch: dict[str, Any], batch_size: int) -> torch.Tensor:
        del batch_size
        return torch.floor(batch["scene_min"] / self.voxel_size_cm) * self.voxel_size_cm


# Compatibility alias for existing imports while the public model name changes.
LCBGPlaceNetDensePlacement = SPACEFormerStage2
