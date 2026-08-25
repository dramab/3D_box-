"""
src/annotation/free_bbox/datatypes.py
-------------------------------------
free_bbox 放置标注的配置和结果数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FreeBBoxConfig:
    """
    放置搜索配置。

    所有长度单位与 canonical scene 的 unit 一致，当前 HOPE 规范数据为 cm。
    """

    voxel_size: float = 1.0
    grid_padding: float = 10.0
    safety_margin: float = 0.5
    yaw_steps: int = 24
    min_surface_area: float = 50.0
    occlusion_threshold: float = 0.3
    dbscan_eps: Optional[float] = None
    dbscan_min_samples: int = 1
    max_reps_total: Optional[int] = None
    stability_chunk_size: int = 2000
    metric_chunk_size: int = 512


@dataclass
class FreeBBoxResult:
    """单个物体的 free_bbox 放置结果。"""

    obj_id: str
    class_name: str
    original_aabb_world: np.ndarray
    placements: list
    num_raw_candidates: int = 0
    num_after_stability: int = 0
    num_after_visibility: int = 0
    num_after_occlusion: int = 0
    num_after_bottom_center: int = 0
