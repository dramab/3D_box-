"""
src/annotation/auto_label.py
----------------------------
为 free_bbox pipeline 产出的放置样本生成自然语言移动指令标注。

结合 3D 物理测距、真实 world box 上下关系判断与图像轴对齐的 world XY 俯视角度，
生成形如 "Move {object} located at {rel_a} {ref_a} to {rel_b} {ref_b}." 的描述。

几何/关系判定逻辑移植自 Spatial-Affordance/tools/auto_label.py，
底层复用本仓库 geometry / canonical 工具，不重复实现。
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.annotation.free_bbox.geometry import (
    get_bbox_corners,
    project_world,
    transform_points,
)
from src.datasets.canonical import CameraParams, ObjectInfo

# ===================== 全局配置与常量 =====================
LABEL_TEMPLATE = "Move {object_name} located at {rel_original} {ref_a_name} to {rel_placement} {ref_b_name}."

# 上下关系判定阈值：XY 足迹重叠足够大时，才把 Z 方向差异解释为上下关系
VERTICAL_FOOTPRINT_OVERLAP_RATIO = 0.50
VERTICAL_TOLERANCE_RATIO = 0.10
MIN_VERTICAL_TOLERANCE = 1.0
VERTICAL_MAX_PENETRATION_RATIO = 0.35
MAX_VERTICAL_PENETRATION = 3.0
VERTICAL_CENTER_SEPARATION_RATIO = 0.20
MIN_VERTICAL_CENTER_SEPARATION = 0.5
# 8 个水平方向均匀划分，每个方向 45° 扇区。
AXIS_DIRECTION_HALF_WIDTH_DEG = 22.5
MIN_VISIBILITY_RATIO = 0.4
MAX_OCCLUSION_RATIO = 0.5
SMALL_IMAGE_AREA_THRESHOLD = 2500
LARGE_IMAGE_AREA_THRESHOLD = 5000
MAX_REFERENCE_CANDIDATES = 3

GLOBAL_MAPPING_CACHE: Dict[str, dict] = {}


# ===================== 1. 类别映射 =====================
def get_mapping(mapping_path: Optional[str] = None) -> dict:
    """
    读取并缓存类别名到展示名的映射表。

    输入: mapping_path 为映射 JSON 路径；None 时返回空表（直接使用 class_name）。
    输出: dict，兼容扁平结构与含 'mapping' 包裹两种格式；读取失败时为空 dict。
    """
    if not mapping_path:
        return {}
    if mapping_path not in GLOBAL_MAPPING_CACHE:
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "mapping" in data:
                data = data["mapping"]
            GLOBAL_MAPPING_CACHE[mapping_path] = data
        except Exception as e:
            print(f"⚠️ 无法读取 Mapping 文件: {e}")
            GLOBAL_MAPPING_CACHE[mapping_path] = {}
    return GLOBAL_MAPPING_CACHE[mapping_path]


# ===================== 2. 空间几何计算 (真实 box + 图像轴对齐的 world XY 俯视角度法) =====================
def get_camera_aabb(corners_world: np.ndarray, E_w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将世界坐标 box 角点变换到相机坐标系并返回 AABB (cam_min, cam_max)。"""
    corners_cam = transform_points(corners_world, E_w2c)
    return corners_cam.min(axis=0), corners_cam.max(axis=0)


def get_2d_bbox(corners_world: np.ndarray, E_w2c: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将世界坐标 box 角点投影到像素坐标并返回 2D AABB (min_uv, max_uv)。"""
    corners_img, _ = project_world(corners_world, K, E_w2c)
    return corners_img.min(axis=0), corners_img.max(axis=0)


def center_distance(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> float:
    """计算两个 AABB 中心点之间的欧氏距离。"""
    center1 = (min1 + max1) / 2.0
    center2 = (min2 + max2) / 2.0
    return float(np.linalg.norm(center1 - center2))


def get_world_aabb(corners_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从世界坐标 box 角点计算 world AABB (world_min, world_max)。"""
    corners_world = np.asarray(corners_world, dtype=np.float64)
    return corners_world.min(axis=0), corners_world.max(axis=0)


def polygon_area_xy(points_xy: np.ndarray) -> float:
    """计算 XY 平面多边形面积；顶点不足 3 个时为 0。"""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.shape[0] < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def cross_xy(origin: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> float:
    """计算二维向量 origin->point_a 与 origin->point_b 的叉积。"""
    return float(
        (point_a[0] - origin[0]) * (point_b[1] - origin[1])
        - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
    )


def polygon_signed_area_xy(points_xy: np.ndarray) -> float:
    """计算 XY 多边形有向面积，逆时针为正，顺时针为负。"""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.shape[0] < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_hull_xy(corners_world: np.ndarray) -> np.ndarray:
    """计算真实 box 角点投影到 XY 平面后的逆时针凸包足迹。"""
    points = np.unique(np.asarray(corners_world, dtype=np.float64)[:, :2], axis=0)
    if points.shape[0] <= 2:
        return points

    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    lower = []
    for point in points:
        while len(lower) >= 2 and cross_xy(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross_xy(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] >= 3 and polygon_signed_area_xy(hull) < 0.0:
        hull = hull[::-1]
    return hull


def line_intersection_xy(p1: np.ndarray, p2: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """计算线段 p1->p2 与裁剪边界 e1->e2 所在直线的交点；近似平行时返回 p2。"""
    segment_vec = p2 - p1
    edge_vec = e2 - e1
    denom = segment_vec[0] * edge_vec[1] - segment_vec[1] * edge_vec[0]
    if abs(float(denom)) < 1e-12:
        return p2
    offset = e1 - p1
    t = (offset[0] * edge_vec[1] - offset[1] * edge_vec[0]) / denom
    return p1 + t * segment_vec


def is_left_of_edge(point_xy: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    """判断点是否位于逆时针凸多边形边的内侧（左侧或边上）。"""
    edge_vec = edge_end - edge_start
    point_vec = point_xy - edge_start
    cross = edge_vec[0] * point_vec[1] - edge_vec[1] * point_vec[0]
    return bool(cross >= -1e-9)


def polygon_intersection_area_xy(subject_xy: np.ndarray, clip_xy: np.ndarray) -> float:
    """用 Sutherland-Hodgman 凸多边形裁剪计算两个 XY 足迹的交集面积。"""
    subject_xy = np.asarray(subject_xy, dtype=np.float64)
    clip_xy = np.asarray(clip_xy, dtype=np.float64)
    if subject_xy.shape[0] < 3 or clip_xy.shape[0] < 3:
        return 0.0

    output = subject_xy
    for edge_idx in range(clip_xy.shape[0]):
        edge_start = clip_xy[edge_idx]
        edge_end = clip_xy[(edge_idx + 1) % clip_xy.shape[0]]
        input_polygon = output
        output = []
        if len(input_polygon) == 0:
            break

        prev_point = input_polygon[-1]
        prev_inside = is_left_of_edge(prev_point, edge_start, edge_end)
        for curr_point in input_polygon:
            curr_inside = is_left_of_edge(curr_point, edge_start, edge_end)
            if curr_inside:
                if not prev_inside:
                    output.append(line_intersection_xy(prev_point, curr_point, edge_start, edge_end))
                output.append(curr_point)
            elif prev_inside:
                output.append(line_intersection_xy(prev_point, curr_point, edge_start, edge_end))
            prev_point = curr_point
            prev_inside = curr_inside
        output = np.asarray(output, dtype=np.float64)

    return polygon_area_xy(np.asarray(output, dtype=np.float64))


def footprint_overlap_ratio(target_corners_world: np.ndarray, ref_corners_world: np.ndarray) -> float:
    """计算两个真实 box 在世界 XY 足迹上的重叠比例（交集 / 较小足迹）。"""
    target_hull = convex_hull_xy(target_corners_world)
    ref_hull = convex_hull_xy(ref_corners_world)
    inter_area = polygon_intersection_area_xy(target_hull, ref_hull)
    target_area = polygon_area_xy(target_hull)
    ref_area = polygon_area_xy(ref_hull)
    base_area = min(target_area, ref_area)
    if base_area <= 1e-12:
        return 0.0
    return float(inter_area / base_area)


def describe_vertical_relation(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    overlap_threshold: float = VERTICAL_FOOTPRINT_OVERLAP_RATIO,
    tolerance_ratio: float = VERTICAL_TOLERANCE_RATIO,
    min_tolerance: float = MIN_VERTICAL_TOLERANCE,
    max_penetration_ratio: float = VERTICAL_MAX_PENETRATION_RATIO,
    max_penetration: float = MAX_VERTICAL_PENETRATION,
    center_separation_ratio: float = VERTICAL_CENTER_SEPARATION_RATIO,
    min_center_separation: float = MIN_VERTICAL_CENTER_SEPARATION,
) -> Optional[str]:
    """
    基于真实 world box 判断上下关系，并允许有限 bbox 穿插。
    输出 "the top of"、"below" 或 None（不属于上下关系）。
    """
    if footprint_overlap_ratio(target_corners_world, ref_corners_world) < overlap_threshold:
        return None

    t_min, t_max = get_world_aabb(target_corners_world)
    r_min, r_max = get_world_aabb(ref_corners_world)
    target_height = max(float(t_max[2] - t_min[2]), 0.0)
    ref_height = max(float(r_max[2] - r_min[2]), 0.0)
    min_height = min(target_height, ref_height)
    contact_tolerance = max(float(min_tolerance), float(tolerance_ratio) * min_height)
    penetration_limit = max(
        contact_tolerance,
        min(float(max_penetration), float(max_penetration_ratio) * min_height),
    )
    center_separation = max(
        float(min_center_separation),
        float(center_separation_ratio) * min_height,
    )

    target_center_z = float((t_min[2] + t_max[2]) * 0.5)
    ref_center_z = float((r_min[2] + r_max[2]) * 0.5)
    top_penetration = max(0.0, float(r_max[2] - t_min[2]))
    below_penetration = max(0.0, float(t_max[2] - r_min[2]))
    if target_center_z > ref_center_z + center_separation and top_penetration <= penetration_limit:
        return "the top of"
    if target_center_z < ref_center_z - center_separation and below_penetration <= penetration_limit:
        return "below"
    return None


def normalize_angle_degrees(angle_deg: float) -> float:
    """将角度归一化到 [-180, 180) 区间。"""
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def describe_angle_relation(angle_deg: float, axis_half_width_deg: float = AXIS_DIRECTION_HALF_WIDTH_DEG) -> str:
    """将图像轴对齐的 world XY 俯视向量角度映射到 8 个均匀水平方向。"""
    axis_half_width_deg = float(axis_half_width_deg)
    if not (0.0 < axis_half_width_deg < 45.0):
        raise ValueError("axis_half_width_deg must be in (0, 45)")

    angle = normalize_angle_degrees(angle_deg)
    right_min = -axis_half_width_deg
    right_max = axis_half_width_deg
    front_min = 90.0 - axis_half_width_deg
    front_max = 90.0 + axis_half_width_deg
    back_min = -90.0 - axis_half_width_deg
    back_max = -90.0 + axis_half_width_deg
    left_start = 180.0 - axis_half_width_deg

    if right_min <= angle <= right_max:
        return "the right of"
    if front_min <= angle <= front_max:
        return "in front of"
    if back_min <= angle <= back_max:
        return "behind"
    if angle >= left_start or angle < -left_start:
        return "the left of"
    if right_max < angle < front_min:
        return "the front right of"
    if front_max < angle < left_start:
        return "the front left of"
    if -left_start <= angle < back_min:
        return "the back left of"
    return "the back right of"


def footprint_center_world_xy(corners_world: np.ndarray) -> np.ndarray:
    """返回真实 box 在 world XY 支撑平面足迹上的中心点。"""
    footprint = convex_hull_xy(corners_world)
    return np.asarray(footprint, dtype=np.float64).mean(axis=0)


def camera_image_axes_world_xy(E_w2c: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """返回绕 world-Z 旋转后与图像视角对齐的 world XY 单位轴。"""
    R_c2w = np.linalg.inv(np.asarray(E_w2c, dtype=np.float64))[:3, :3]
    axis_x = R_c2w[:2, 0].astype(np.float64)
    norm_x = float(np.linalg.norm(axis_x))
    if norm_x <= 1e-9:
        return None
    axis_x = axis_x / norm_x

    image_y_projection = R_c2w[:2, 1].astype(np.float64)
    axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float64)
    if float(axis_y @ image_y_projection) > 0.0:
        axis_y = -axis_y
    return axis_x, axis_y


def describe_horizontal_relation_image_aligned_world_xy(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    E_w2c: np.ndarray,
    axis_half_width_deg: float = AXIS_DIRECTION_HALF_WIDTH_DEG,
) -> Optional[str]:
    """
    在 world XY 支撑平面上，使用绕 world-Z 旋转到图像视角的 XY 轴判断 8 向水平关系。
    中心重合时无法给出可靠方向，返回 None 让上层跳过该参照物。
    """
    target_xy = footprint_center_world_xy(target_corners_world)
    ref_xy = footprint_center_world_xy(ref_corners_world)
    direction_world_xy = target_xy - ref_xy
    if float(np.linalg.norm(direction_world_xy)) <= 1e-9:
        return None

    axes = camera_image_axes_world_xy(E_w2c)
    if axes is None:
        return None
    image_x_world_xy, image_y_world_xy = axes

    direction_image_xy = np.array(
        [
            float(direction_world_xy @ image_x_world_xy),
            float(direction_world_xy @ image_y_world_xy),
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(direction_image_xy)) <= 1e-9:
        return None

    angle_deg = float(np.degrees(np.arctan2(direction_image_xy[1], direction_image_xy[0])))
    return describe_angle_relation(angle_deg, axis_half_width_deg=axis_half_width_deg)


def describe_spatial_relation(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    E_w2c: np.ndarray,
    K: np.ndarray,
) -> Optional[str]:
    """
    先用真实 world box 判断上下关系；不是上下时用图像轴对齐的 world XY 俯视关系判断水平关系。
    输出方向关系；中心重合且不是上下时返回 None。
    """
    vertical_relation = describe_vertical_relation(target_corners_world, ref_corners_world)
    if vertical_relation is not None:
        return vertical_relation
    return describe_horizontal_relation_image_aligned_world_xy(target_corners_world, ref_corners_world, E_w2c)


# ===================== 3. 参照物筛选 =====================
def get_object_corners_world(obj: ObjectInfo) -> np.ndarray:
    """将 ObjectInfo 的 canonical AABB 通过 pose_world 转为真实 world box 角点 (8,3)。"""
    return transform_points(get_bbox_corners(obj.bbox3d_canonical), obj.pose_world)


def build_reference_projection_info(
    reference_objects: List[ObjectInfo],
    E_w2c: np.ndarray,
    K: np.ndarray,
) -> Dict[str, dict]:
    """预计算参照物的真实角点、2D bbox 和相机深度，供筛选与遮挡检测复用。"""
    obj_info_map = {}
    for obj in reference_objects:
        corners_world = get_object_corners_world(obj)
        min_2d, max_2d = get_2d_bbox(corners_world, E_w2c, K)
        depth = get_camera_aabb(corners_world, E_w2c)[0][2]
        obj_info_map[obj.obj_id] = {
            "corners_world": corners_world,
            "min_2d": min_2d,
            "max_2d": max_2d,
            "depth": depth,
            "obj": obj,
        }
    return obj_info_map


def compute_projected_box_area(min_2d: np.ndarray, max_2d: np.ndarray) -> float:
    """计算像素坐标 2D bbox 面积。"""
    box_w = max(0.0, float(max_2d[0] - min_2d[0]))
    box_h = max(0.0, float(max_2d[1] - min_2d[1]))
    return box_w * box_h


def compute_image_intersection_area(min_2d: np.ndarray, max_2d: np.ndarray, img_w: int, img_h: int) -> float:
    """计算 2D bbox 与图像画幅的交集面积。"""
    inter_xmin = max(float(min_2d[0]), 0.0)
    inter_ymin = max(float(min_2d[1]), 0.0)
    inter_xmax = min(float(max_2d[0]), float(img_w))
    inter_ymax = min(float(max_2d[1]), float(img_h))
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    return inter_w * inter_h


def passes_reference_visibility_filter(
    min_2d: np.ndarray,
    max_2d: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[bool, float]:
    """检查候选参照物是否有足够大的画面内可见区域，返回 (是否通过, 完整投影面积)。"""
    area = compute_projected_box_area(min_2d, max_2d)
    inter_area = compute_image_intersection_area(min_2d, max_2d, img_w, img_h)
    if inter_area <= 0.0:
        return False, area

    area_threshold = SMALL_IMAGE_AREA_THRESHOLD if img_w < 800 else LARGE_IMAGE_AREA_THRESHOLD
    if inter_area < area_threshold:
        return False, area

    visibility_ratio = inter_area / (area + 1e-6)
    return visibility_ratio >= MIN_VISIBILITY_RATIO, area


def compute_bbox_overlap_area(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    """计算两个像素 2D bbox 的重叠面积。"""
    overlap_xmin = max(float(min_a[0]), float(min_b[0]))
    overlap_ymin = max(float(min_a[1]), float(min_b[1]))
    overlap_xmax = min(float(max_a[0]), float(max_b[0]))
    overlap_ymax = min(float(max_a[1]), float(max_b[1]))
    overlap_w = max(0.0, overlap_xmax - overlap_xmin)
    overlap_h = max(0.0, overlap_ymax - overlap_ymin)
    return overlap_w * overlap_h


def compute_occlusion_ratio(
    ref_id: str,
    ref_min_2d: np.ndarray,
    ref_max_2d: np.ndarray,
    ref_area: float,
    ref_depth: float,
    obj_info_map: Dict[str, dict],
    exclude_id: Optional[str] = None,
) -> float:
    """估计候选参照物被更靠近相机的其他参照物遮挡的比例。"""
    occluded_area = 0.0
    for other_id, other_info in obj_info_map.items():
        if other_id == ref_id or other_id == exclude_id:
            continue
        if other_info["depth"] >= ref_depth:
            continue
        occluded_area += compute_bbox_overlap_area(
            ref_min_2d,
            ref_max_2d,
            other_info["min_2d"],
            other_info["max_2d"],
        )
    return occluded_area / (ref_area + 1e-6)


def collect_reference_candidates(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: Optional[str] = None,
    apply_visibility_filters: bool = True,
) -> List[Tuple[int, float, float, ObjectInfo, str]]:
    """收集有效参照物候选；无阈值模式只排除目标自身并跳过 near 关系。"""
    E_w2c = camera.E_w2c
    K = camera.K
    img_w, img_h = camera.img_w, camera.img_h
    t_min_c, t_max_c = get_camera_aabb(target_corners_world, E_w2c)
    obj_info_map = build_reference_projection_info(reference_objects, E_w2c, K)

    valid_candidates = []
    for ref in reference_objects:
        if exclude_id is not None and ref.obj_id == exclude_id:
            continue

        ref_info = obj_info_map[ref.obj_id]
        ref_corners_world = ref_info["corners_world"]
        r_min_c, r_max_c = get_camera_aabb(ref_corners_world, E_w2c)
        if apply_visibility_filters:
            is_visible, ref_area = passes_reference_visibility_filter(
                ref_info["min_2d"], ref_info["max_2d"], img_w, img_h
            )
            if not is_visible:
                continue

            occlusion_ratio = compute_occlusion_ratio(
                ref_id=ref.obj_id,
                ref_min_2d=ref_info["min_2d"],
                ref_max_2d=ref_info["max_2d"],
                ref_area=ref_area,
                ref_depth=ref_info["depth"],
                obj_info_map=obj_info_map,
                exclude_id=exclude_id,
            )
            if occlusion_ratio >= MAX_OCCLUSION_RATIO:
                continue

        center_dist = center_distance(t_min_c, t_max_c, r_min_c, r_max_c)
        score = 1.0 / (center_dist + 1e-5)
        relation = describe_spatial_relation(target_corners_world, ref_corners_world, E_w2c, K)
        if relation is None:
            continue
        vertical_priority = 1 if relation in ("the top of", "below") else 0
        valid_candidates.append((vertical_priority, score, center_dist, ref, relation))

    valid_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return valid_candidates


def find_nearest_reference(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: Optional[str] = None,
) -> List[Tuple[ObjectInfo, str, float, str]]:
    """
    先按可见性/遮挡筛选参照物；若筛空，则取消这些阈值并按距离选参照物。

    返回 list[(ObjectInfo, relation, distance, selection_mode)]。没有可用参照物时返回空列表。
    """
    available_refs = [
        ref
        for ref in reference_objects
        if exclude_id is None or str(ref.obj_id) != str(exclude_id)
    ]
    if not available_refs:
        return []

    valid_candidates = collect_reference_candidates(
        target_corners_world,
        reference_objects,
        camera,
        exclude_id=exclude_id,
        apply_visibility_filters=True,
    )
    selection_mode = "filtered"
    if not valid_candidates:
        valid_candidates = collect_reference_candidates(
            target_corners_world,
            reference_objects,
            camera,
            exclude_id=exclude_id,
            apply_visibility_filters=False,
        )
        selection_mode = "unfiltered"
    if not valid_candidates:
        return []

    top_candidates = valid_candidates[:MAX_REFERENCE_CANDIDATES]
    return [(ref, relation, dist, selection_mode) for _, _, dist, ref, relation in top_candidates]


def build_spatial_relation_record(
    ref: ObjectInfo,
    relation: str,
    distance_cm: float,
    selection_mode: str,
    mapping_data: Optional[dict] = None,
) -> dict:
    """将参照物候选转换为结构化关系记录。"""
    if mapping_data is None:
        mapping_data = {}
    return {
        "relation": str(relation),
        "reference_object_id": str(ref.obj_id),
        "reference_class_name": str(ref.class_name),
        "reference_name": mapping_data.get(ref.class_name, ref.class_name),
        "distance_cm": None if not np.isfinite(distance_cm) else float(distance_cm),
        "reference_selection_mode": str(selection_mode),
    }


def calculate_spatial_relation_records(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: Optional[str] = None,
    mapping_data: Optional[dict] = None,
) -> List[dict]:
    """计算目标 box 与候选参照物的结构化空间关系记录。"""
    if mapping_data is None:
        mapping_data = {}
    all_candidates = find_nearest_reference(target_corners_world, reference_objects, camera, exclude_id)
    return [
        build_spatial_relation_record(ref, relation, distance, selection_mode, mapping_data)
        for ref, relation, distance, selection_mode in all_candidates
    ]


# ===================== 4. 标注生成 =====================
def generate_label_for_placement(
    obj_record: dict,
    placement: dict,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    mapping_data: dict,
) -> Tuple[Optional[str], dict]:
    """
    为单个 placement 生成自然语言移动指令，并返回结构化空间关系。

    输入:
        obj_record: placements JSON 中的物体记录（含 class_name、canonical_aabb_object、original_pose_world）
        placement: 该物体的一个候选放置（含 corners_world）
        reference_objects: 场景全部参照物（ObjectInfo 列表）
        camera: CameraParams
        mapping_data: 类别名 -> 展示名映射
    输出:
        (label, spatial_relation_dict)，后者含 original / placement 两段记录；无法生成有效方向时 label 为 None。
    """
    target_obj_id = str(obj_record["object_id"])
    target_class_name = obj_record.get("class_name")
    target_object_name = mapping_data.get(target_class_name, target_class_name) if target_class_name else "the object"

    canonical_corners = get_bbox_corners(np.asarray(obj_record["canonical_aabb_object"], dtype=np.float64))
    original_corners = transform_points(
        canonical_corners, np.asarray(obj_record["original_pose_world"], dtype=np.float64)
    )
    placement_corners = np.asarray(placement["corners_world"], dtype=np.float64)

    # 1. 原始位置关系（排除目标物体自身）
    original_records = calculate_spatial_relation_records(
        original_corners, reference_objects, camera, exclude_id=target_obj_id, mapping_data=mapping_data
    )
    if not original_records:
        return None, {"skip_reason": "no_original_reference"}

    # 2. 目标放置位置关系
    placement_records = calculate_spatial_relation_records(
        placement_corners, reference_objects, camera, exclude_id=None, mapping_data=mapping_data
    )
    if not placement_records:
        return None, {"original": original_records[0], "skip_reason": "no_placement_reference"}

    original_record = None
    placement_record = None
    for original_candidate in original_records:
        for placement_candidate in placement_records:
            same_reference = (
                original_candidate["reference_object_id"] == placement_candidate["reference_object_id"]
            )
            same_relation = original_candidate["relation"] == placement_candidate["relation"]
            if same_reference and same_relation:
                continue
            original_record = original_candidate
            placement_record = placement_candidate
            break
        if original_record is not None:
            break

    if original_record is None or placement_record is None:
        return None, {
            "original": original_records[0],
            "placement": placement_records[0],
            "skip_reason": "duplicate_original_and_placement_relation",
        }

    rel_original = original_record["relation"]
    ref_a_name = original_record["reference_name"]
    rel_placement = placement_record["relation"]
    ref_b_name = placement_record["reference_name"]

    label = LABEL_TEMPLATE.format(
        object_name=target_object_name,
        rel_original=rel_original,
        ref_a_name=ref_a_name,
        rel_placement=rel_placement,
        ref_b_name=ref_b_name,
    )
    return label, {"original": original_record, "placement": placement_record}
