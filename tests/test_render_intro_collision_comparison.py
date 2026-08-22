from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.robobrain2_5.zero_shot_visualization import Camera
from tools.render_intro_collision_comparison import (
    align_xy,
    camera_aligned_basis,
    object_display_name,
    stable_item_key,
)


def test_stable_item_key_ignores_content_hash() -> None:
    assert stable_item_key("omni__label_041919__0123456789abcdef") == "omni__label_041919"


def test_object_display_name_formats_dataset_class_name() -> None:
    assert object_display_name({"class_name": "white_candle"}) == "White Candle"


def test_camera_aligned_basis_matches_image_right_and_up() -> None:
    transform = np.eye(4, dtype=np.float64)
    camera = Camera(
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        width=1,
        height=1,
        e_c2w=transform,
    )
    right, image_up = camera_aligned_basis(camera)
    assert np.allclose(right, [1.0, 0.0])
    assert np.allclose(image_up, [0.0, -1.0])

    aligned = align_xy(np.array([[2.0, 3.0]], dtype=np.float64), camera)
    assert np.allclose(aligned, [[2.0, -3.0]])


def test_camera_aligned_basis_uses_projected_image_axes() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera = Camera(1.0, 1.0, 0.0, 0.0, 1, 1, transform)

    right, image_up = camera_aligned_basis(camera)
    assert float(right @ transform[:2, 0]) > 0.0
    assert float(image_up @ transform[:2, 1]) < 0.0


def test_camera_aligned_basis_does_not_mutate_extrinsics() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 2] = [-0.1, 0.8, -0.6]
    camera = Camera(1.0, 1.0, 0.0, 0.0, 1, 1, transform)
    before = camera.e_c2w.copy()

    camera_aligned_basis(camera)

    assert np.array_equal(camera.e_c2w, before)
