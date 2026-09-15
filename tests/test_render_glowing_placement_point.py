from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.render_glowing_placement_point import POINT_COLOR, add_glowing_point


def test_add_glowing_point_changes_only_local_region() -> None:
    image = Image.new("RGB", (100, 80), color=(255, 255, 255))
    rendered = add_glowing_point(image, (50, 40), glow_radius=20, point_radius=4)
    pixels = np.asarray(rendered)

    assert tuple(pixels[40, 50]) == POINT_COLOR
    assert tuple(pixels[0, 0]) == (255, 255, 255)
    assert pixels[40, 65, 2] > pixels[40, 65, 0]


def test_add_glowing_point_rejects_invalid_geometry() -> None:
    image = Image.new("RGB", (20, 20))
    with pytest.raises(ValueError, match="inside"):
        add_glowing_point(image, (21, 10))
    with pytest.raises(ValueError, match="smaller"):
        add_glowing_point(image, (10, 10), glow_radius=5, point_radius=5)
