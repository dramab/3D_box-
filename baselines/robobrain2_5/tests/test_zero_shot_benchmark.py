from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.robobrain2_5.benchmark_zero_shot import (
    convert_prediction_row,
    yaw_radians_to_bin,
)


def _item() -> SimpleNamespace:
    return SimpleNamespace(
        item_id="item_0",
        source_box_gt=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64),
        place_box_gt=np.asarray([7, 8, 9, 4, 5, 6, 0], dtype=np.float64),
    )


def test_observed_yaw_maps_to_nearest_official_bin() -> None:
    assert yaw_radians_to_bin(math.radians(29.0)) == 2
    assert yaw_radians_to_bin(math.radians(179.0)) == 0


def test_valid_point_box_becomes_one_official_candidate() -> None:
    row = {
        "status": "ok",
        "render_box_world": [1, 2, 3, 4, 5, 6, math.radians(29.0)],
    }
    converted = convert_prediction_row(row, _item())

    assert converted["prediction_status"] == "ok"
    assert converted["placements"][0]["yaw_bin"] == 2
    assert converted["placements"][0]["box"] == row["render_box_world"]


def test_depth_failure_emits_no_candidate() -> None:
    converted = convert_prediction_row(
        {"status": "depth_failed", "render_box_world": None},
        _item(),
    )

    assert converted["prediction_status"] == "depth_failed"
    assert converted["placements"] == []
