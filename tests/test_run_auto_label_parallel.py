"""run_auto_label 并发合并辅助逻辑测试。"""

from __future__ import annotations

import json

from tools.run_auto_label import (
    build_filter_stats,
    find_cross_file_heatmap_output_conflicts,
    merge_filter_stats,
)


def test_merge_filter_stats_preserves_serial_key_order() -> None:
    """多进程局部统计按文件顺序合并时，key 顺序应与串行一致。"""
    target = build_filter_stats()
    first = build_filter_stats()
    second = build_filter_stats()

    first["skipped_by_reason"]["missing_heatmap_file"] = 1
    first["skipped_by_reason"]["no_original_reference"] = 2
    second["skipped_by_reason"]["no_original_reference"] = 3
    second["skipped_by_reason"]["heatmap_no_positive_points"] = 4

    merge_filter_stats(target, first)
    merge_filter_stats(target, second)

    assert target["skipped_by_reason"] == {
        "missing_heatmap_file": 1,
        "no_original_reference": 5,
        "heatmap_no_positive_points": 4,
    }
    assert list(target["skipped_by_reason"]) == [
        "missing_heatmap_file",
        "no_original_reference",
        "heatmap_no_positive_points",
    ]


def test_find_cross_file_heatmap_output_conflicts(tmp_path) -> None:
    """跨文件写同名 filtered heatmap 时应被检测出来。"""
    placements_dir = tmp_path / "placements"
    placements_dir.mkdir()
    first = placements_dir / "sample_a__placements.json"
    second = placements_dir / "sample_b__placements.json"
    payload = {
        "objects": [
            {
                "placements": [
                    {"heatmap_ply": "heatmaps/shared__heatmap.ply"},
                ]
            }
        ]
    }
    first.write_text(json.dumps(payload), encoding="utf-8")
    second.write_text(json.dumps(payload), encoding="utf-8")

    conflicts = find_cross_file_heatmap_output_conflicts([first, second], tmp_path)

    assert len(conflicts) == 1
    assert conflicts[0][1] == first
    assert conflicts[0][2] == second
