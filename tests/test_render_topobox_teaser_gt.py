from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.render_topobox_teaser_gt import (
    FIGURE_SIZE,
    SLOT_SPECS,
    make_output_paths,
)


def test_slot_specs_stay_inside_teaser_canvas() -> None:
    width, height = FIGURE_SIZE

    for slot in SLOT_SPECS:
        x, y, w, h = slot.rect
        assert slot.name
        assert w > 0
        assert h > 0
        assert 0 <= x < width
        assert 0 <= y < height
        assert x + w <= width
        assert y + h <= height


def test_output_paths_use_project_output_directory() -> None:
    root = Path("/tmp/project")
    paths = make_output_paths(root)

    assert paths.output_dir == root / "outputs" / "topobox_teaser_gt"
    assert paths.figure_png == paths.output_dir / "topobox_teaser_gt.png"
    assert paths.metadata_json == paths.output_dir / "topobox_teaser_gt_metadata.json"
