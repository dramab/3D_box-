import copy
import importlib.util
import json
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "enrich_labels_safe_common.py"
SPEC = importlib.util.spec_from_file_location("enrich_labels_safe_common", MODULE_PATH)
safe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = safe
SPEC.loader.exec_module(safe)

build_label_user_prompt = safe.build_label_user_prompt
build_batch_user_prompt = safe.build_batch_user_prompt
build_text_payload = safe.build_text_payload
chunk_tasks = safe.chunk_tasks
materialize_output = safe.materialize_output
parse_batch_labels = safe.parse_batch_labels
validate_rewritten_label = safe.validate_rewritten_label


def _record():
    return {
        "sample_id": "sample_0",
        "object_id": "obj_0",
        "placement_sample_id": "sample_0_obj_0_cluster_000",
        "label": "Move Red Box located at the right of Blue Can to behind Green Bowl.",
        "spatial_relation": {
            "original": {
                "relation": "the right of",
                "reference_object_id": "obj_1",
                "reference_name": "Blue Can",
            },
            "placement": {
                "relation": "behind",
                "reference_object_id": "obj_2",
                "reference_name": "Green Bowl",
            },
        },
        "untouched": {"value": 3},
    }


def test_rewrite_may_omit_secondary_source_location():
    record = _record()
    rewritten = "Place Red Box behind Green Bowl."

    errors = validate_rewritten_label(record, rewritten)

    assert errors == []
    assert "Blue Can" not in rewritten


def test_rewrite_accepts_natural_target_relation_paraphrase():
    record = _record()
    rewritten = "Position Red Box at the back of Green Bowl."

    errors = validate_rewritten_label(record, rewritten)

    assert errors == []


def test_rewrite_preserves_each_composite_target_direction():
    record = _record()
    record["spatial_relation"]["placement"]["relation"] = "the back left of"

    valid = "Place Red Box behind and to the left of Green Bowl."
    missing_left = "Place Red Box behind Green Bowl."

    assert validate_rewritten_label(record, valid) == []
    assert any("left" in error for error in validate_rewritten_label(record, missing_left))


def test_rewrite_rejects_missing_core_placement_information():
    record = _record()
    missing_object = "Place the item behind Green Bowl."
    missing_reference = "Move Red Box behind the target object."
    missing_relation = "Move Red Box to Green Bowl."

    assert any("移动物体" in error for error in validate_rewritten_label(record, missing_object))
    assert any("目标参考物体" in error for error in validate_rewritten_label(record, missing_reference))
    assert any("目标空间关系" in error for error in validate_rewritten_label(record, missing_relation))


def test_rewrite_rejects_exact_copy_and_multiple_sentences():
    record = _record()
    exact_errors = validate_rewritten_label(record, record["label"])
    multi_errors = validate_rewritten_label(record, "Move Red Box. Place it behind Green Bowl.")

    assert any("完全相同" in error for error in exact_errors)
    assert any("一个以句号或问号结尾的句子" in error for error in multi_errors)


def test_rewrite_accepts_a_single_request_question():
    rewritten = "Could you place Red Box behind Green Bowl?"

    assert validate_rewritten_label(_record(), rewritten) == []


def test_label_prompt_only_supplies_original_label_as_sample_content():
    prompt = build_label_user_prompt(
        _record(),
        feedback=None,
        variation_route=safe.VARIATION_ROUTES[2],
    )

    assert "input_image" not in prompt
    assert "crop" not in prompt.lower()
    assert "appearance" not in prompt.lower()
    assert "Protected semantic slots" not in prompt
    assert "<MOVING_OBJECT>" not in prompt
    assert prompt.count(_record()["label"]) == 1
    assert "may simplify or omit secondary source-location details" in prompt
    assert "Request question" in prompt
    assert "rather than only replacing Move with a synonym" in prompt


def test_batch_prompt_assigns_stable_distinct_expression_routes():
    tasks = [
        safe.Task("test", index, _record(), f"sample_{index}", f"task-{index}")
        for index in range(3)
    ]

    prompt = build_batch_user_prompt(tasks, feedback={})

    assert safe.VARIATION_ROUTES[0] in prompt
    assert safe.VARIATION_ROUTES[1] in prompt
    assert safe.VARIATION_ROUTES[2] in prompt
    assert "avoiding repeated openings and sentence frames" in prompt


def test_batch_label_parser_requires_exact_order_and_schema():
    tasks = [
        safe.Task("test", 0, _record(), "sample_0", "task-0"),
        safe.Task("test", 1, _record(), "sample_1", "task-1"),
    ]
    response = json.dumps({
        "labels": [
            {
                "index": 0,
                "label": "Place Red Box behind Green Bowl.",
            },
            {
                "index": 1,
                "label": "Position Red Box at the back of Green Bowl.",
            },
        ]
    })

    parsed = parse_batch_labels(response, tasks)

    assert list(parsed) == [0, 1]
    assert parsed[0].startswith("Place")


def test_deepseek_chat_completion_payload():
    payload = build_text_payload("chat_completions", "system", "user", max_tokens=256)

    assert safe.BASE_URL == "https://api.deepseek.com"
    assert safe.MODEL == "deepseek-v4-flash"
    assert safe.API_STYLE == "chat_completions"
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["max_tokens"] == 256
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert "store" not in payload


def test_chunk_tasks_uses_requested_batch_size():
    tasks = [safe.Task("test", index, _record(), f"sample_{index}", f"task-{index}") for index in range(5)]

    batches = chunk_tasks(tasks, 2)

    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_materialized_output_only_changes_label(tmp_path):
    record = _record()
    task = safe.Task("test", 0, record, "sample_0", "task-key")
    output_path = tmp_path / "all_labels_enriched.json"
    materialize_output(
        output_path,
        [record],
        [task],
        {"task-key": "Please relocate Red Box located at the right of Blue Can to behind Green Bowl."},
        limited=False,
    )

    enriched = json.loads(output_path.read_text())[0]
    expected = copy.deepcopy(record)
    expected["label"] = enriched["label"]
    assert enriched == expected
