#!/usr/bin/env python3
"""并发安全去模板化 placement label，不发送图片给大模型。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import ssl
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    input_label_json: str
    input_rgb_image_dir: str
    output_dir: str
    run_mode: str


BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
REASONING_EFFORT = "low"
API_STYLE = "chat_completions"
API_KEY = "sk-1e34c8e078b34d7ab0c2bb4f6f6b3189"
STORE_RESPONSES = False
EXTRA_HEADERS: Dict[str, str] = {}
AUTH_JSON_PATHS = [
    str(Path.home() / ".codex" / "auth.json"),
    str(Path(__file__).resolve().parent / "auth.json"),
    str(Path.cwd() / "auth.json"),
]

REQUEST_TIMEOUT_SECONDS = 180
MAX_API_ATTEMPTS = 6
MAX_GENERATION_ATTEMPTS = 4
RETRY_BASE_SECONDS = 2.0
REQUEST_INTERVAL_SECONDS = 0.0
MAX_WORKERS = 10
MAX_LABEL_OUTPUT_TOKENS = 180
MAX_BATCH_OUTPUT_TOKENS = 4096
VERIFY_SSL = True

OUTPUT_FILENAME = "all_labels_enriched.json"
STATE_FILENAME = ".safe_enrichment_state.json"
SAVE_EVERY_N_RESULTS = 500
DEFAULT_LIMIT_TOTAL: Optional[int] = None
DEFAULT_BATCH_SIZE = 8

MAX_LABEL_WORDS = 40
MAX_LABEL_EXTRA_WORDS = 6

TARGET_RELATION_TERMS = {
    "left": {"left"},
    "right": {"right"},
    "front": {"front", "ahead"},
    "back": {"back", "behind"},
    "behind": {"back", "behind"},
    "top": {"top", "above", "over", "atop"},
    "below": {"below", "under", "beneath"},
}
ENRICHMENT_POLICY_VERSION = "natural_text_rewrite_v8_structural_variety"

# 按标签索引稳定轮换句法路线，避免批量生成收敛为同一个高频句框。
VARIATION_ROUTES = (
    "Direct imperative: give a concise placement command without copying the original Move ... located at ... to ... frame.",
    "Polite directive: use a natural courteous instruction, such as a please construction, and end with a period.",
    "Request question: write a grammatical modal request question and end with a question mark.",
    "Desired end state: make the moving object the subject and state where it should end up.",
    "Destination first: begin with the target location, then tell the reader which object to place there.",
    "Spot selection: tell the reader to choose or use a spot having the target relation for the moving object.",
    "Result construction: ask the reader to arrange the moving object so that it has the target relation.",
    "Placement requirement: express where the moving object needs to be placed without using a direct Move command.",
)

_PRINT_LOCK = threading.Lock()


LABEL_SYSTEM_PROMPT = r"""
Rewrite automatically generated object-placement labels as varied, natural English instructions.

Return JSON only using the schema requested by the user.

Rules:
1. Understand the original label as a complete instruction before rewriting it.
2. Preserve the main placement intent: the object being moved, its intended spatial relation, and the target reference object.
3. Keep the human-readable object names while correcting awkward grammar, missing articles, and rigid template-like phrasing.
4. You may freely reorganize the sentence and may simplify or omit secondary source-location details when they are unnecessary for expressing the main placement action.
5. Do not add objects, attributes, spatial relations, distances, actions, or scene details that are unsupported by the original label.
6. Follow the expression route assigned to each item. Change the clause structure, not merely the first verb, and do not collapse different routes into the same "Move X to ..." frame.
7. Across a batch, vary openings and sentence architecture. Repeated or similar original labels must still be worded independently.
8. Prefer ordinary, idiomatic English over forced synonyms. Neutral wording such as "a spot", "somewhere", or polite request language is allowed, but it must not introduce a new spatial constraint.
9. Keep every component of a compound target relation explicit, such as both "front" and "left" for "front left".
10. Do not mention labels, annotations, images, coordinates, bounding boxes, or visualizations.
11. Produce exactly one concise English instruction ending with one period or one question mark.
12. Do not return explanations or Markdown.
""".strip()


@dataclass(frozen=True)
class Task:
    dataset: str
    index: int
    record: Dict[str, Any]
    sample_id: str
    key: str


@dataclass
class Result:
    task: Task
    enriched_label: str
    used_fallback: bool
    attempts: int
    error: Optional[str] = None


def log(message: str) -> None:
    with _PRINT_LOCK:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)


def json_dump_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_api_key_from_auth_json(paths: Sequence[str]) -> Tuple[str, Optional[Path]]:
    seen = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            payload = load_json(path)
        except Exception as exc:
            log(f"警告：无法读取 auth.json {path}: {exc}")
            continue
        value = payload.get("OPENAI_API_KEY") if isinstance(payload, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip(), path
    return "", None


def normalize_base_url(url: str) -> str:
    return url.strip().rstrip("/")


def endpoint_for(base_url: str, style: str) -> str:
    suffix = "/chat/completions" if style == "chat_completions" else "/responses"
    return base_url if base_url.endswith(suffix) else base_url + suffix


def get_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context() if VERIFY_SSL else ssl._create_unverified_context()  # noqa: SLF001


def label_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


def count_words(text: str) -> int:
    return len(label_tokens(text))


def extract_constraints(record: Dict[str, Any]) -> Dict[str, str]:
    label = record.get("label")
    spatial = record.get("spatial_relation")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("record.label 缺失或不是字符串")
    if not isinstance(spatial, dict):
        raise ValueError("record.spatial_relation 缺失")
    original = spatial.get("original")
    placement = spatial.get("placement")
    if not isinstance(original, dict) or not isinstance(placement, dict):
        raise ValueError("spatial_relation.original/placement 缺失")

    marker = " located at "
    if marker not in label:
        raise ValueError(f"label 不含预期结构 {marker!r}")
    left = label.split(marker, 1)[0]
    moving_name = left[5:] if left.startswith("Move ") else left
    values = {
        "original_label": label.strip(),
        "moving_name": moving_name,
        "moving_object_id": str(record.get("object_id", "")),
        "source_relation": original.get("relation"),
        "source_reference": original.get("reference_name"),
        "source_reference_id": original.get("reference_object_id"),
        "target_relation": placement.get("relation"),
        "target_reference": placement.get("reference_name"),
        "target_reference_id": placement.get("reference_object_id"),
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ValueError("label 的物体名称、object_id、空间关系或 reference 字段缺失")
    return values


def make_task_key(dataset: str, index: int, record: Dict[str, Any]) -> str:
    stable = "\n".join([
        ENRICHMENT_POLICY_VERSION,
        dataset,
        str(index),
        str(record.get("placement_sample_id", "")),
        str(record.get("label", "")),
    ])
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]


class APIError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def parse_sse_events(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    events = []
    for block in re.split(r"\n\s*\n", normalized):
        event_name = ""
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise APIError(f"SSE data 不是合法 JSON: {data[:1000]}") from exc
        if isinstance(payload, dict):
            events.append((event_name, payload))
    return events


def response_has_output_text(response: Dict[str, Any]) -> bool:
    if isinstance(response.get("output_text"), str):
        return True
    for item in response.get("output", []) if isinstance(response.get("output"), list) else []:
        for part in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return True
    return False


def parse_http_api_response(text: str, content_type: str = "") -> Dict[str, Any]:
    stripped = text.lstrip("\ufeff \t\r\n")
    looks_like_sse = "text/event-stream" in content_type.lower() or stripped.startswith(("event:", "data:"))
    if not looks_like_sse:
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise APIError(f"API 返回的不是合法 JSON/SSE: {text[:1000]}") from exc
        if not isinstance(payload, dict):
            raise APIError("API JSON 顶层不是对象")
        return payload

    events = parse_sse_events(text)
    completed = None
    last_response = None
    done_texts: List[str] = []
    delta_texts: List[str] = []
    for event_name, payload in events:
        event_type = payload.get("type") or event_name
        nested = payload.get("response")
        if isinstance(nested, dict):
            last_response = nested
        if event_type == "response.failed":
            raise APIError(f"Responses API 失败: {json.dumps(payload, ensure_ascii=False)[:1500]}")
        if event_type == "response.completed" and isinstance(nested, dict):
            completed = nested
        elif event_type == "response.output_text.done" and isinstance(payload.get("text"), str):
            done_texts.append(payload["text"])
        elif event_type == "response.output_text.delta" and isinstance(payload.get("delta"), str):
            delta_texts.append(payload["delta"])
    if completed is not None and response_has_output_text(completed):
        return completed
    if done_texts:
        return {"output_text": "\n".join(done_texts)}
    if delta_texts:
        return {"output_text": "".join(delta_texts)}
    if completed is not None:
        return completed
    if last_response is not None:
        return last_response
    raise APIError("SSE 中没有找到模型输出")


def post_json(url: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "safe-visual-label-enricher/2.0",
    }
    headers.update(EXTRA_HEADERS)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS, context=get_ssl_context()) as response:
            body = response.read().decode("utf-8", errors="replace")
            return parse_http_api_response(body, response.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise APIError(f"HTTP {exc.code}: {body[:1500]}", exc.code, body) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise APIError(f"网络错误: {exc}") from exc


def request_payload_with_retries(
    base_url: str,
    api_key: str,
    style: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    url = endpoint_for(base_url, style)
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = post_json(url, payload, api_key)
            if REQUEST_INTERVAL_SECONDS > 0:
                time.sleep(REQUEST_INTERVAL_SECONDS)
            return response
        except APIError as exc:
            last_error = exc
            unsupported_format = exc.status == 400 and any(
                marker in exc.body for marker in ("response_format", "text.format", "json_object")
            )
            if unsupported_format:
                payload = copy.deepcopy(payload)
                payload.pop("response_format", None)
                payload.pop("text", None)
                continue
            retryable = exc.status is None or exc.status == 429 or 500 <= exc.status < 600
            if attempt >= MAX_API_ATTEMPTS or not retryable:
                raise
            delay = RETRY_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.8)
            log(f"API 请求失败，{delay:.1f}s 后重试 ({attempt}/{MAX_API_ATTEMPTS}): {exc}")
            time.sleep(delay)
    raise RuntimeError(f"API 请求失败: {last_error}")


def extract_response_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [part.get("text") or part.get("content") for part in content if isinstance(part, dict)]
            if any(isinstance(value, str) for value in texts):
                return "\n".join(value for value in texts if isinstance(value, str))
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    texts = []
    for item in response.get("output", []) if isinstance(response.get("output"), list) else []:
        for part in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if texts:
        return "\n".join(texts)
    for key in ("text", "content", "response"):
        if isinstance(response.get(key), str):
            return response[key]
    raise ValueError(f"无法提取模型文本: {json.dumps(response, ensure_ascii=False)[:1500]}")


def parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError(f"模型没有返回 JSON object: {stripped[:800]}")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("模型 JSON 顶层不是对象")
    return payload


def build_text_payload(
    style: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_LABEL_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    if style == "chat_completions":
        return {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "thinking": {"type": "disabled"},
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    return {
        "model": MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "reasoning": {"effort": REASONING_EFFORT},
        "max_output_tokens": max_tokens,
        "text": {"format": {"type": "json_object"}},
        "store": STORE_RESPONSES,
    }


def build_label_user_prompt(
    record: Dict[str, Any],
    feedback: Optional[str],
    include_schema: bool = True,
    variation_route: Optional[str] = None,
) -> str:
    original_words = count_words(str(record["label"]))
    max_final_words = min(MAX_LABEL_WORDS, original_words + MAX_LABEL_EXTRA_WORDS)
    retry = ""
    if feedback:
        retry = f"\nPrevious output failed validation:\n{feedback}\nCorrect the output.\n"
    schema = '- Return JSON only with this schema: {"label":"..."}.'
    route = variation_route or (
        "Choose a natural sentence architecture that is clearly different from the original template."
    )
    return (
        f"Original label:\n{record['label']}\n\n"
        "Rewrite the original label into one natural English instruction.\n\n"
        f"Expression route for this item:\n- {route}\n\n"
        "Requirements:\n"
        "- Preserve the names of the object being moved and the target reference object, as well as the intended placement relation.\n"
        "- You may simplify or omit secondary source-location details when they are unnecessary.\n"
        "- Follow the assigned expression route; change the sentence structure rather than only replacing Move with a synonym.\n"
        "- Keep all directional components explicit when the target relation combines directions.\n"
        "- Do not add information unsupported by the original label.\n"
        "- Return exactly one instruction sentence ending with one period or one question mark.\n"
        f"- The rewritten label must not exceed {max_final_words} words.\n"
        + (schema if include_schema else "")
        + retry
    )


def build_batch_user_prompt(tasks: Sequence[Task], feedback: Dict[str, str]) -> str:
    sections = []
    for task in tasks:
        item_feedback = feedback.get(task.key)
        route = VARIATION_ROUTES[task.index % len(VARIATION_ROUTES)]
        prompt = build_label_user_prompt(
            task.record,
            item_feedback,
            include_schema=False,
            variation_route=route,
        )
        sections.append(f"Item index: {task.index}\n{prompt}")
    return (
        "Rewrite each item independently. Return JSON only with this schema:\n"
        '{"labels":[{"index":0,"label":"..."}]}\n\n'
        "Return exactly one labels entry for every item index below, in the same order.\n\n"
        "The assigned expression routes are mandatory. Keep semantic accuracy first, while avoiding repeated openings and sentence frames across the batch.\n\n"
        + "\n\n---\n\n".join(sections)
    )


def parse_batch_labels(text: str, tasks: Sequence[Task]) -> Dict[int, str]:
    payload = parse_json_object(text)
    if set(payload) != {"labels"} or not isinstance(payload["labels"], list):
        raise ValueError("batch label JSON 必须且只能包含 labels list")
    received = payload["labels"]
    if len(received) != len(tasks):
        raise ValueError(f"batch label 数量不一致: {len(received)} != {len(tasks)}")
    expected_indices = [task.index for task in tasks]
    parsed: Dict[int, str] = {}
    for position, item in enumerate(received):
        if not isinstance(item, dict):
            raise ValueError(f"labels[{position}] 不是对象")
        if set(item) != {"index", "label"} or not isinstance(item["label"], str):
            raise ValueError(f"labels[{position}] 字段必须为 index 和 label")
        raw_index = item["index"]
        if isinstance(raw_index, int):
            index = raw_index
        elif isinstance(raw_index, str) and raw_index.isdigit():
            index = int(raw_index)
        else:
            raise ValueError(f"labels[{position}].index 类型错误")
        if index != expected_indices[position]:
            raise ValueError(f"labels[{position}].index 顺序错误: {index} != {expected_indices[position]}")
        if index in parsed:
            raise ValueError(f"labels index 重复: {index}")
        parsed[index] = item["label"].strip()
    return parsed


def validate_rewritten_label(record: Dict[str, Any], rewritten: str) -> List[str]:
    errors = []
    if "\n" in rewritten or "\r" in rewritten:
        errors.append("改写标签必须为单行")
    if not re.fullmatch(r"[^.?\r\n]+[.?]", rewritten):
        errors.append("改写标签必须为一个以句号或问号结尾的句子")

    constraints = extract_constraints(record)
    original = constraints["original_label"]
    original_words = count_words(original)
    max_final_words = min(MAX_LABEL_WORDS, original_words + MAX_LABEL_EXTRA_WORDS)
    if count_words(rewritten) > max_final_words:
        errors.append(
            f"最终标签超过允许长度 {max_final_words} 词"
            f"（原标签 {original_words} 词 + 最多 {MAX_LABEL_EXTRA_WORDS} 词）"
        )

    rewritten_folded = rewritten.casefold()
    if constraints["moving_name"].casefold() not in rewritten_folded:
        errors.append(f"缺少移动物体名称: {constraints['moving_name']}")
    if constraints["target_reference"].casefold() not in rewritten_folded:
        errors.append(f"缺少目标参考物体名称: {constraints['target_reference']}")

    rewritten_words = set(word.casefold() for word in label_tokens(rewritten))
    target_words = set(word.casefold() for word in label_tokens(constraints["target_relation"]))
    for component, alternatives in TARGET_RELATION_TERMS.items():
        if component in target_words and not rewritten_words.intersection(alternatives):
            errors.append(f"缺少目标空间关系分量: {component}")

    if rewritten.strip().casefold() == original.strip().casefold():
        errors.append("改写标签与原标签完全相同")
    return errors


def validate_task_output(task: Task, rewritten: str) -> Tuple[List[str], str]:
    return validate_rewritten_label(task.record, rewritten), rewritten


def batch_output_token_budget(batch_size: int) -> int:
    return min(MAX_BATCH_OUTPUT_TOKENS, max(MAX_LABEL_OUTPUT_TOKENS, 128 + MAX_LABEL_OUTPUT_TOKENS * batch_size))


def call_model_for_batch(
    tasks: Sequence[Task],
    base_url: str,
    api_key: str,
    style: str,
) -> List[Result]:
    pending = list(tasks)
    feedback: Dict[str, str] = {}
    last_errors: Dict[str, str] = {}
    results: List[Result] = []
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        if not pending:
            break
        try:
            prompt = build_batch_user_prompt(pending, feedback)
            payload = build_text_payload(
                style,
                LABEL_SYSTEM_PROMPT,
                prompt,
                max_tokens=batch_output_token_budget(len(pending)),
            )
            response = request_payload_with_retries(base_url, api_key, style, payload)
            rewritten_labels = parse_batch_labels(extract_response_text(response), pending)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            index_span = f"{pending[0].index}-{pending[-1].index}" if pending else "<empty>"
            log(
                f"批量标签改写失败 {tasks[0].dataset}[{index_span}] "
                f"({attempt}/{MAX_GENERATION_ATTEMPTS}): {last_error}"
            )
            for task in pending:
                feedback[task.key] = last_error
                last_errors[task.key] = last_error
            continue

        retry_tasks = []
        for task in pending:
            rewritten = rewritten_labels[task.index]
            errors, enriched = validate_task_output(task, rewritten)
            if not errors:
                results.append(Result(task, enriched, False, attempt))
                continue
            last_error = "; ".join(errors)
            feedback[task.key] = last_error
            last_errors[task.key] = last_error
            retry_tasks.append(task)
            log(
                f"标签改写失败 {task.dataset}[{task.index}] "
                f"({attempt}/{MAX_GENERATION_ATTEMPTS}): {last_error}"
            )
        pending = retry_tasks

    for task in pending:
        last_error = last_errors.get(task.key, "未知错误")
        log(f"警告：{task.dataset}[{task.index}] 使用原始标签回退: {last_error}")
        results.append(Result(task, str(task.record["label"]), True, MAX_GENERATION_ATTEMPTS, last_error))
    return results


def choose_api_style(base_url: str) -> str:
    if API_STYLE in ("chat_completions", "responses"):
        return API_STYLE
    if API_STYLE != "auto":
        raise ValueError(f"未知 API_STYLE: {API_STYLE}")
    return "responses" if base_url.lower().endswith("/responses") else "chat_completions"


def build_tasks(
    config: DatasetConfig,
    label_json_path: Path,
    rgb_image_dir: Path,
    limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Task]]:
    if not label_json_path.is_file() or label_json_path.name != "all_labels.json":
        raise FileNotFoundError(f"INPUT_LABEL_JSON 必须指向存在的 all_labels.json: {label_json_path}")
    records = load_json(label_json_path)
    if not isinstance(records, list):
        raise TypeError("all_labels.json 顶层必须是 list")
    # 当前文本策略不读取图片；保留参数只为兼容各数据集入口和旧命令。
    _ = rgb_image_dir
    take = len(records) if limit is None else min(int(limit), len(records))
    selected = records[:take]
    tasks = []
    for index, record in enumerate(selected):
        extract_constraints(record)
        sample_id = str(record["sample_id"])
        tasks.append(Task(
            dataset=config.dataset,
            index=index,
            record=record,
            sample_id=sample_id,
            key=make_task_key(config.dataset, index, record),
        ))
    return records, tasks


def output_path_for(output_root: Path, dataset: str, output_name: str) -> Path:
    return output_root / f"auto_labels_{dataset}" / output_name


def state_path_for(output_root: Path, dataset: str) -> Path:
    return output_root / f"auto_labels_{dataset}" / STATE_FILENAME


def load_state(path: Path, restart: bool) -> Dict[str, Any]:
    empty = {"policy_version": ENRICHMENT_POLICY_VERSION, "completed": {}}
    if restart or not path.is_file():
        return empty
    try:
        state = load_json(path)
    except Exception as exc:
        log(f"警告：无法读取状态缓存，将重新处理: {path}: {exc}")
        return empty
    if not isinstance(state, dict) or state.get("policy_version") != ENRICHMENT_POLICY_VERSION:
        log("状态缓存策略版本不匹配，将重新处理。")
        return empty
    if not isinstance(state.get("completed"), dict):
        return empty
    return state


def materialize_output(
    path: Path,
    all_records: Sequence[Dict[str, Any]],
    tasks: Sequence[Task],
    completed: Dict[str, str],
    limited: bool,
) -> None:
    if limited:
        output_records = []
        for task in tasks:
            record = copy.deepcopy(task.record)
            if task.key in completed:
                record["label"] = completed[task.key]
            output_records.append(record)
    else:
        output_records = copy.deepcopy(list(all_records))
        for task in tasks:
            if task.key in completed:
                output_records[task.index]["label"] = completed[task.key]
    json_dump_atomic(path, output_records)


def chunk_tasks(tasks: Sequence[Task], batch_size: int) -> List[List[Task]]:
    return [list(tasks[index:index + batch_size]) for index in range(0, len(tasks), batch_size)]


def parse_args(config: DatasetConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="不发送图片，仅基于原始 label 文本自然改写 all_labels.json。"
    )
    parser.add_argument("--data-root", default=str(Path(config.input_label_json).expanduser().resolve().parents[1]))
    parser.add_argument("--label-json", default=config.input_label_json)
    parser.add_argument("--rgb-image-dir", default=config.input_rgb_image_dir)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-style", choices=["chat_completions", "responses", "auto"], default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT_TOTAL)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并行请求的 batch 数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每个 API 请求包含的标签数")
    parser.add_argument("--appearance-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=config.output_dir)
    parser.add_argument("--output-name", default=OUTPUT_FILENAME)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-first", type=int, default=3)
    return parser.parse_args()


def run(config: DatasetConfig) -> int:
    global MODEL, API_STYLE
    args = parse_args(config)
    if args.model:
        MODEL = args.model
    if args.api_style:
        API_STYLE = args.api_style
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须 >= 1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须 > 0")

    data_root = Path(args.data_root).expanduser().resolve()
    label_json_path = Path(args.label_json).expanduser().resolve()
    rgb_image_dir = Path(args.rgb_image_dir).expanduser().resolve()
    raw_output_root = Path(args.output_dir).expanduser()
    output_root = raw_output_root.resolve() if raw_output_root.is_absolute() else (data_root / raw_output_root).resolve()
    output_path = output_path_for(output_root, config.dataset, args.output_name)
    state_path = state_path_for(output_root, config.dataset)

    log(f"数据集: {config.dataset}")
    log(f"原始 label: {label_json_path}")
    all_records, tasks = build_tasks(config, label_json_path, rgb_image_dir, args.limit)
    log(f"检查通过：{len(tasks)} 条标签；当前策略不读取或发送图片。")

    if args.dry_run:
        for task in tasks[:max(0, args.show_first)]:
            prompt = build_label_user_prompt(
                task.record,
                feedback=None,
                variation_route=VARIATION_ROUTES[task.index % len(VARIATION_ROUTES)],
            )
            log(f"DRY-RUN 标签 {task.index}: {task.record['label']}")
            log(f"  prompt:\n{prompt}")
        log("dry-run 完成：未调用 API，也未写文件。")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    auth_key, auth_path = load_api_key_from_auth_json(AUTH_JSON_PATHS)
    api_key = (args.api_key or API_KEY or os.environ.get("OPENAI_API_KEY", "") or auth_key).strip()
    base_url = normalize_base_url(args.base_url or os.environ.get("OPENAI_BASE_URL", "") or BASE_URL)
    if not api_key:
        raise RuntimeError("缺少 API 密钥，请使用 --api-key、OPENAI_API_KEY 或 auth.json")
    if not base_url:
        raise RuntimeError("缺少 API 地址")
    if auth_path and not (args.api_key or API_KEY or os.environ.get("OPENAI_API_KEY", "")):
        log(f"已从 auth.json 读取 API 密钥: {auth_path}")
    style = choose_api_style(base_url)
    log(
        f"API: {endpoint_for(base_url, style)} | 模型: {MODEL} | "
        f"batch workers: {args.workers} | batch size: {args.batch_size}"
    )

    state = load_state(state_path, args.restart)
    completed: Dict[str, str] = state["completed"]

    def save_state() -> None:
        json_dump_atomic(state_path, state)

    try:
        pending_tasks = [task for task in tasks if task.key not in completed]
        pending_batches = chunk_tasks(pending_tasks, args.batch_size)
        log(
            f"标签缓存：已完成 {len(tasks) - len(pending_tasks)}，"
            f"待处理 {len(pending_tasks)} 条 / {len(pending_batches)} 个 batch。"
        )
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for batch in pending_batches:
                future = executor.submit(call_model_for_batch, batch, base_url, api_key, style)
                futures[future] = batch
            completed_count = 0
            next_save_count = SAVE_EVERY_N_RESULTS
            for future in as_completed(futures):
                batch_results = future.result()
                fallback_in_batch = 0
                for result in batch_results:
                    completed[result.task.key] = result.enriched_label
                    fallback_in_batch += int(result.used_fallback)
                completed_count += len(batch_results)
                total_done = len(tasks) - len(pending_tasks) + completed_count
                elapsed = max(time.time() - start_time, 1e-3)
                suffix = f"（原标签回退 {fallback_in_batch} 条）" if fallback_in_batch else ""
                log(
                    f"标签完成 {total_done}/{len(tasks)} {suffix} | "
                    f"{completed_count / elapsed * 60.0:.2f} 条/分钟"
                )
                if completed_count >= next_save_count:
                    save_state()
                    materialize_output(output_path, all_records, tasks, completed, args.limit is not None)
                    next_save_count += SAVE_EVERY_N_RESULTS
        save_state()
        materialize_output(output_path, all_records, tasks, completed, args.limit is not None)
    except KeyboardInterrupt:
        log("收到中断，正在保存状态与 enriched JSON……")
        save_state()
        materialize_output(output_path, all_records, tasks, completed, args.limit is not None)
        raise

    fallback_count = sum(completed[task.key] == task.record["label"] for task in tasks if task.key in completed)
    log(f"完成：{len(completed)}/{len(tasks)}，原标签回退 {fallback_count} 条。")
    log(f"最终标签文件: {output_path}")
    log(f"内部断点缓存: {state_path}")
    return 0


def main(config: DatasetConfig) -> None:
    try:
        raise SystemExit(run(config))
    except KeyboardInterrupt:
        log("用户中断。已完成结果已保存，下次可直接续跑。")
        raise SystemExit(130)
    except Exception as exc:
        log(f"致命错误: {type(exc).__name__}: {exc}")
        if os.environ.get("LABEL_ENRICHER_DEBUG") == "1":
            traceback.print_exc()
        raise SystemExit(1)
