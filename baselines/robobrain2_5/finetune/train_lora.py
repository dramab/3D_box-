#!/usr/bin/env python
"""Fine-tune RoboBrain2.5 for bottom-center ``(x, y, d)`` placement with LoRA.

Usage:
    conda activate brain
    torchrun --standalone --nproc_per_node=8 \
        baselines/robobrain2_5/finetune/train_lora.py \
        --config configs/robobrain2_5_sft_point_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG = PROJECT_ROOT / "configs/robobrain2_5_sft_point_lora.yaml"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune RoboBrain2.5 on bottom-center (x, y, d) supervision.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_data_manifest(
    annotations_dir: Path,
    benchmark_config_path: Path,
    expected_schema_version: str,
    expected_prompt_variant: str,
) -> None:
    """Reject annotations that were not built from the configured enriched split."""
    manifest_path = annotations_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    benchmark_cfg = load_yaml(benchmark_config_path)
    expected_split_dir = resolve_path(benchmark_cfg["data"]["split_dir"]).resolve()
    actual_config = Path(manifest["config"]).resolve()
    actual_split_dir = Path(manifest["split_dir"]).resolve()
    manifest_matches = (
        actual_config == benchmark_config_path.resolve()
        and actual_split_dir == expected_split_dir
        and manifest.get("schema_version") == expected_schema_version
        and manifest.get("prompt_variant") == expected_prompt_variant
    )
    if not manifest_matches:
        raise ValueError(
            "SFT annotations do not match the configured benchmark: "
            f"manifest_config={actual_config}, expected_config={benchmark_config_path.resolve()}, "
            f"manifest_split={actual_split_dir}, expected_split={expected_split_dir}, "
            f"manifest_schema={manifest.get('schema_version')}, expected_schema={expected_schema_version}, "
            f"manifest_prompt={manifest.get('prompt_variant')}, expected_prompt={expected_prompt_variant}"
        )


class PointSFTDataset(Dataset[dict[str, Any]]):
    """Small in-memory annotation index with lazy RGB loading."""

    def __init__(self, annotations_path: Path) -> None:
        with annotations_path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"No SFT rows found in {annotations_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            rgb = image.convert("RGB").copy()
        return {"image": rgb, "model_prompt": row["model_prompt"], "answer": row["answer"]}


def find_last_subsequence(sequence: list[int], pattern: list[int]) -> int:
    """Return the final start position of a short token pattern."""
    candidates = range(len(sequence) - len(pattern), -1, -1)
    return next((start for start in candidates if sequence[start : start + len(pattern)] == pattern), -1)


class PointSFTCollator:
    """Build multimodal batches while supervising assistant answer tokens only."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.assistant_header_ids = processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": example["model_prompt"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
            ]
            for example in examples
        ]
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in messages
        ]
        batch = self.processor(
            text=texts,
            images=[example["image"] for example in examples],
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for row_index, row in enumerate(labels):
            token_ids = row.tolist()
            header_start = find_last_subsequence(token_ids, self.assistant_header_ids)
            if header_start < 0:
                raise ValueError("Assistant header was not found after multimodal tokenization")
            answer_start = header_start + len(self.assistant_header_ids)
            labels[row_index, :answer_start] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def build_training_arguments(cfg: dict[str, Any], output_dir: Path) -> TrainingArguments:
    training = cfg["training"]
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training["epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=int(training["logging_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=int(training["num_workers"]),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=[],
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    set_seed(int(cfg["training"]["seed"]))

    model_dir = resolve_path(cfg["model"]["base_model"])
    output_dir = resolve_path(cfg["training"]["output_dir"])
    benchmark_config_path = resolve_path(cfg["data"]["benchmark_config"])
    annotations_dir = resolve_path(cfg["data"]["annotations_dir"])
    validate_data_manifest(
        annotations_dir,
        benchmark_config_path,
        str(cfg["data"]["schema_version"]),
        str(cfg["data"]["prompt_variant"]),
    )
    train_path = annotations_dir / "train.jsonl"
    valid_path = annotations_dir / "valid.jsonl"
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        min_pixels=int(cfg["data"]["min_pixels"]),
        max_pixels=int(cfg["data"]["max_pixels"]),
    )
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        attn_implementation=str(cfg["model"]["attn_implementation"]),
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_cfg = cfg["model"]["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(LORA_TARGET_MODULES),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=build_training_arguments(cfg, output_dir),
        train_dataset=PointSFTDataset(train_path),
        eval_dataset=PointSFTDataset(valid_path),
        data_collator=PointSFTCollator(processor),
        processing_class=processor,
    )
    resume = str(resolve_path(args.resume_from_checkpoint)) if args.resume_from_checkpoint is not None else None
    trainer.train(resume_from_checkpoint=resume)
    best_adapter_dir = output_dir / "best_adapter"
    trainer.save_model(str(best_adapter_dir))
    processor.save_pretrained(best_adapter_dir)
    trainer.save_state()
    if trainer.is_world_process_zero():
        print(f"Saved the best LoRA adapter to {best_adapter_dir}")


if __name__ == "__main__":
    main()
