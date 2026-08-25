# RoboBrain2.5 point baseline and LoRA SFT

This baseline runs RoboBrain2.5-8B-NV on every instruction in the fixed LC-BGPlaceNet test split. It is a qualitative diagnostic only: it predicts a 2D point from RGB and language, then uses the canonical RGB-D image plus oracle source geometry solely to render an upright source box at that point. It does not evaluate collision, support, yaw feasibility, or task success.

## Environment

The `brain` environment contains the compatible RoboBrain and Qwen-VL dependencies.

```bash
conda activate brain
python -c "from transformers import AutoConfig; print('transformers ready')"
```

## Run

The checkpoint must be stored at `hf_cache/RoboBrain2.5-8B-NV/` in this directory.

```bash
conda activate brain
python baselines/robobrain2_5/run_zero_shot_test.py \
  --model-dir baselines/robobrain2_5/hf_cache/RoboBrain2.5-8B-NV
```

Each move instruction is converted to RoboBrain's official pointing style: `Identify spot within the vacant space that's <destination relation>.` Front/back relations are swapped to match RoboBrain's direction convention: `in front of` and `behind`, `front left` and `back left`, plus `front right` and `back right` are exchanged bidirectionally. Other relations remain unchanged. The official coordinate-output suffix is appended unchanged, and the exact model prompt is stored in every JSONL row. GT placement is overlaid only in the visualization: green point and box are GT, red point-derived box is RoboBrain's prediction, and cyan is the source's observed box. Results are written to `outputs/robobrain2_5_front_behind_swapped/`, preserving the earlier prompt variants. The job writes one durable JSONL row per completed instruction and resumes automatically after interruption. To inspect only a short smoke run, append `--max-samples 1`. To rebuild the summary and gallery without model inference, run:

```bash
conda activate brain
python baselines/robobrain2_5/run_zero_shot_test.py \
  --finalize-only \
  --output-dir outputs/robobrain2_5_front_behind_swapped
```

Open `outputs/robobrain2_5_front_behind_swapped/web_vis/index.html` to browse the rendered test examples.

## LoRA bottom-center `(x, y, d)` supervised fine-tuning

The SFT experiment maps RGB plus the verbatim enriched label to one legal
placement bottom-center represented as `(x, y, d)`. Here `x` and `y` are
normalized image coordinates in `[0, 1000]`, and `d` is absolute camera depth in
centimeters. The label is not parsed, rewritten, or direction-swapped. A short
task-specific output-format instruction is appended. Train, validation, and test
are all bound to `configs/lc_bgplacenet_stage2_enriched.yaml`, whose labels and
fixed split come from `data/splits/active_aligned_enriched/`.

First generate the full train and validation annotations in the `spatial`
environment:

```bash
conda activate spatial
python baselines/robobrain2_5/finetune/prepare_sft_data.py \
  --config configs/lc_bgplacenet_stage2_enriched.yaml
```

Then launch LoRA training on eight 24 GB GPUs in the `brain` environment:

```bash
conda activate brain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  baselines/robobrain2_5/finetune/train_lora.py \
  --config configs/robobrain2_5_sft_point_lora.yaml
```

The configuration uses bf16 DDP, gradient checkpointing, PyTorch SDPA, LoRA on
the language model projection layers, and AdamW with a cosine learning-rate
schedule plus 3% warmup. Its global batch size is 64: eight GPUs, one sample per
GPU, and eight gradient-accumulation steps. Before model loading, the training
entrypoint verifies that the annotation manifest was generated from the same
enriched benchmark config and split. The selected adapter is saved under
`outputs/robobrain2_5_sft_bottom_center_xyd_lora_enriched/checkpoints/best_adapter/`.

Run the complete fixed test split with the selected adapter:

```bash
conda activate brain
CUDA_VISIBLE_DEVICES=0 python baselines/robobrain2_5/run_zero_shot_test.py \
  --config configs/lc_bgplacenet_stage2_enriched.yaml \
  --model-dir baselines/robobrain2_5/hf_cache/RoboBrain2.5-8B-NV \
  --adapter-dir outputs/robobrain2_5_sft_bottom_center_xyd_lora_enriched/checkpoints/best_adapter \
  --prompt-variant enriched_label_bottom_center_xyd \
  --output-dir outputs/robobrain2_5_sft_bottom_center_xyd_lora_enriched/inference_test \
  --max-new-tokens 32
```

The inference job writes one JSONL row after every item and resumes safely from
the same output directory. Do not reuse the zero-shot output directory for the
adapter run.

Finally, evaluate all 7,626 test items with the shared placement benchmark:

```bash
conda activate spatial
python baselines/robobrain2_5/benchmark_zero_shot.py \
  --config configs/lc_bgplacenet_stage2_enriched.yaml \
  --predictions outputs/robobrain2_5_sft_bottom_center_xyd_lora_enriched/inference_test/predictions.jsonl \
  --output-dir outputs/robobrain2_5_sft_bottom_center_xyd_lora_enriched/benchmark_official_top1_oracle_geometry \
  --model-label RoboBrain2.5-8B-NV-LoRA-Bottom-Center-XYD-SFT-Enriched \
  --training-protocol lora_sft \
  --prediction-format xyd
```

This remains a Top-1 point baseline. The predicted `(x, y, d)` is backprojected
without borrowing RGB-D depth. Oracle source dimensions/current yaw are still
used after inference to construct the candidate box, Source IoU is not
applicable, and Placement Success@5 equals Placement Success@1.

## Official quantitative benchmark

The existing 2D points can be evaluated as Top-1 3D candidates with the shared
placement benchmark implementation. This post-processing step uses the `spatial`
environment and does not load RoboBrain again:

```bash
conda activate spatial
python baselines/robobrain2_5/benchmark_zero_shot.py
```

Results are written to
`outputs/robobrain2_5_front_behind_swapped/benchmark_official_top1_oracle_geometry/`.
Predictions without valid local depth emit no candidate and count as failures.
Because RoboBrain does not predict a source box, Source IoU is reported as not
applicable. Candidate dimensions and observed source yaw come from oracle source
geometry after inference, so Placement Size IoU and all resulting placement
metrics must be labeled as oracle-geometry-conditioned. With one candidate per
valid point, Placement Success@5 is identical to Placement Success@1.
