# RoboBrain2.5 zero-shot test visualization

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
