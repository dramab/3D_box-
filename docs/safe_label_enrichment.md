# 原始标签自然改写

五个 `outputs/enrich_*_full.py` 入口复用 `outputs/enrich_labels_safe_common.py`。当前策略不发送图片，只把 `all_labels.json` 中的原始 `label` 文本提供给大模型，让模型理解原句后生成更自然的一句话指令。

当前大模型服务配置为：

```text
Base URL: https://api.deepseek.com
API: /chat/completions
Model: deepseek-v4-flash
Thinking: disabled
```

API Key 继续通过公共脚本中的 `API_KEY` 常量硬编码，并保留原有的命令行参数、环境变量和 `auth.json` 回退读取逻辑。

## 数据流程

1. 读取 `outputs/auto_labels_*/all_labels.json`，逐条取得原始 `label`；结构化 `spatial_relation` 仅用于本地结果校验，不写入提示词。
2. API 只接收原始标签文本，不接收结构化语义槽、原始RGB、裁剪图、投影图、框线图或placement visualization。
3. 每条标签根据其稳定索引轮换八类表达路线，包括直接指令、礼貌指令、请求问句、目标状态、地点前置、位置选择、结果结构和放置要求。多样性来自句法与信息组织变化，而不只是替换 `Move` 的同义词。
4. 大模型可以省略不影响主要放置意图的source当前位置描述。例如，可将 `Move A located at the right of B to behind C.` 简化为 `Place A behind C.`。
5. 移动物体、target目标关系和target参照物属于核心放置信息，不能删除或改变；不能添加原句没有的物体、属性、关系、距离或场景信息。
6. 本地校验要求输出为一句话，允许以句号或问号结尾，并检查移动物体名称、target参照物名称和target方向分量。关系允许自然同义改写，例如 `behind` 可写为 `at the back of`。
7. 最终标签同时限制为不超过40词且不超过原标签长度加6词。
8. 输出与原句完全相同或校验失败时会重试；达到重试上限后回退到原始标签。最终 `all_labels_enriched.json` 保持原记录结构，仅替换 `label`。

## 并发与断点

- `--workers` 控制并行请求的 batch 数，默认10。
- `--batch-size` 控制每个 API 请求包含的标签数，默认8。
- `.safe_enrichment_state.json` 只用于标签改写断点恢复，不改变最终标签JSON格式，也不参与训练。
- 提示词策略版本发生变化时，旧状态缓存会自动失效并重新生成，避免新旧表达策略混用；也可以显式使用 `--restart`。
- `--restart` 忽略现有状态并按当前策略重新生成。
- `--rgb-image-dir` 和 `--appearance-workers` 为兼容旧命令保留，但当前策略不会读取图片，也不会执行外观请求。

最终结果沿用原先的两级目录存储方式，例如：

```text
outputs/enriched_label_outputs_dopose/auto_labels_dopose/all_labels_enriched.json
```

输入仍来自 `outputs/auto_labels_*/all_labels.json`。可以删除旧的
`outputs/enriched_label_outputs_*` 输出目录，但不能删除输入目录 `outputs/auto_labels_*`。

## 运行示例

先检查前几条标签的实际文本提示词，不调用API：

```bash
python outputs/enrich_dopose_full.py --dry-run --show-first 3
```

小规模API检查时应使用独立输出根目录，避免覆盖全量结果：

```bash
python outputs/enrich_dopose_full.py \
  --limit 20 \
  --output-dir enriched_label_outputs_dopose_pilot \
  --workers 8 \
  --batch-size 4
```

全量运行：

```bash
python outputs/enrich_dopose_full.py \
  --workers 16 \
  --batch-size 8 \
  --restart
```

其他数据集将入口替换为：

```text
outputs/enrich_hope_full.py
outputs/enrich_housecat_full.py
outputs/enrich_omni_full.py
outputs/enrich_ycbv_full.py
```

运行依赖Python 3.8+。API密钥读取顺序为 `--api-key`、代码中的 `API_KEY`、`OPENAI_API_KEY`、`auth.json`。
