# HSP 数据集构建方案

版本：v0.3  
日期：2026-05-30

## 1. 当前主线决策

当前 HSP 主线统一使用去污染后的 NuminaMath-CoT `synthetic_math` split：

```text
SFT:
  ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl
  ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl

RL / GRPO:
  ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
  ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
```

也就是说：

```text
SFT 和 RL 可以来自同一个去污染原始题池，但文件形式不同。
SFT 用 protocol transcript 学协议范式。
RL 用 raw question + gold_answer 学策略。
```

DAPO 不属于当前主线。它只作为 RelayLLM 原论文分布的 baseline / ablation，可选使用，且必须单独报告。

## 2. 为什么不把 DAPO 作为默认 RL 数据

之前曾考虑：

```text
SFT 用 NuminaMath
RL 用 DAPO-17k
```

这个方案有规模优势，但会带来两个问题：

```text
1. 主线数据源不统一，容易把协议学习和策略学习混入不同分布。
2. DAPO 与评测基准的去污染状态不如当前 NuminaMath split 清晰。
```

因此当前阶段选择：

```text
主线：NuminaMath synthetic_math 去污染 split
对照：DAPO / 8B_filtered_data，仅用于 baseline 或 ablation
```

这和 `RL_stage/examples/config_hsp.yaml` 保持一致：

```yaml
data:
  train_files: ${oc.env:INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
  val_files: ${oc.env:INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
  prompt_key: question
  answer_key: gold_answer
```

## 3. 数据目录

默认不要求仓库内存在 `data/`。主线数据直接使用外部目录：

```text
$INFOBUY_GENERATED_DATA
```

也就是：

```text
$INFOBUY_STORE/datasets/infobuy/...
```

`setup/link_data.sh` 只作为旧命令兼容桥接，需要旧 `data/...` 路径时再手动执行。

主线数据目录：

```text
${INFOBUY_GENERATED_DATA}/raw/
${INFOBUY_GENERATED_DATA}/protocol/
${INFOBUY_GENERATED_DATA}/flat/
${INFOBUY_GENERATED_DATA}/replay/
${INFOBUY_GENERATED_DATA}/trust/
${INFOBUY_GENERATED_DATA}/purchase/
${INFOBUY_GENERATED_DATA}/manifests/
${INFOBUY_GENERATED_DATA}/splits/
```

其中：

```text
raw:
  原始题库、去污染 train/validation split。

protocol:
  结构化 segment-level HSP SFT transcript。

flat:
  上传或兼容普通 SFT trainer 的扁平格式。

replay:
  outcome-selected rollout replay SFT。

trust:
  可信协作数据集，训练 ACCEPT / REJECT。

purchase:
  信息购买数据集，训练 NO_PURCHASE / BUY_*。

manifests:
  source、hash、held-out snapshot、去污染报告。

splits:
  数据切分记录。
```

## 4. SFT 数据链路

### 4.1 拉取源题池

```bash
python -m SFT_stage.fetch_hsp_source_dataset \
  --max_records 1000 \
  --request_interval_seconds 2 \
  --output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
  --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_seed_v0_1000.manifest.json
```

源数据：

```text
AI-MO/NuminaMath-CoT
config: default
split: train
source: synthetic_math
```

选择 `synthetic_math` 的原因：

```text
1. 有 problem 和完整 solution，适合构造 SFT transcript。
2. 与 MATH/GSM8K/AIME/Olympiad 等评测源隔离更清晰。
3. Apache-2.0，后续公开数据处理更方便。
```

### 4.2 去污染与切分

```bash
python -m SFT_stage.build_hsp_source_splits \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
  --train_size 800 \
  --validation_size 200 \
  --seed 42 \
  --train_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --validation_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
  --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.json
```

复现已有 held-out snapshot：

```bash
python -m SFT_stage.build_hsp_source_splits \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
  --train_size 800 \
  --validation_size 200 \
  --seed 42 \
  --train_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --validation_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
  --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.json \
  --heldout_snapshot_input ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.heldout_snapshot.json
```

去污染基准：

```text
MATH-500
GSM8K test
AMC23 test
Minerva test
OlympiadBench test
AIME 2024
AIME 2025
```

### 4.3 构造 protocol SFT

训练集：

```bash
python -m SFT_stage.build_hsp_sft \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --variants_per_problem 1 \
  --seed 42
```

验证集：

```bash
python -m SFT_stage.build_hsp_sft \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl \
  --emit_all_types \
  --seed 42
```

六类样本：

```text
normal
ask_help
verify_confirm
verify_accept_correction
verify_reject_bad_feedback
verify_uncertain
```

### 4.4 数据 preflight

```bash
python -m SFT_stage.preflight_hsp \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --require_all_types \
  --require_context_tokens

python -m SFT_stage.preflight_hsp \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl \
  --require_all_types \
  --require_context_tokens
```

## 5. RL 数据链路

RL 当前直接读取 raw split：

```text
${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
```

字段：

```json
{
  "question": "...",
  "gold_answer": "42"
}
```

`RLHFDataset` 使用：

```yaml
prompt_key: question
answer_key: gold_answer
```

奖励函数只需要：

```text
student final answer
ground_truth
hsp_events
teacher token cost
protocol counters
```

因此 RL 不需要 `gold_solution`。

## 6. Outcome Replay 数据

在 SFT cold-start 之后，可采集真实 rollout：

```bash
bash eval/collect_hsp_candidates.bash \
  ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft local_json teacher_name 7778 8 \
  --name ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --output_tag pilot_r1 \
  --max_interactions 3 \
  --ask_budget_tokens 64 \
  --verify_budget_tokens 96
```

构造 replay：

```bash
python -m SFT_stage.build_hsp_outcome_sft \
  --input ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp*.json \
  --output ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
  --teacher_token_budget 192 \
  --teacher_cost_weight 0.15 \
  --min_score 1.0
```

混合：

```bash
python -m SFT_stage.mix_hsp_sft \
  --protocol_data ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --replay_data ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/replay/hsp_sft_mixed_pilot_r1.jsonl \
  --max_replay_fraction 0.50
```

Replay 只用于增强，不替代 protocol seed。

## 7. Trust Dataset

可信数据集放在：

```text
${INFOBUY_GENERATED_DATA}/trust/
```

目标：

```text
teacher feedback 到达后，student 学会 ACCEPT / REJECT / IGNORE。
```

核心字段：

```json
{
  "problem_id": "...",
  "problem": "...",
  "student_state_before_feedback": "...",
  "purchase_action": "BUY_REASONING_VERIFICATION",
  "teacher_observation": "...",
  "teacher_observation_correct": true,
  "student_action_after_feedback": "ACCEPT",
  "student_final_answer": "42",
  "final_correct": true,
  "trust_label": "should_accept"
}
```

用途：

```text
Trust SFT
DPO chosen/rejected
wrong_accept reward analysis
```

## 8. Information Purchase Dataset

信息购买数据集放在：

```text
${INFOBUY_GENERATED_DATA}/purchase/
```

目标：

```text
teacher feedback 到达前，student 学会 NO_PURCHASE / BUY_*。
```

核心字段：

```json
{
  "problem_id": "...",
  "problem": "...",
  "student_state": "...",
  "candidate_purchases": [
    "NO_PURCHASE",
    "BUY_HINT",
    "BUY_PLAN",
    "BUY_FINAL_VERIFICATION",
    "BUY_REASONING_VERIFICATION"
  ],
  "counterfactual_outcomes": {
    "NO_PURCHASE": {"correct": false, "teacher_tokens": 0, "utility": 0.0},
    "BUY_HINT": {"correct": true, "teacher_tokens": 28, "utility": 0.956}
  },
  "best_purchase": "BUY_HINT"
}
```

用途：

```text
Utility SFT
DPO / IPO
GRPO reward analysis
purchase regret evaluation
```

## 9. DAPO 可选对照

DAPO 只作为可选分支：

```text
data source:
  guanning-ai/dapo17k
  ChengsongHuang/8B_filtered_data

storage:
  $INFOBUY_STORE/datasets/hf_downloads/optional_baselines/
```

如果要跑 DAPO 对照，必须显式改配置：

```yaml
data:
  train_files: guanning-ai/dapo17k@train[:80%]
  val_files: guanning-ai/dapo17k@train[80%:]
  prompt_key: problem
  answer_key: answer
```

并在实验报告中单独标注：

```text
This is a DAPO-distribution ablation, not the main decontaminated NuminaMath HSP setting.
```

## 10. 当前文件状态

已存在 pilot 数据：

```text
${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl
${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_seed.jsonl
${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl
${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl
${INFOBUY_GENERATED_DATA}/flat/hsp_sft_train.jsonl
${INFOBUY_GENERATED_DATA}/flat/hsp_sft_validation.jsonl
${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.json
```

正式扩大规模时：

```text
1. 从同一 source=synthetic_math 抽更大的 raw pool。
2. 重新做去污染和 train/val split。
3. 再构造 protocol / replay / trust / purchase 数据。
```
