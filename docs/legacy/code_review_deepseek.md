# RelayLLM 代码审查报告

> 审查模型：DeepSeek-V4-Pro  
> 审查日期：2026-05-27  
> 审查范围：全代码库（SFT_stage / RL_stage / eval / utils）

---

## 严重程度：中

### 1. `math_help_group.py` — 全错 group 中仍然奖励 LLM 调用

**文件：** [RL_stage/examples/reward_function/math_help_group.py:104-108](RL_stage/examples/reward_function/math_help_group.py#L104-L108)

```python
else:  # has_no_call_correct=False and has_call_correct=False
    if if_call:
        overall_score = call_ratio  # BUG: 全部错误时还正向奖励调用
    else:
        overall_score = 0.0
```

**问题：** 当 GRPO group 中所有 rollout 都回答错误时（没有任何一条正确），有 `<call>` 的样本仍然得到 `call_ratio` 的正向奖励（例如 call_ratio=0.3 得 0.3 分），而没调用的得 0 分。这会训练模型在"调用了也没用"的场景下仍然倾向于调用 LLM，浪费推理成本。

**建议修复：** 该分支统一返回 0 或负值（如 `-call_ratio`），表示"全错时调用也是代价而非收益"。

---

### 2. `hsp_collator.py` — `_in_loss_span` 边界 token 可能丢失

**文件：** [SFT_stage/hsp_collator.py:71-72](SFT_stage/hsp_collator.py#L71-L72)

```python
@staticmethod
def _in_loss_span(start: int, end: int, loss_spans: list[tuple[int, int]]) -> bool:
    return end > start and any(start >= span_start and end <= span_end
                               for span_start, span_end in loss_spans)
```

**问题：** 该函数要求 token 的字符偏移区间**完全**落在某个 loss span 内（`start >= span_start` 且 `end <= span_end`）。当 token 的字符跨 straddle segment 边界时（例如 token 从 student 段末尾延伸到 teacher 段开头），既不被计入 loss span，也不被计入非 loss span，导致该 token 被错误 mask 为 `IGNORE_INDEX`。

**影响：** 边界附近的少量 token 可能损失训练信号。对于中文字符（每个字符一个 token）影响较小，对于英文/数学符号可能影响个别 token。

**建议修复：** 改为宽松匹配（如 `end > span_start and start < span_end`），或至少对部分重叠的 token 也纳入 loss。

---

### 3. `help_vllm_rollout_spmd.py` — RL rollout event 缺少 `requested_budget_tokens` 字段

**文件：** [RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py:471-482](RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py#L471-L482)

**问题：** 在 RL rollout 代码的 `_update_hsp_request_state` 方法中创建 event 时：

```python
req["events"].append({
    "action": action,
    "executed": True,
    "student_before_feedback": req["student_output_for_grading"],
    "student_before_feedback_scope": "cumulative_student_visible",
    "teacher_text": None,
    "teacher_tokens_used": 0,
    "accepted": False if action == "verify" else None,
    "student_after_feedback": None,
    "observation_status": "pending",
    "error": None,
})
```

缺少 `requested_budget_tokens` 字段。该字段直到后续 `_handle_hsp_large_model_step`（line 600-601）才补充设置。而 `eval/generate_withhelp.py` 中的 `_queue_hsp_action`（line 473）在创建 event 时就完整设置了 `requested_budget_tokens`。

**影响：** 如果下游代码（reward 函数 `math_hsp_group.py`、结果分析脚本 `summarize_hsp_results.py` 等）在 event 上假设 `requested_budget_tokens` 一定存在，RL 训练路径的 event 就会出现 KeyError 或 None 值。当前 reward 函数不依赖此字段，但未来扩展时可能踩坑。

**建议修复：** 在 `_update_hsp_request_state` 创建 event 时统一加上 `requested_budget_tokens` 字段（由 `_handle_hsp_large_model_step` 覆盖为实际 applied 值）。

---

## 严重程度：低

### 4. `build_hsp_sft.py` — `clean_gold_answer` 冗余的双重 `.strip()`

**文件：** [SFT_stage/build_hsp_sft.py:120](SFT_stage/build_hsp_sft.py#L120)

```python
def clean_gold_answer(text: str) -> str:
    return text.strip().rstrip(".,;:").strip()  # 第二个 .strip() 完全冗余
```

`.rstrip(".,;:")` 之后已无尾部空白字符，第二个 `.strip()` 是空操作。

---

### 5. `build_hsp_sft.py` — `incorrect_answer` 对负数行为不符合预期

**文件：** [SFT_stage/build_hsp_sft.py:189-194](SFT_stage/build_hsp_sft.py#L189-L194)

```python
def incorrect_answer(gold_answer: str) -> str:
    compact = gold_answer.replace(",", "").strip()
    if re.fullmatch(r"[-+]?\d+", compact):
        return str(int(compact) + 1)  # "-5" → "-4"
```

**问题：** 当正确答案为负数（如 `-5`）时，生成的"错误答案"为 `-4`。在合成训练数据中（`verify_accept_correction`、`verify_reject_bad_feedback` 等类型），`-4` 和 `-5` 过分相似，不易形成明显的"纠错"信号。

**建议修复：** 考虑对负数取反或乘以倍数（如 `-5` → `5` 或 `-10`），使错误答案更明显。

---

### 6. `add_command_upload.py` — 函数参数 `insertion_point` 完全未被使用

**文件：** [SFT_stage/add_command_upload.py:17-18](SFT_stage/add_command_upload.py#L17-L18)

```python
def insert_special_string_between_tokens(
    tokenizer: AutoTokenizer,
    sentence: str,
    insertion_point: int | None = None  # 此参数在函数体内完全被忽略
) -> str:
```

函数文档也明确写了 "this parameter is ignored inside the function"。调用者在 line 95-99 还计算了 `insertion_point` 并传入，形成死代码。

**建议修复：** 移除该参数，或在函数中实际使用它（当不为 None 时用作固定插入位置）。

---

### 7. `results_recheck.py` — `DATASETS_TO_CHECK` 硬编码遗漏多个数据集

**文件：** [eval/results_recheck.py:15](eval/results_recheck.py#L15)

```python
DATASETS_TO_CHECK = ["math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"]
```

**问题：** [eval/datasets_loader.py](eval/datasets_loader.py) 中额外定义了 `mmlu_pro`、`bbeh`、`super_gpqa`、`gpqa` 等多个数据集处理器，但重新检查脚本的硬编码列表不包含它们。如果用户在这些数据集上运行了评测，`results_recheck.py` 会静默跳过。

**建议修复：** 从 `datasets_loader.py` 的 registry 中动态获取数据集列表，或将遗漏的数据集补入硬编码列表。

---

### 8. `help_vllm_rollout_spmd.py` — RL 代码缺少 `force_verify_after_draft` 模式支持

**文件：** [RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py](RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py)

**问题：** `eval/generate_withhelp.py` 的 `run_hsp_generation` 支持 `collection_mode="force_verify_after_draft"`（line 598-614），让模型先生成草稿再强制请求验证。但 RL rollout 代码中完全没有对应的逻辑分支。目前 RL 训练只使用默认的 `policy` 模式，影响有限，但如果未来有人尝试在 RL 中使用该模式，行为会不符合预期。

---

## 非 Bug 但值得注意的设计问题

### A. RL rollout 与 eval 代码高度重复

[help_vllm_rollout_spmd.py](RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py) 和 [generate_withhelp.py](eval/generate_withhelp.py) 有大量几乎相同的 HSP 交互逻辑：
- `_record_hsp_review_decision` vs `_record_review_decision`
- teacher batch 构建和处理
- 环境通知注入
- token 预算管理

两份代码独立维护，已经出现了上述 Bug #3 和 #8 的不一致。随着协议演进，更多分歧可能出现。建议抽取公共 HSP 交互引擎。

### B. `_append_hsp_context` 每次调用重新 tokenize 整个序列

**文件：** [eval/generate_withhelp.py:303-306](eval/generate_withhelp.py#L303-L306)

```python
def _append_hsp_context(req, text, tokenizer, source):
    start = _token_length(tokenizer, req["current_solution"])  # 每次 O(n) 重新 tokenize
    req["current_solution"] += text
    end = _token_length(tokenizer, req["current_solution"])      # 又一次 O(n)
```

每次追加一个 segment 都要完整重新 tokenize 当前序列，总复杂度 O(n²)。在长序列（如 8192 token）上，累积耗时可能不可忽略。

### C. `normalize_record` 定义在多处被导入

**文件：** [SFT_stage/build_hsp_sft.py](SFT_stage/build_hsp_sft.py) 定义 `normalize_record`，[SFT_stage/build_hsp_source_splits.py:24-26](SFT_stage/build_hsp_source_splits.py#L24-L26) 通过 try/except 导入：

```python
try:
    from SFT_stage.build_hsp_sft import normalize_record
except ImportError:
    from build_hsp_sft import normalize_record
```

如果修改了 `build_hsp_sft.py` 中的 `normalize_record`，`build_hsp_source_splits.py` 的行为也会随之变化且不易察觉。

---

## 总结

| 严重程度 | 数量 | 关键问题 |
|---------|------|---------|
| 中 | 3 | 奖励信号错误、边界 token 丢失、event 结构不一致 |
| 低 | 5 | 冗余代码、死参数、硬编码遗漏、行为预期偏差 |
| 设计建议 | 3 | 代码重复、性能、依赖耦合 |

整体而言，代码库架构清晰、测试覆盖较好（SFT_stage/tests、RL_stage/tests、eval/tests 均有测试），上述问题主要分布在边界条件和模块间一致性上，不影响核心训练流程的正确性。
