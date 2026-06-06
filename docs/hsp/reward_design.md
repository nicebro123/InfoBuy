# HSP 奖励函数设计文档

版本：v0.1
日期：2026-05-29

## 1. 设计理念

HSP 奖励函数的目标是训练小模型在推理过程中做出三类正确的协作决策：

```
能力层面：推不动了 → <ASK>（求助）
认知层面：不确定对不对 → <VERIFY>（审查）
信念层面：看到反馈后 → <ACCEPT> 或 拒绝（信任判断）
```

奖励设计的核心原则：

> 尽量自己解，不会就求助，不确定就审查。错了接受纠正，但不要盲信老师。老师也不确定的时候，靠自己。

## 2. 奖励公式

### 2.1 信任决策矩阵（核心）

HSP 的核心是训练学生的信任判断。VERIFY 之后学生的决策和 Teacher 反馈的真实质量构成一个 2×2 矩阵：

| | Teacher 反馈正确 | Teacher 反馈错误 |
|---|---|---|
| **Student ACCEPT** | useful_accept **(+0.10)** | wrong_accept **(-0.50)** |
| **Student 不 ACCEPT** | **wrong_reject (-0.50)** | resist_bad_review **(+0.10)** |

**wrong_reject 是当前代码缺失的关键惩罚项**：学生原本答错了，Teacher 给了正确纠正，学生却拒绝接受。这与 `wrong_accept`（盲信错误反馈）对称——两者都是信任判断的严重失误，都应重罚。

### 2.2 当前实现（config_hsp_shaped.yaml）

```
R = accuracy                               # 最终答案正确 +1.0
  + useful_accept × 0.10                   # 接受有效纠正
  + resist_bad_review × 0.10               # 拒绝错误反馈且答对
  + independent_correct × 0.05             # 零交互独立答对
  - teacher_cost × 0.15 × cost_ratio       # Teacher token 成本
  - wrong_accept × 0.50                    # 盲信错误反馈
  - wrong_reject × 0.50                    # 拒绝正确反馈 ← 缺失！需新增
  - implicit_adoption × 0.05               # 不用ACCEPT但偷用了Teacher答案
  - wrong_implicit_adoption × 0.50         # 偷用错误答案
  - unsupported_accept × 0.10              # Teacher反馈不明确就ACCEPT
  - invalid_accept × 0.10                  # 协议格式错误
  - invalid_protocol × 0.20                # 协议违规
  - denied_action × 0.05                   # 超出交互次数上限
  - teacher_error × 0.10                   # Teacher调用失败
```

### 2.3 典型场景得分

| 场景 | 计算 | 得分 |
|---|---|---|
| 独立做对，零交互 | 1.0 + 0.05 | **1.05** |
| 求助(ASK 64 tok)后答对 | 1.0 - 0.15×(64/192) | **0.95** |
| 审查(VERIFY) + 拒绝错误反馈 + 答对 | 1.0 + 0.10 - 0.15×(96/192) | **1.025** |
| 审查 + ACCEPT有效纠正 + 答对 | 1.0 + 0.10 - 0.15×(96/192) | **1.025** |
| 审查 + Teacher不确定 + 独立答对 | 1.0 - 0.15×(96/192) | **0.925** |
| 盲信错误反馈（wrong_accept） | 0.0 - 0.50 - cost | **≈ -0.58** |
| **拒绝正确反馈（wrong_reject）** | **0.0 - 0.50 - cost** | **≈ -0.58** |
| 不做交互但答错 | 0.0 | **0.0** |

## 3. 信号维度全景

以下列出所有可观测、可量化的协作行为信号，标注当前状态和潜在用途。

### 3.1 已实现信号

| 信号 | 来源 | 计算方式 | 用途 |
|---|---|---|---|
| **最终正确性** | 答案比对 | mathruler grade_answer | 基础奖励 +1.0 |
| **求助次数** | rollout req 计数器 | ask_count, verify_count | 成本惩罚分母 |
| **Teacher token 消耗** | rollout req 汇总 | teacher_tokens_used | 成本惩罚 |
| **有效 ACCEPT** | event.accepted + teacher feedback 正确 + 学生原答案错 + 最终对 | useful_accepts 计数 | +0.10 |
| **错误 ACCEPT** | event.accepted + teacher feedback 错误 | wrong_accepts 计数 | -0.50 |
| **拒绝错误反馈** | not accepted + teacher feedback 错误 + 最终对 | resisted_bad_reviews 计数 | +0.10 |
| **拒绝正确反馈（缺失）** | not accepted + teacher feedback 正确 + 学生原答案错 | wrong_rejects 计数 | **-0.50 应新增** |
| **隐式采纳（无ACCEPT但用了Teacher答案）** | adopted_teacher_answer_without_accept() | implicit_adoptions 计数 | -0.05 |
| **隐式采纳错误答案** | 同上 + teacher feedback 错误 | wrong_implicit_adoptions 计数 | -0.50 |
| **无依据 ACCEPT** | accepted + teacher feedback 不明确（uncertain/无答案） | unsupported_accepts 计数 | -0.10 |
| **零交互独立正确** | accuracy>0.5 + interaction_count==0 | correct_without_interaction | +0.05 |

### 3.2 已有元数据但未利用

rollout 引擎每条 event 都记录了以下字段，奖励函数可以读取但当前未使用：

```
event = {
    "action":                   "ask" | "verify",
    "requested_budget_tokens":  64,           # 学生申请了多少 token
    "teacher_tokens_used":      42,           # Teacher 实际消耗 token
    "student_before_feedback":  "...",        # 学生求助前的完整文本
    "student_after_feedback":   "...",        # 学生获得反馈后的回复
    "accepted":                 true|false|null,
    "observation_status":       "pending" | "delivered" | "omitted",
    "error":                    null | "...",
}
```

### 3.3 可新增的信号维度

以下信号可以从已有数据中计算得出，无需修改 rollout 引擎。

#### 信号 A（关键缺失）：拒绝正确反馈 — wrong_reject

与 `wrong_accept` 对称，是信任判断 2×2 矩阵中缺失的象限。当前代码有 `useful_accept`、`wrong_accept`、`resist_bad_review`，但没有对"Teacher 给正确反馈但学生拒绝"的惩罚。

```python
wrong_reject = (
    not accepted                          # student 没输出 <ACCEPT>
    and feedback_correctness is True      # Teacher 反馈是正确的
    and tentative_correct is not True     # student 原答案确实是错的
    and accuracy < 0.5                    # 最终也没修正 → 固执导致失败
)
```

与 `implicit_adoption` 的区分：
- `wrong_reject`：student 没 ACCEPT，**也没采纳** Teacher 的正确答案，最终仍然答错 → 纯固执
- `implicit_adoption`：student 没 ACCEPT 但**偷偷用了** Teacher 的正确答案 → 协议违规但至少答案对了

两者不能重叠计数。判断逻辑：

```python
if not accepted and feedback_correctness is True and tentative_correct is not True:
    if adopted_teacher_answer_without_accept(event, response):
        # 隐式采纳（不 ACCEPT 但用了 Teacher 答案） → implicit_adoption
        pass
    elif accuracy < 0.5:
        # 纯拒绝，最终答错 → wrong_reject
        wrong_rejects += 1
```

**建议权重**: `wrong_reject_weight = 0.50`（与 `wrong_accept` 对称，同等重罚）

#### 信号 B：Token 预算效率（Budget Efficiency）

```python
requested = event["requested_budget_tokens"]
actual = event["teacher_tokens_used"]
efficiency = actual / requested

# 理想区间 0.5~0.9
# > 0.95 → Teacher 可能被截断，申请太小
# < 0.3  → 申请太慷慨，浪费额度
budget_waste = max(0, 0.9 - efficiency)  # 只罚浪费，不罚不足
```

**奖励设计**：轻微的预算浪费惩罚，鼓励学生学会精确评估需要多少帮助。
- `budget_waste_weight`: 建议 0.02~0.05

#### 信号 C：求助时机（Interaction Timing）

```python
# 首次 ASK/VERIFY 发生的位置占学生总输出的比例
student_tokens_before_first_interaction = ...
total_student_tokens = ...
timing_ratio = student_tokens_before_first_interaction / max(total_student_tokens, 1)

# 太早 < 0.1 → 没充分尝试就求助（lazy）
# 太晚 > 0.9 → 浪费大量 token 在错误推理上（stubborn）
# 理想区间 0.2~0.7
timing_penalty = 0
if timing_ratio < 0.1:
    timing_penalty = 0.05 * (1 - timing_ratio / 0.1)  # 越早罚越重
elif timing_ratio > 0.9:
    timing_penalty = 0.05 * ((timing_ratio - 0.9) / 0.1)  # 越晚罚越重
```

**奖励设计**：过早或过晚求助的轻微惩罚，鼓励在适当的时候求助。
- 需要 rollout 引擎新增字段 `student_tokens_before_first_interaction`
- `timing_penalty_weight`: 建议 0.03~0.05

#### 信号 D：VERIFY 前置条件 — 是否有答案可审

```python
student_text = event["student_before_feedback"]
has_boxed_before_verify = "\\boxed" in student_text

# VERIFY 语义是 "我有答案了，帮我审查"
# 没有答案就 VERIFY → 协议误用，应该用 ASK
# 有答案才 VERIFY → 协议使用正确

verify_without_answer = (action=="verify" and not has_boxed_before_verify)
```

**奖励设计**：无答案就 VERIFY 的协议误用惩罚。
- `verify_without_answer_weight`: 建议 0.05~0.10

#### 信号 E：交互类型选择准确性（ASK vs VERIFY 不应混淆）

```python
student_text = event["student_before_feedback"]
has_boxed = "\\boxed" in student_text
action = event["action"]

# 有明确答案 → 应该 VERIFY（审查答案），不应 ASK（已经有答案了）
# 没有答案 → 应该 ASK（需要帮助），不应 VERIFY（没东西可审查）

type_confusion = (action=="ask" and has_boxed) or (action=="verify" and not has_boxed)
```

**奖励设计**：交互类型混淆的惩罚，鼓励学生根据自身状态选择正确的协议动作。
- `type_confusion_weight`: 建议 0.05~0.10

#### 信号 F：拒绝反馈后的自主修正

```python
# 当前 resist_bad_review 覆盖：拒绝错误反馈 + 最终答对
# 但可以进一步区分：

# case 1: 拒绝 + 坚持原答案 = 判断力正确但没进步
# case 2: 拒绝 + 修正自己的推理 = 更强的元认知

student_before = event["student_before_feedback"]
student_after = event["student_after_feedback"]
before_boxed = extract_boxed_content(student_before)
after_boxed = extract_boxed_content(student_after)

self_corrected_after_reject = (
    not accepted 
    and feedback_is_correct == False 
    and before_boxed != after_boxed  # 修改了自己的答案
    and accuracy > 0.5               # 最终正确
)
```

**奖励设计**：拒绝错误反馈后自主修正的额外奖励（比单纯拒绝更有价值）。
- `self_correction_weight`: 建议 0.05~0.10

#### 信号 G：Teacher 不确定时独立答对

```python
# verify_uncertain 场景：Teacher 说 "Verdict: uncertain"
# student 独立完成推理并答对 → 有轻微奖励

uncertain_independent_correct = (
    action == "verify"
    and feedback_correctness is None     # Teacher 不确定
    and not accepted                     # 没 ACCEPT
    and accuracy > 0.5                   # 最终答对
)
```

**奖励设计**：面对 Teacher 的不确定性仍能独立判断正确的轻微奖励。
- `uncertain_independent_weight`: 建议 0.03~0.05

#### 信号 H：难度校准（Difficulty Calibration）

```python
# 同一道题的 8 条 rollout (GRPO n=8)，计算组内统计：
# group_pass_rate = 组内答对次数 / 8

# 简单题（group_pass_rate > 0.7）还求助 → 过度依赖
# 难题（group_pass_rate < 0.3）不求助且答错 → 该求助时不求助

over_help_on_easy = (group_pass_rate > 0.7 and interaction_count > 0 and accuracy > 0.5)
under_help_on_hard = (group_pass_rate < 0.3 and interaction_count == 0 and accuracy < 0.5)
```

**奖励设计**：简单题过度求助和难题拒绝求助的惩罚。
- 需要 GRPO 组内统计（batch reward 模式下自然可得）
- `over_help_weight`: 建议 0.05~0.10
- `under_help_weight`: 建议 0.03~0.05

## 4. 扩展后的奖励函数设计

### 4.1 完整公式

```
R = accuracy                              # 基础正确性 +1.0
  # === 正反馈（奖励好行为） ===
  + useful_accept × w₁                    # 接受有效纠正
  + resist_bad_review × w₂               # 拒绝错误反馈
  + self_correction_after_reject × w₃    # 拒绝后自主修正（新）
  + independent_correct × w₄             # 零交互独立答对
  + uncertain_independent_correct × w₅   # Teacher不确定+独立答对（新）
  # === 负反馈 — 信任判断失误（惩罚坏行为） ===
  - wrong_accept × w₆                    # 盲信错误反馈
  - wrong_reject × w₇                    # 拒绝正确反馈 ← 关键缺失！
  # === 负反馈 — 协议/成本 ===
  - teacher_cost × w₈ × cost_ratio       # Teacher token 成本
  - implicit_adoption × w₉               # 不用ACCEPT但偷用Teacher答案
  - wrong_implicit_adoption × w₁₀        # 偷用错误答案
  - unsupported_accept × w₁₁             # Teacher反馈不明确就ACCEPT
  - invalid_accept × w₁₂                 # 协议格式错误
  - invalid_protocol × w₁₃               # 协议违规
  - denied_action × w₁₄                  # 超出交互次数上限
  - teacher_error × w₁₅                  # Teacher调用失败
  # === 新增可选信号 ===
  - budget_waste × w₁₆                   # Token预算浪费
  - timing_penalty × w₁₇                 # 求助时机不当
  - type_confusion × w₁₈                 # 交互类型混淆
  - over_help × w₁₉                      # 简单题过度求助
```

### 4.2 推荐权重

| 权重 | 含义 | 推荐值 | 优先级 |
|---|---|---|---|
| w₁ `useful_accept` | 接受有效纠正 | 0.10 | 核心 |
| w₂ `resist_bad_review` | 拒绝错误反馈 | 0.10 | 核心 |
| w₃ `self_correction_after_reject` | 拒绝后自主修正 | 0.05 | 可选 |
| w₄ `independent_correct` | 零交互独立答对 | 0.05 | 核心 |
| w₅ `uncertain_independent_correct` | Teacher不确定+独立答对 | 0.03 | 可选 |
| w₆ `wrong_accept` | 盲信错误反馈 | 0.50 | 核心 |
| **w₇ `wrong_reject`** | **拒绝正确反馈（缺失）** | **0.50** | **核心** |
| w₈ `teacher_cost` | Token成本 | 0.15 | 核心 |
| w₉ `implicit_adoption` | 隐式采纳 | 0.05 | 核心 |
| w₁₀ `wrong_implicit_adoption` | 隐式采纳错误答案 | 0.50 | 核心 |
| w₁₁ `unsupported_accept` | ACCEPT无依据 | 0.10 | 核心 |
| w₁₂ `invalid_accept` | 协议格式错误 | 0.10 | 辅助 |
| w₁₃ `invalid_protocol` | 协议违规 | 0.20 | 辅助 |
| w₁₄ `denied_action` | 超出交互上限 | 0.05 | 辅助 |
| w₁₅ `teacher_error` | Teacher调用失败 | 0.10 | 辅助 |
| w₁₆ `budget_waste` | Token预算浪费 | 0.03 | 可选 |
| w₁₇ `timing_penalty` | 求助时机不当 | 0.03 | 可选 |
| w₁₈ `type_confusion` | 交互类型混淆 | 0.05 | 可选 |
| w₁₉ `over_help` | 简单题过度求助 | 0.05 | 可选 |

## 5. 实现路线

### 第一阶段（当前可用 + 立即补充）
- [x] 核心正确性 + 成本信号（accuracy, teacher_cost）
- [x] 信任信号（useful_accept, wrong_accept, implicit_adoption）
- [x] 独立解题信号（independent_correct）
- [x] 协议合规信号（invalid_accept, invalid_protocol, denied_action）
- [ ] **wrong_reject（拒绝正确反馈）— 信任 2×2 矩阵缺失象限，优先级最高**

### 第二阶段（轻度扩展，不修改 rollout 引擎）
- [ ] 预算效率信号（budget_waste）— event 中已有 `requested_budget_tokens` 和 `teacher_tokens_used`
- [ ] VERIFY 前置条件（type_confusion）— event 中已有 `student_before_feedback`
- [ ] Teacher 不确定+独立答对（uncertain_independent_correct）— 纯 reward 函数逻辑

### 第三阶段（需要修改 rollout 引擎）
- [ ] 求助时机信号（timing_penalty）— 需要新增 `student_tokens_before_first_interaction`
- [ ] 拒绝后自主修正（self_correction_after_reject）— event 已有 `student_after_feedback`，但需要提取 boxed answer 对比
- [ ] 难度校准信号（over_help）— 需要 GRPO 组内统计

## 6. 与 RelayLLM Difficulty-Aware Reward 的对比

| 维度 | RelayLLM | HSP |
|---|---|---|
| 正确性 | :white_check_mark: | :white_check_mark: |
| 成本控制 | :white_check_mark: | :white_check_mark: |
| 难度感知 | :white_check_mark:（group 内比较） | 可选扩展（over_help） |
| 信任判断 | — | :white_check_mark:（核心创新） |
| 协议适当性 | — | :white_check_mark:（type_confusion） |
| 预算精确性 | — | :white_check_mark:（budget_waste） |
| 求助时机 | — | :white_check_mark:（timing_penalty） |

HSP 奖励函数的核心差异化在于**信任维度和协议适当性维度**——不仅奖励"答对了"，还奖励"信任判断正确"和"协议使用恰当"。这与 HSP 将单一 `<call>` 扩展为 `<ASK>/<VERIFY>/<ACCEPT>` 三 token 协议的设计哲学一致。
