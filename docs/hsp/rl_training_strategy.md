# HSP 强化学习训练策略

版本：v0.1
日期：2026-05-29

## 1. 核心挑战

HSP 的 RL 训练面临三个根本性困难：

| 挑战 | 描述 | 后果 |
|---|---|---|
| **动作空间大** | 学生同时决策：正常推理 / ASK(N) / VERIFY(N) / ACCEPT，N 有 3-4 种选择 | 探索效率低，收敛慢 |
| **奖励稀疏** | 奖励只看最终答案是否正确，中间每一步的 ASK/VERIFY/ACCEPT 没有即时反馈 | 信用分配困难 |
| **多目标拉扯** | 省 token vs 多求助、独立解 vs 求助解、信任 vs 怀疑，目标互相矛盾 | 策略震荡，局部最优 |

## 2. 总体策略：分阶段课程学习（Curriculum + Staged RL）

核心思想：**不要一次性把三个 action 都扔给模型，分阶段逐步放开**。

```
Phase 1: 只学 ASK（求助策略）
  放开: <ASK>N</ASK>
  冻结: <VERIFY>, <ACCEPT>
  目标: 学会什么时候该求助、申请多少预算

Phase 2: 加入 VERIFY（审查策略）
  放开: <ASK>N</ASK>, <VERIFY>N</VERIFY>
  冻结: <ACCEPT>
  目标: 学会区分"该求助"和"该审查"两种场景

Phase 3: 加入 ACCEPT（信任策略）
  放开: 全部三个 action
  目标: 学会选择性信任 Teacher 反馈
```

### 2.1 为什么分阶段有效

- **Phase 1** 只学 ASK → 动作空间缩小为 {正常推理, ASK(32), ASK(64), ASK(96), ASK(128)}，只有 5 个离散选择 + 连续推理
- **Phase 2** 加入 VERIFY → 多了一个 action，但模型已经知道什么时候不该 ASK，只需要学会"不该 ASK 但又不确定时可以 VERIFY"
- **Phase 3** 加入 ACCEPT → 最难的信任决策放最后，此时模型已经稳定掌握了求助和审查的时机

类比：先学走路（ASK），再学看路（VERIFY），最后学判断别人指的路对不对（ACCEPT）。

### 2.2 实现方式

最简单的方式是在 rollout 引擎中按 phase 屏蔽特定 action：

```python
# Phase 1: 只允许 ASK
allowed_actions = {"ask"}  # 学生输出 <VERIFY> 和 <ACCEPT> 被当作普通 token 忽略

# Phase 2: 允许 ASK + VERIFY
allowed_actions = {"ask", "verify"}

# Phase 3: 全部放开
allowed_actions = {"ask", "verify", "accept"}
```

### 2.3 Phase 切换时机

监控以下指标决定何时切换到下一 phase：

| 切换点 | 指标 | 条件 |
|---|---|---|
| Phase 1 → 2 | ASK 准确率（求助后答对率） | > 0.7 且稳定 200 steps |
| Phase 1 → 2 | 独立答对率 | > 0.5（不求助也能解不少题） |
| Phase 2 → 3 | VERIFY 触发率 | 稳定在 15-30%（不过度也不不用） |
| Phase 2 → 3 | ASK vs VERIFY 区分度 | ASK 在有答案时触发率低，VERIFY 在有答案时触发率高 |

也可以简单地按 step 数切换：Phase 1 占 40% steps，Phase 2 占 30%，Phase 3 占 30%。

## 3. 课程学习：问题难度递进

除了分阶段放 action，还需要控制训练数据的难度。

### 3.1 难度分级

主线使用去污染后的 NuminaMath raw train split。利用 Teacher (Qwen3-8B)
的 pass_rate 对这些训练题分级：

```
简单题: Teacher pass_rate > 0.7   → 学生自己大概率能解
中等题: 0.3 < pass_rate ≤ 0.7     → 需要一定努力
困难题: pass_rate ≤ 0.3           → 需要外部帮助
```

### 3.2 难度课程

```
Step 0 ~ N/3:     只用中等题（让模型既需要求助，又不能完全依赖）
Step N/3 ~ 2N/3:  加入简单题（训练独立解题能力）
Step 2N/3 ~ N:    加入困难题（训练极限场景下的求助 + 信任判断）
```

### 3.3 为什么不用全部数据

如果一开始就上全量数据：
- 简单题太多 → 模型学会"什么都不做也能拿分"，不学求助
- 困难题太多 → 怎么求助都错，负反馈占主导，策略崩溃

中等题是最佳起点：模型需要做出决策才能答对，但决策质量能体现在最终结果上。

## 4. KL 散度管理

GRPO 用 KL 惩罚防止策略偏离太远。HSP 场景需要更精细的控制。

### 4.1 KL 系数退火

```
训练早期: kl_coef = 0.05  → 紧贴 SFT 策略，稳定探索
训练中期: kl_coef = 0.01  → 逐步放权给 RL 信号
训练后期: kl_coef = 0.005 → 策略基本稳定，微调
```

当前 config 固定 `kl_coef: 0.01`，建议改为退火调度。

### 4.2 每阶段重置 KL

每个 Phase 开始时，将 reference model 更新为上一 Phase 的最终 checkpoint。这样 KL 惩罚是相对于"上一个阶段已学会的策略"，而不是相对于初始 SFT 模型。允许每个阶段有足够的策略改进空间。

## 5. 中间奖励信号（Reward Shaping）

纯结局奖励（只看最终答案）导致信用分配困难。引入轻量中间信号。

### 5.1 协议正确性奖励（Dense, 小权重）

每一步交互都给一个微小信号：

```python
# 正确使用了 ASK（没有 boxed answer 时触发 ASK）→ 微奖励
protocol_ask_correct = (action=="ask" and not has_boxed_before) * 0.02

# 正确使用了 VERIFY（有 boxed answer 时触发 VERIFY）→ 微奖励
protocol_verify_correct = (action=="verify" and has_boxed_before) * 0.02

# 误用协议 → 微惩罚
protocol_misuse = (action=="ask" and has_boxed_before) * -0.02 \
                + (action=="verify" and not has_boxed_before) * -0.02
```

这些权重很小（0.02），不会主导训练，但提供了梯度方向，告诉模型"ASK 是没答案时用的，VERIFY 是有答案时用的"。

### 5.2 预算合理性奖励

```python
# 申请的预算和实际使用量不要差太多
requested = event["requested_budget_tokens"]
actual = event["teacher_tokens_used"]
if actual > 0:
    budget_ratio = actual / requested
    # 理想区间 [0.4, 0.9]，太浪费或太紧都轻微惩罚
    if budget_ratio < 0.3:    # 申请太多
        budget_signal = -0.01
    elif budget_ratio > 0.95: # 可能不够
        budget_signal = -0.01
    else:
        budget_signal = 0.0
```

### 5.3 中间奖励与最终奖励的关系

中间奖励总权重不超过最终奖励的 10%，只起"指引方向"的作用，不改变优化目标。最终奖励（答案正确性 + 信任判断）始终占主导。

## 6. 迭代式 SFT ↔ RL

单轮 RL 容易遗忘 SFT 学到的协议格式。多轮迭代可以使策略逐步精细化。

```
SFT (Protocol 格式) 
  → RL Phase 1 (ASK) 
    → Outcome Replay SFT (选最优 ASK 轨迹，混合 Protocol 数据)
      → RL Phase 2 (ASK + VERIFY)
        → Outcome Replay SFT (选最优 ASK+VERIFY 轨迹)
          → RL Phase 3 (全部 action)
            → Outcome Replay SFT (最终精调)
```

### 6.1 Outcome Replay 的作用

每次 RL 后收集成功的 rollout 轨迹作为新的 SFT 数据。关键筛选标准：

```
保留: 最终答对 + 协议格式正确 + teacher_tokens 在预算内
丢弃: 最终答错 或 协议违规 或 过度使用 teacher
```

然后与原始 Protocol SFT 数据按比例混合（例如 50:50），重新 SFT 一轮，再进入下一阶段 RL。

### 6.2 好处

- 防止灾难性遗忘（不会忘掉协议格式）
- 将 RL 探索到的好策略"固化"到 SFT 权重中
- 每轮 RL 的起点策略越来越好

## 7. 推荐的完整训练流程

```
═══════════════════════════════════════════════════════════════
Step 1: SFT 冷启动
  数据: NuminaMath Protocol SFT (6 种样本类型)
  目标: 学会三个 action 的格式和基本语义
  产出: SFT checkpoint

Step 2: 数据分级
  对 NuminaMath raw train split 用 Teacher pass_rate 分为简单/中等/困难三级

Step 3: RL Phase 1 — ASK only (40% 总 steps)
  数据: 中等题
  allowed_actions: {ask}
  kl_coef: 0.05 → 0.01
  奖励: accuracy + independent_correct + wrong_accept (仅限于 ASK 场景)
  目标: 学会"推不动就求助"

Step 4: Outcome Replay SFT 1
  收集 Phase 1 最优 rollout，混合 Protocol SFT → 重新 SFT

Step 5: RL Phase 2 — ASK + VERIFY (30% 总 steps)
  数据: 中等题 + 简单题
  allowed_actions: {ask, verify}
  kl_coef: 0.01 → 0.005
  奖励: 加入 VERIFY 相关信号 (type_confusion, budget_waste)
  目标: 学会区分"求助"和"审查"场景

Step 6: Outcome Replay SFT 2
  收集 Phase 2 最优 rollout，混合 → 重新 SFT

Step 7: RL Phase 3 — 全部 action (30% 总 steps)
  数据: 全部题目 (简单 + 中等 + 困难)
  allowed_actions: {ask, verify, accept}
  kl_coef: 0.005
  奖励: 完整 HSP 奖励函数 (含 wrong_reject, wrong_accept 等全部信任信号)
  目标: 学会选择性信任

Step 8: 最终 Outcome Replay SFT
  收集最优轨迹，最终精调

Step 9: 评估
  6 个评测基准，interaction_policy=hsp
═══════════════════════════════════════════════════════════════
```

## 8. 关键超参数建议

| 参数 | Phase 1 | Phase 2 | Phase 3 | 说明 |
|---|---|---|---|---|
| `kl_coef` | 0.05→0.01 | 0.01→0.005 | 0.005 | 退火降低 |
| `max_interactions` | 3 | 3 | 3 | 保持不变 |
| `ask_budget_tokens` | 64 | 64 | 64 | Phase 1 可放宽到 128 |
| `verify_budget_tokens` | — | 96 | 96 | Phase 2 开始生效 |
| `n` (rollouts) | 8 | 8 | 8 | GRPO 组大小 |
| `lr` | 1e-6 | 5e-7 | 5e-7 | Phase 2+ 降低学习率 |
| 训练数据难度 | 中等 | 中等+简单 | 全部 | 逐级加入 |
| `independent_correct_weight` | 0.05 | 0.05 | 0.05 | 保持不变 |
| `useful_accept_weight` | — | — | 0.10 | Phase 3 生效 |
| `wrong_reject_weight` | — | — | 0.50 | Phase 3 生效 |
| `wrong_accept_weight` | — | — | 0.50 | Phase 3 生效 |

## 9. 风险与对策

| 风险 | 表现 | 对策 |
|---|---|---|
| **Phase 1 ASK 过度** | 每道题都 ASK | 提高 `teacher_cost_weight`，降低 `independent_correct_weight` |
| **Phase 1 ASK 不足** | 从不 ASK | 只用困难题训练，迫使模型求助 |
| **Phase 2 ASK/VERIFY 混淆** | 有答案还 ASK，没答案却 VERIFY | 开启 `type_confusion` 惩罚（0.05） |
| **Phase 3 盲信 ACCEPT** | wrong_accept 率高 | 提高 `wrong_accept_weight` 到 0.80 |
| **Phase 3 从不 ACCEPT** | accept_count 为 0 | 提高 `useful_accept_weight` 到 0.20，加入 `wrong_reject` 惩罚 |
| **策略崩溃** | reward 突然归零 | 检查 KL 是否过大，回退到上一个 checkpoint，降低 lr 继续 |
| **灾难性遗忘** | 乱用协议格式 | 立即做一轮 Outcome Replay SFT 再继续 RL |

## 10. 简化版（第一版可执行）

如果分阶段课程太复杂，第一版可以先用简化方案：

```
═══════════════════════════════════════════════════
简化版: 单阶段 GRPO + 难度课程

1. SFT (NuminaMath Protocol)
2. RL single-phase:
   - 全部 action 放开
   - 数据: 中等题 → 全部题 (仅难度课程)
   - kl_coef: 0.01 固定
   - 完整 HSP 奖励函数 (含 wrong_reject)
   - lr: 1e-6
3. 1轮 Outcome Replay SFT (可选)
4. 评估

优点: 实现简单，快速验证
缺点: 收敛可能不如分阶段方案稳定
═══════════════════════════════════════════════════
```

两种方案的选择取决于：如果想快速验证 HSP 范式是否有效，用简化版；如果想得到最优策略，用分阶段版。
