# Help-Seeking Policy for Small-Large Model Collaborative Reasoning

版本：v0.1  
日期：2026-05-26

## 1. 研究动机

本课题关注大小模型协同推理中的一个核心问题：小模型不应该只是被动地被大模型替代，而应该主动学习如何管理外部帮助。

我们希望训练出一个具备 help-seeking policy 的小模型，使其在推理过程中能够判断：

1. 什么时候自己继续推理；
2. 什么时候主动请求大模型帮助；
3. 什么时候请求大模型验证当前答案；
4. 什么时候采纳大模型反馈；
5. 如何在准确率、调用成本、延迟和可靠性之间取得平衡。

核心思想可以概括为：

> 小模型不仅要学会回答问题，还要学会何时求助、何时审查、何时相信外部反馈。

这与传统模型路由不同。传统 router 往往在 query level 决定整道题交给小模型还是大模型；我们的目标是在 reasoning process 中进行动态协作决策。

## 2. 与 RelayLLM 的关系

RelayLLM 的主要思想是让小模型作为 active controller，在生成过程中输出特殊调用命令，例如：

```text
<call>128</call>
```

当小模型生成该命令时，系统会调用大模型续写指定数量的 token，然后小模型继续生成。RelayLLM 的训练流程包括：

1. Supervised warm-up：让小模型先学会调用命令的格式；
2. GRPO：通过奖励函数学习什么时候调用大模型；
3. Difficulty-aware reward：鼓励小模型在简单题上独立完成，在困难题上有限求助。

RelayLLM 的 SFT 数据构造方式比较简单。其公开脚本 `SFT_stage/add_command_upload.py` 的核心逻辑是：

```text
读取 question + answer
将 answer tokenize
随机选择 0 到 3 个位置
插入 <call></call>
得到带调用标记的 SFT answer
```

也就是说，RelayLLM 的 SFT 阶段主要是格式冷启动，并不是精确标注“哪里真的应该求助”。真正的策略学习主要发生在 RL 阶段。

我们的设计借鉴 RelayLLM 的 active controller 思想，但不直接照搬 `<call>`。我们更关注小模型的协作心智状态：求助、验证、采纳。

## 3. 我们的 Token 协议

我们设计三个特殊 token：

```text
<ASK>
<VERIFY>
<ACCEPT>
```

三者的语义不同。

### 3.1 `<ASK>`

`<ASK>` 表示能力不足型求助。

小模型认为当前推理推进困难，或者需要大模型补充关键步骤，于是主动请求大模型帮助。

示例：

```text
Question: ...
Small model:
We need to derive the next equation, but I am unsure how to transform this term.
<ASK>

Large model:
Use substitution here ...

Small model:
Following that hint, ...
```

`<ASK>` 默认表示小模型愿意把大模型回复作为后续推理上下文。最终答案仍由小模型生成。

更准确地说，`<ASK>` 默认接受的是大模型提供的“帮助过程”，而不是无条件接受大模型的最终答案。大模型输出的若干 token 会被拼接回当前上下文，小模型继续基于这段帮助完成后续推理。

因此 `<ASK>` 解决的是能力不足问题：

```text
我当前推不动了，请你帮我补一段路，我接着走。
```

在系统实现上，`<ASK>` 后的 teacher 可以被要求输出一段 bounded assistance，例如最多 64 或 128 tokens。这样求助成本可以被明确计算。

### 3.2 `<VERIFY>`

`<VERIFY>` 表示不确定性验证。

小模型已经形成了当前推理或候选答案，但不确定是否正确，于是请求大模型审查。

示例：

```text
Small model:
I think the answer is 42 because ...
<VERIFY>

Large model:
There is an arithmetic mistake in the second step. The correct value should be 36.
```

`<VERIFY>` 不默认接受大模型反馈。它只是请求大模型进行审查。

`<VERIFY>` 后的大模型不应该自由续写完整答案，而应该扮演 verifier。推荐让 teacher 输出结构化审查意见：

```text
Verdict: correct / incorrect / uncertain
Issue: 如果有错，指出错误发生在哪一步
Correction: 如果能修正，给出修正后的关键步骤或答案
Confidence: 可选，表示 teacher 对判断的置信程度
```

例如：

```text
Verdict: incorrect.
Issue: The multiplication in step 2 is wrong.
Correction: 7 * 5 = 35, not 42.
Suggested answer: \boxed{35}.
```

或者：

```text
Verdict: correct.
Issue: None.
Correction: None.
```

因此 `<VERIFY>` 解决的是不确定性审查问题：

```text
我已经有了一个答案，但我不确定它是否可靠，请你检查。
```

### 3.3 `<ACCEPT>`

`<ACCEPT>` 表示显式采纳大模型反馈。

该 token 只应该出现在大模型反馈之后。它将“请求验证”和“相信反馈”分离开，使我们可以研究小模型是否具有选择性采纳能力。

示例：

```text
Large model:
There is an arithmetic mistake. The correct value should be 36.

Small model:
<ACCEPT>
Therefore, the final answer is 36.
```

如果小模型不输出 `<ACCEPT>`，而是继续自己的推理，且没有采用 review 新提出的可解析答案，则可视为拒绝或独立复核；若最终采用新答案，系统记录为隐式采纳并施加协议惩罚。因此第一版不需要额外设计 `<REJECT>`。

`<ACCEPT>` 的价值在于显式建模“信念更新”。如果没有 `<ACCEPT>`，我们只能从最终答案里间接推测小模型是否采纳了大模型反馈；有了 `<ACCEPT>` 后，采纳动作变成可观察、可训练、可奖励、可惩罚的行为。

例如：

```text
Small model:
I think the answer is \boxed{42}.
<VERIFY>

Large model:
Verdict: incorrect.
Issue: The expression should use 7 * 5, not 7 * 6.
Correction: The correct answer is \boxed{35}.

Small model:
<ACCEPT>
Therefore, the final answer is \boxed{35}.
```

如果 teacher 反馈是错的，小模型理想情况下不应该输出 `<ACCEPT>`：

```text
Small model:
I think the answer is \boxed{42}.
<VERIFY>

Large model:
Verdict: incorrect.
Correction: The answer should be \boxed{40}.

Small model:
After rechecking the calculation, my original derivation is still consistent.
Therefore, the answer is \boxed{42}.
```

这里不输出 `<ACCEPT>` 且最终保留自己的可验证结论，表示小模型没有盲目采纳 teacher 的反馈。

所以 `<ACCEPT>` 解决的是采纳决策问题：

```text
大模型已经给出审查意见，我是否应该更新自己的答案？
```

### 3.4 三个 Token 的层次关系

三个 token 对应三个不同层次的协作行为：

```text
<ASK>     能力层面的求助：我不会继续推了，需要外部帮助。
<VERIFY>  认知层面的审查：我有答案但不确定，需要外部检查。
<ACCEPT>  信念层面的更新：我看到了反馈，并决定采纳它。
```

这套协议的优雅之处在于，它没有把大小模型协同简化成“交不交给大模型”，而是拆成了三个更细的决策：

1. 是否借用大模型的推理能力；
2. 是否借用大模型的审查能力；
3. 是否相信大模型的反馈并更新自己的答案。

换句话说：

```text
<ASK> 让小模型借用大模型的能力；
<VERIFY> 让小模型借用大模型的判断；
<ACCEPT> 让小模型学习是否更新自己的信念。
```

这使得小模型不只是一个被路由的对象，而是一个能够管理外部协作的 active controller。

## 4. 研究问题

我们的 help-seeking policy 可以被定义为：

```text
给定问题 x 和当前推理状态 s_t，
小模型策略 pi_theta 选择生成普通 token 或特殊动作 token：

a_t in {normal token, <ASK>, <VERIFY>, <ACCEPT>}
```

其中：

1. `<ASK>` 触发大模型提供帮助；
2. `<VERIFY>` 触发大模型审查当前推理或答案；
3. `<ACCEPT>` 表示采纳最近一次大模型反馈；
4. 普通 token 表示小模型继续独立推理。

我们最终希望回答三个问题：

1. 小模型什么时候应该求助？
2. 小模型什么时候应该验证？
3. 小模型什么时候应该采纳大模型反馈？

## 5. 数据集总体思路

数据集不需要从零人工标注。主实验已确定使用 `AI-MO/NuminaMath-CoT` 的
`default/train` 分片，并限定为 `source=synthetic_math`。选择它的原因是它同时
提供题目与完整正确推理过程，能够直接监督 `<ASK>`、`<VERIFY>`、`<ACCEPT>`
在 teacher observation 前后的合法用法。

RelayLLM 使用的 DAPO 只保留作可选对照实验：它适合在与原论文一致的题目分布
上比较策略，但不作为本研究主线的必要输入。即使能够获得 RelayLLM 的 gated
SFT 数据，也不直接复用其中的 `<call>` 标记，因为其位置主要来自随机插入，
不代表真实求助时机。

推荐流程是：

```text
NuminaMath-CoT / synthetic_math
        ↓
保留 question + clean gold_solution + gold_answer
        ↓
用我们的协议重新构造样本
        ↓
生成 <ASK>, <VERIFY>, <ACCEPT> SFT 数据
```

关键原则：

> 主线只使用 NuminaMath 训练切分学习 HSP；DAPO 不参与主结果，除非明确声明为 RelayLLM 对照实验。

## 6. SFT 数据设计

SFT 的目标不是让模型学会最优策略，而是让模型学会协议格式和基本语义。

也就是说：

```text
SFT 教模型“怎么协作”
RL / preference optimization 教模型“什么时候协作”
```

### 6.1 样本类型 A：No-Action 样本

目的：防止模型学成题题都求助。

构造方式：

```text
Question -> 正常推理 -> final answer
```

示例：

```text
User:
Question ...

Assistant:
Step-by-step reasoning ...
Therefore, the answer is \boxed{42}.
```

建议占比：50% 到 70%。

### 6.2 样本类型 B：ASK 样本

目的：让模型知道在推理卡住时可以输出 `<ASK>`。

构造方式：

```text
Question
    ↓
Small partial reasoning
    ↓
<ASK>
    ↓
Teacher help
    ↓
Small continuation and final answer
```

示例：

```text
User:
Question ...

Assistant:
We can set up the equation as ..., but I am unsure how to simplify the next step.
<ASK>

Teacher:
Use substitution to eliminate x, then solve for y.

Assistant:
Using substitution, ...
Therefore, the answer is \boxed{...}.
```

注意：如果使用标准 SFTTrainer，需要小心处理 loss mask。Teacher 部分应该作为上下文，不应该训练小模型去生成 teacher feedback。

建议占比：10% 到 20%。

### 6.3 样本类型 C：VERIFY 样本

目的：让模型知道在已有候选答案但不确定时可以请求验证。

构造方式：

```text
Question
    ↓
Small reasoning + tentative answer
    ↓
<VERIFY>
    ↓
Teacher verification
    ↓
Small final answer
```

如果小模型原答案正确：

```text
Teacher:
The reasoning is correct.

Assistant:
Therefore, the answer is \boxed{...}.
```

如果小模型原答案错误：

```text
Teacher:
There is an error in step 2. The correct calculation is ...

Assistant:
<ACCEPT>
Therefore, the answer is \boxed{...}.
```

建议占比：15% 到 25%。

### 6.4 样本类型 D：VERIFY + ACCEPT 样本

目的：显式训练小模型在大模型指出有效错误时使用 `<ACCEPT>`。

示例：

```text
User:
Question ...

Assistant:
I think the answer is \boxed{42}.
<VERIFY>

Teacher:
The final answer is incorrect. The mistake is that 7 * 6 was computed as 42, but the expression required 7 * 5. The correct answer is 35.

Assistant:
<ACCEPT>
I accept the correction. Therefore, the answer is \boxed{35}.
```

这类数据对我们的研究非常关键，因为它训练的不是“问大模型”，而是“是否采纳大模型反馈”。

建议占比：5% 到 15%。

## 7. 推荐的数据格式

为了支持 teacher feedback 不参与 loss，建议数据不要只保存成单一 `text` 字段，而是保存结构化字段：

```json
{
  "id": "gsm8k_000001_verify_accept",
  "question": "...",
  "gold_answer": "35",
  "student_prefix": "I think the answer is \\boxed{42}.\\n<VERIFY>",
  "teacher_feedback": "The final answer is incorrect. The mistake is ... The correct answer is 35.",
  "student_continuation": "<ACCEPT>\\nTherefore, the answer is \\boxed{35}.",
  "sample_type": "verify_accept",
  "loss_on": ["student_prefix", "student_continuation"]
}
```

实现时不能假设标准 `DataCollatorForCompletionOnlyLM` 会正确处理交替出现的 student / teacher 段。RelayLLM 当前的 `SFT_stage/train.py` 只针对单段 assistant answer 计算 loss；若直接把 teacher feedback 填进一段多轮文本，teacher 内容可能被错误地纳入 student 的训练目标。

因此 HSP SFT 必须使用显式 segment-level loss mask。可保存为如下可读形式：

```json
{
  "messages": [
    {"role": "user", "content": "Question ..."},
    {"role": "assistant", "content": "I think the answer is \\boxed{42}.\\n<VERIFY>"},
    {"role": "user", "content": "Teacher feedback: The final answer is incorrect ..."},
    {"role": "assistant", "content": "<ACCEPT>\\nTherefore, the answer is \\boxed{35}."}
  ]
}
```

其中训练构造器需将两个 assistant 段标记为 `loss=true`，将 user 和 teacher feedback 段标记为 `loss=false`。训练脚本需要实现自定义 collator 或预先生成 `labels`，对所有非 student-action / student-continuation token 设置 `-100`。

## 8. SFT 构造方法

第一版可以采用半自动构造。

### 8.1 准备原始数据

每条数据至少需要：

```text
question
gold_answer
gold_solution 或 teacher_solution
```

### 8.2 生成 no-action 样本

直接使用 gold solution 或 teacher solution：

```text
question -> solution with final boxed answer
```

### 8.3 生成 ASK 样本

从正确推理中截取前半部分，然后插入 `<ASK>`。

简单规则：

```text
将 solution 按句子或换行切分
随机选择 30% 到 60% 的位置作为 cut point
prefix + <ASK> + teacher continuation + final answer
```

这类似 RelayLLM 的随机插入，但 `<ASK>` 的位置应尽量落在推理步骤边界，而不是任意 token 中间。

### 8.4 生成 VERIFY 样本

构造一个 tentative answer，然后请求验证。

可选做法：

1. 使用小模型采样得到错误答案；
2. 使用规则扰动 gold answer，制造常见计算错误；
3. 使用 teacher model 生成“看似合理但错误”的解法。

然后让 teacher 生成验证反馈。

### 8.5 生成 ACCEPT 样本

当 teacher feedback 指出错误且给出正确修正时，构造：

```text
<ACCEPT>
revised reasoning
final answer
```

第一版不能只构造正确采纳样本。否则模型几乎只会看到“验证反馈值得相信”的条件分布，`<ACCEPT>` 会退化为 `<VERIFY>` 后的固定动作。

第一版就应加入反事实验证样本：

```text
teacher feedback 正确、student 原答案错误 -> 应输出 <ACCEPT>
teacher feedback 错误、student 原答案正确 -> 不应输出 <ACCEPT>
teacher feedback 错误、student 原答案错误但修正方向仍错误 -> 不应输出 <ACCEPT>
teacher 表示 uncertain -> 不应立即 <ACCEPT>，而应重新检查或保留当前答案
```

这样 `<ACCEPT>` 才真正表示“选择性信任”，而不是礼貌性确认标记。

## 9. 奖励函数设计

RL 阶段的目标是让模型自己学会什么时候用 `<ASK>`、`<VERIFY>` 和 `<ACCEPT>`。

基础奖励：

```text
R = R_correct
    - lambda_ask * N_ask
    - lambda_verify * N_verify
    - lambda_token * teacher_tokens
    + beta_accept * correct_accept
    - gamma_accept * wrong_accept
    - gamma_overhelp * over_help
```

### 9.1 正确性奖励

```text
R_correct = 1.0 if final answer is correct else 0.0
```

数学任务中可以使用 boxed answer extraction 和自动判分。

### 9.2 求助成本

```text
N_ask = number of <ASK>
N_verify = number of <VERIFY>
teacher_tokens = tokens generated by large model
```

建议：

```text
ask 与 verify 都应收费，并分别报告真实 teacher token 成本
```

`<ASK>` 通常提供短帮助，`<VERIFY>` 则需要结构化审查，实际谁更贵由 budget 与真实生成 token 数决定，不应仅凭动作名称预设。

### 9.3 ACCEPT 奖励与惩罚

这是我们与 RelayLLM 区分度最高的部分。

```text
correct_accept:
大模型反馈正确，小模型 <ACCEPT> 后最终答对

wrong_accept:
大模型反馈错误，小模型 <ACCEPT> 后最终答错
```

`wrong_accept` 应该重罚，因为我们不希望小模型盲目信任大模型。

### 9.4 第一版推荐参数

可以从以下简单版本开始：

```text
R = 1.0 * final_correct
    - 0.15 * N_ask
    - 0.05 * N_verify
    - 0.001 * teacher_tokens
    + 0.20 * correct_accept
    - 0.50 * wrong_accept
```

如果发现模型过度求助，提高 `lambda_ask` 和 `lambda_verify`。  
如果发现模型不敢使用 `<ACCEPT>`，提高 `beta_accept`。  
如果发现模型盲目接受大模型反馈，提高 `gamma_accept`。

## 10. Difficulty-Aware Reward

可以借鉴 RelayLLM 的 difficulty-aware 思想。

对同一道题采样多条 rollout，然后判断这一组样本中是否存在：

1. 不求助也答对；
2. 求助后答对；
3. 怎么都答不对。

奖励逻辑：

```text
如果存在 no-help correct:
    最奖励不求助且答对
    求助且答对也给分，但扣成本

如果只有 help correct:
    奖励求助且答对
    惩罚不求助且答错

如果没有任何正确轨迹:
    不鼓励盲目大量求助
```

这比简单地“小模型错、大模型对就标记求助”更优雅，因为策略是从 reward 中长出来的。

## 11. 训练路线

推荐路线：

```text
Stage 1: SFT warm-up
    数据：NuminaMath protocol seed
    学会 <ASK>, <VERIFY>, <ACCEPT> 的格式和基础语义

Stage 2: Rollout collection
    数据：去污染后的 NuminaMath 训练题池
    对每道题采样多条协作轨迹

Stage 3: Reward scoring
    根据最终正确性、调用成本、teacher tokens、accept 正误打分

Stage 4: Policy optimization
    数据：去污染后的 NuminaMath 训练题池与筛选出的 replay
    使用 GRPO / PPO / DPO / SimPO 优化 help-seeking policy
```

DAPO 不属于上述主线阶段。仅当实验目标扩展为“在 RelayLLM 原论文题目分布上
控制变量比较协议设计”时，另建 DAPO 对照分支，并独立报告其数据来源与去污染
结果。

如果希望实现难度更低，第一版可以：

```text
SFT + DPO
```

将 rollout 轨迹构造成偏好对：

```text
答对且少求助 > 答对但多求助
答对且正确 ACCEPT > 答错且错误 ACCEPT
VERIFY 后修正答对 > 不 VERIFY 答错
自己答对不求助 > 自己答对还求助
```

后续再升级到 GRPO。

## 12. 实验设置

### 12.1 模型

小模型：

```text
Qwen3-0.6B
Qwen3-1.7B
Qwen2.5-1.5B
```

大模型：

```text
Qwen3-8B
Qwen2.5-7B
更强 API 模型作为 teacher / verifier
```

### 12.2 数据集

第一版：

```text
GSM8K
MATH-500
```

扩展版：

```text
NuminaMath
OlympiadBench
AIME
ARC-Challenge
```

### 12.3 Baselines

至少比较：

1. Small model only；
2. Large model only；
3. Always ask；
4. Always verify；
5. Confidence-based routing；
6. RelayLLM-style `<call>`；
7. Ours: `<ASK>/<VERIFY>/<ACCEPT>` help-seeking policy。

### 12.4 指标

```text
Accuracy
Ask rate
Verify rate
Accept rate
Wrong accept rate
Teacher token ratio
Cost-adjusted accuracy
Latency
```

其中最关键的是：

```text
Cost-adjusted accuracy = Accuracy - alpha * teacher_token_ratio
```

以及：

```text
Wrong accept rate
```

因为它衡量小模型是否盲目信任大模型。

## 13. 可能的创新点

1. 从 query-level routing 变成 process-level help-seeking；
2. 从单一 `<call>` 变成 `<ASK>/<VERIFY>/<ACCEPT>` 三阶段协作协议；
3. 将“求助”和“采纳”分离，显式研究小模型是否选择性相信大模型；
4. 引入 wrong accept penalty，避免小模型盲目服从大模型；
5. 在成本受限条件下学习小模型的协作推理策略。

## 14. 第一版可执行计划

第一步：做数据构造脚本。

输入：

```text
question
gold_solution
gold_answer
```

输出：

```text
normal samples
ask samples
verify samples
verify_accept samples
```

第二步：训练 SFT warm-up 模型。

目标：

```text
模型能稳定生成 <ASK>, <VERIFY>, <ACCEPT>
不会频繁乱用特殊 token
```

第三步：实现协作 rollout。

规则：

```text
遇到 <ASK>:
    调用 teacher 生成 bounded assistance
    teacher 输出会直接进入上下文
    student 继续推理并生成最终答案

遇到 <VERIFY>:
    调用 teacher 生成 structured verification
    推荐格式：
        Verdict: correct / incorrect / uncertain
        Issue: ...
        Correction: ...
    student 决定是否输出 <ACCEPT>

遇到 <ACCEPT>:
    记录采纳动作
    如果 teacher feedback 正确且最终答对，奖励
    如果 teacher feedback 错误且最终答错，惩罚
```

第四步：实现 reward function。

先用简单版本：

```text
final_correct
ask cost
verify cost
teacher token cost
wrong accept penalty
```

第五步：跑小规模实验。

建议从：

```text
GSM8K 1k training examples
Qwen3-0.6B student
Qwen3-8B teacher
```

开始。

## 15. 当前结论

我们的方案可以被命名为：

```text
HSP: Help-Seeking Policy
```

更完整的论文题目可以是：

```text
Learning When to Ask and When to Trust:
Help-Seeking Policies for Small-Large Model Collaborative Reasoning
```

中文题目可以是：

```text
面向大小模型协同推理的帮助寻求策略学习
```

一句话摘要：

> 本研究训练小模型在推理过程中主动决定何时求助大模型、何时请求验证，以及何时采纳大模型反馈，从而在保证推理准确性的同时降低大模型调用成本并提升协作可靠性。

## 16. 基于当前 RelayLLM 源码的判断

本节基于项目目录中现有代码逐文件阅读后的结论，用于把研究构想落到真实实现上。

### 16.1 当前代码链路

| 阶段 | 当前文件 | 当前行为 | HSP 是否复用 |
| --- | --- | --- | --- |
| SFT 数据构造 | `SFT_stage/add_command_upload.py` | 在普通 answer 中随机插入 0 到 3 个 `<call></call>` | 保留作 RelayLLM baseline，新增 HSP 构造器 |
| SFT 训练 | `SFT_stage/train_hsp.py` / `SFT_stage/hsp_collator.py` | 多段 HSP transcript 训练，teacher observation 只作为上下文，student span 进入 loss | 已新增 HSP trainer/collator，保留原 `train.py` 作为普通 baseline |
| RL 题库准备 | `RL_stage/data_filter.py` | 用 teacher 对数学问题采样，保留 teacher 至少一次答对的问题 | 可复用为在线 RL 题库过滤 |
| RL 数据装载 | `RL_stage/verl/utils/dataset.py` | 读取 `problem` / `answer`，生成 prompt 和 ground truth | 小改，保持 question + gold 的 RL 数据形式 |
| 协作 rollout | `RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py` | 检测 `<call>N</call>`，请求 teacher 续写 | HSP 主改造点 |
| Teacher 服务 | `utils/vllm_service.py` | 给定 prompt 和 token budget 生成文本 | 增加 action mode / structured response 支持 |
| Reward 管道 | `RL_stage/verl/workers/reward/function.py` | 给 reward function 传 response 和 ground truth | 增加事件元数据传递 |
| Reward 函数 | `RL_stage/examples/reward_function/math_help_group.py` | 正则解析 `<call>N</call>`，做 difficulty-aware 奖励 | 新建 HSP reward，按事件和真实 teacher token 计算 |
| 离线评估 | `eval/generate_withhelp.py` | 独立实现 `<call>` 协作生成 | 必须与训练 rollout 使用同一协议 |

### 16.2 当前实现与 HSP 不兼容的关键点

#### 问题 A：SFT 的 `<call>` 语法与 RL 解析语法不一致

SFT 构造脚本插入：

```text
<call></call>
```

而 RL rollout 和评估代码解析：

```text
<call>128</call>
```

即使复现原方法，这里也应先修正。对 HSP 来说，我们不再让 student 请求任意数字预算，而采用配置固定的 `<ASK>` 和 `<VERIFY>` budget，从根本上消除这一不一致。

#### 问题 B：训练 rollout 与评估 rollout 对 teacher 输出的处理不同

训练 rollout 在 `help_vllm_rollout_spmd.py` 中会 rollback 小模型触发调用的整个当前生成片段，然后放入 teacher 输出。

评估脚本 `eval/generate_withhelp.py` 则保留小模型已经生成的片段和 `<call>`，并把 teacher 输出直接追加在后面。

这造成 train / eval interaction semantics 不一致。HSP 必须定义唯一语义：

```text
student 已产生的推理和动作标记保留；
teacher 输出作为 environment observation 追加；
student 在该观察之后继续生成。
```

因此 HSP 不应 rollback `<ASK>` 或 `<VERIFY>` 之前的 student 推理。

#### 问题 C：teacher 注入 token 当前会进入 actor loss

当前 `_pack_final_batch()` 中 teacher contribution 的 mask 代码被注释掉了。这样 teacher 生成的文字会与 student 动作 token 一起进入 `response_mask`，随后被当成 student policy 的输出参与 PPO/GRPO 更新。

这是 HSP 的阻断问题。特别是 `<VERIFY>` 后的 review 是环境反馈，不能作为 student 自己的动作训练。

HSP 必须维护两个不同 mask：

```text
valid_response_mask:
    所有实际上下文 token 均为 1，包括 teacher feedback；
    用于 attention 和完整 transcript 解码。

policy_response_mask:
    只有 student 生成 token 为 1；
    teacher 注入的 help / review token 为 0；
    用于 actor loss、KL 和 advantage broadcasting。
```

在现有框架中，可以继续让 `response_mask` 表示 `policy_response_mask`，新增 `valid_response_mask` 用于注意力和完整 transcript 记录。最终答案评测必须使用去除 teacher spans 后的 student 输出。构造 `attention_mask` 时必须使用 `valid_response_mask`，否则 student 将看不到 teacher feedback。

#### 问题 D：reward 当前根据文本声明的预算计费，而非真实调用计费

现有 reward 从 response 中读取 `<call>N</call>` 并以 `N` 计算 call ratio。但 teacher 可能少生成、调用失败，或者 HSP 使用固定 budget；文本声明不再可靠。

HSP 应由 rollout 记录真实事件：

```text
ask_count
verify_count
accept_count
teacher_tokens_used
teacher_help_tokens
teacher_review_tokens
raw teacher review text
student 是否在该 review 后输出 <ACCEPT>
```

reward 从这些事件读取成本，并在结合 ground truth 后计算 useful accept / wrong accept 等行为结果。

#### 问题 E：修复前 dataset 与 eval 中会影响复现实验的代码问题

本节保留为历史问题清单，用于说明 HSP 落地时优先修复过哪些工程阻塞点；当前代码状态以第 24 节的“当前落地状态”和 `docs/hsp/training_testing.md` 的运行说明为准。

1. `RL_stage/verl/utils/dataset.py` 的 `_build_messages()` 在开头直接返回，后面的 `format_prompt` 与多模态分支不可达。
2. 同文件 `_filter_overlong_prompts()` 对 text-only prompt 使用 `tokenize=False` 后直接计算字符串长度，不是 token 长度。
3. `eval/generate_withhelp.py` 导入 `evaluation.datasets_loader`，但实际目录名是 `eval`；`eval/evaluate_forhelp.bash` 也以 `evaluation` 模块执行。
4. `example.bash` 调用 `RL_stage/script/model_merger.py`，但源码目录为 `RL_stage/scripts/model_merger.py`。
5. 普通 `vLLMRollout.generate_sequences()` 中存在直接 `exit()`；当前 worker 固定构建 `helpvLLMRollout`，所以不影响 collaborative 路径，但会妨碍普通 rollout baseline。

以上问题不全属于方法创新，但必须在正式实验前修复，否则 baseline、验证集或指标可能不可信。

#### 问题 F：teacher review 中的答案可能污染最终正确率

如果 `<TEACHER_REVIEW>` 中包含：

```text
Suggested answer: \boxed{35}
```

而 reward 直接对包含 teacher span 的完整 response 抽取 boxed answer，那么即使 student 没有接受或没有输出最终答案，也可能被误判为答对。

HSP 必须将两个概念分开：

```text
full_transcript:
    包含 student 与 teacher 内容，用于后续上下文和轨迹分析。

student_output_for_grading:
    删除 teacher help/review span 后，仅保留 student 生成文本；
    最终准确率只能从该文本中提取答案。
```

## 17. HSP 的可执行交互协议

### 17.1 动作标记与环境边界

第一版无需真的扩展 tokenizer 词表。可以沿用 RelayLLM 的方式，把动作实现为可解析的文本控制标记：

```text
<ASK>
<VERIFY>
<ACCEPT>
```

其中 `<ASK>` 和 `<VERIFY>` 是触发环境调用的动作；`<ACCEPT>` 是不触发调用的 student 决策动作。

为了让 teacher 内容在上下文中具有明确来源，系统追加只由环境写入的边界标记：

```text
<TEACHER_HELP> ... </TEACHER_HELP>
<TEACHER_REVIEW> ... </TEACHER_REVIEW>
```

这两个边界不是 student 要学习选择的动作，不计为 action space；它们只用于区分“student 的推理”和“teacher 的观察反馈”。

### 17.2 `<ASK>` 路径

```text
Student:
... I cannot determine the next transformation.
<ASK>

Environment / Teacher:
<TEACHER_HELP>
Use substitution at this point; replace y with 2x + 1 and simplify.
</TEACHER_HELP>

Student:
Substituting gives ...
Therefore, the final answer is \boxed{...}.
```

规则：

1. student 输出 `<ASK>` 后立即停止本轮生成；
2. teacher 使用 help prompt，只给 bounded assistance，不主动替 student 提交最终答案；实现中若 help 回复出现 `\boxed{...}`、`final answer` 或 `suggested answer`，该次回复会被记为失败并替换为不含答案的 `<ENVIRONMENT_NOTICE>`；
3. teacher help 被直接追加到可见上下文；
4. student 不需要对 help 再输出 `<ACCEPT>`，因为发出 `<ASK>` 本身已表示“我请求并临时使用这段能力支持”；
5. 如果 help 不可靠，student 仍可重新推理，或后续再输出 `<VERIFY>`。

### 17.3 `<VERIFY>` 与 `<ACCEPT>` 路径

```text
Student:
... I obtain \boxed{42}, but I am not confident in step 2.
<VERIFY>

Environment / Teacher:
<TEACHER_REVIEW>
Verdict: incorrect
Issue: The multiplication in step 2 uses 7 * 6 instead of 7 * 5.
Correction: The corrected final answer is \boxed{35}.
</TEACHER_REVIEW>

Student:
<ACCEPT>
After correcting step 2, the final answer is \boxed{35}.
```

规则：

1. `<VERIFY>` 只表示请求审查，不蕴含采纳；
2. teacher 必须按 reviewer prompt 生成 review，而不是自由续写题解；
3. `<ACCEPT>` 只在存在尚未处理的 `<TEACHER_REVIEW>` 时合法；
4. 不输出 `<ACCEPT>` 而继续推理，只有在最终答案没有采用 review 新提出的可解析答案时才视为不采纳；若最终复制了新的 suggested answer，则记为 `implicit_adoption_without_accept`；
5. `<ASK>` 后不使用 `<ACCEPT>`，保持“能力补充”和“信念更新”两个概念分离。

### 17.4 状态机

```text
STATE_STUDENT_REASONING
    normal text / final answer -> STATE_DONE
    <ASK>                       -> STATE_WAIT_HELP
    <VERIFY>                    -> STATE_WAIT_REVIEW
    <ACCEPT>                    -> INVALID unless pending review exists

STATE_WAIT_HELP
    teacher bounded help        -> STATE_STUDENT_REASONING

STATE_WAIT_REVIEW
    teacher structured review   -> STATE_AFTER_REVIEW

STATE_AFTER_REVIEW
    <ACCEPT> + continuation     -> STATE_STUDENT_REASONING or STATE_DONE
    continuation without accept -> STATE_STUDENT_REASONING or STATE_DONE
    <ASK> / <VERIFY>            -> allowed under max interaction budget

STATE_WAIT_HELP / STATE_WAIT_REVIEW with insufficient remaining context
    no observation can fit      -> STATE_DONE with observation_status=omitted_no_context_budget
```

配置中应限制：

```text
max_interactions_per_sample = 3
ask_budget_tokens = 64
verify_budget_tokens = 96
```

`VERIFY` 可能比 `ASK` 需要更多 token，因为 review 需要 verdict、issue 和 correction；其调用成本仍可通过 prompt 限定与单独 penalty 控制，而不必预设验证一定更便宜。

## 18. HSP SFT 数据构造的落地方案

### 18.1 保留 baseline，新建 HSP 脚本

不要覆盖 RelayLLM 的原始脚本，以便后续公平对比。新增：

```text
SFT_stage/build_hsp_sft.py
SFT_stage/build_hsp_outcome_sft.py
SFT_stage/mix_hsp_sft.py
SFT_stage/train_hsp.py
SFT_stage/hsp_collator.py
SFT_stage/preflight_hsp.py
```

职责：

| 文件 | 职责 |
| --- | --- |
| `build_hsp_sft.py` | 从公开数学数据或清洗后的 RelayLLM 数据产生 synthetic protocol cold-start transcript |
| `build_hsp_outcome_sft.py` | 从多次真实 HSP rollout 中按正确性与调用成本筛选 replay SFT transcript |
| `mix_hsp_sft.py` | 将 synthetic protocol 与 replay 数据按上限比例混合，避免真实回灌数据抹掉动作覆盖 |
| `hsp_collator.py` | tokenization 后按 segment 构建 labels，只训练 student token |
| `train_hsp.py` | 加载结构化数据、强制执行协议校验并使用 HSP collator 完成 warm-up |
| `preflight_hsp.py` | 在训练前验证动作覆盖、segment loss 隔离、student 不得伪造环境 marker、tokenizer 和 GRPO 配置契约 |

### 18.2 SFT 原始输入

每道题需要：

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": "...",
  "gold_solution": "..."
}
```

来源优先级：

1. 可获得的 RelayLLM/teacher-filtered 数学题作为题目池；
2. NuminaMath 等带解题过程的数据用于 SFT；
3. GSM8K / MATH 用于小规模原型和评测。

若拿到 RelayLLM SFT answer，先删除 `<call...></call>`，不要将随机插入位置作为我们的动作监督。

### 18.3 样本分布

第一版推荐分布：

| 样本类型 | 比例 | 目的 |
| --- | ---: | --- |
| `normal` | 45% | 保持独立解题能力，抑制过度求助 |
| `ask_help` | 15% | 学会能力不足时触发 `<ASK>` 并利用帮助 |
| `verify_confirm` | 10% | 正确但不确定，经 review 确认后继续完成 |
| `verify_accept_correction` | 15% | 错误答案遇到正确纠正，学习 `<ACCEPT>` |
| `verify_reject_bad_feedback` | 10% | teacher 给出错误纠正，学习不盲从 |
| `verify_uncertain` | 5% | teacher 不确定时，学习不立刻 accept |

比例不是最终结论，而是 cold-start 的起点。RL 后应观察动作频率再调整。

### 18.4 结构化样本格式

建议 JSONL 记录 segments，而不是把所有文本提前拼成一个 answer：

```json
{
  "id": "math_0001_verify_accept",
  "sample_type": "verify_accept_correction",
  "gold_answer": "35",
  "segments": [
    {"source": "user", "text": "Question: ...", "loss": false},
    {"source": "student", "text": "I obtain \\\\boxed{42}.\\n<VERIFY>", "loss": true},
    {"source": "teacher", "text": "<TEACHER_REVIEW>\\nVerdict: incorrect\\nIssue: ...\\nCorrection: \\\\boxed{35}\\n</TEACHER_REVIEW>", "loss": false},
    {"source": "student", "text": "<ACCEPT>\\nAfter correction, the final answer is \\\\boxed{35}.", "loss": true}
  ]
}
```

`ask_help` 样本也使用同一格式：

```json
{
  "sample_type": "ask_help",
  "segments": [
    {"source": "user", "text": "Question: ...", "loss": false},
    {"source": "student", "text": "I cannot complete the next step.\\n<ASK>", "loss": true},
    {"source": "teacher", "text": "<TEACHER_HELP>\\nUse substitution ...\\n</TEACHER_HELP>", "loss": false},
    {"source": "student", "text": "Using that step, ... \\\\boxed{4}.", "loss": true}
  ]
}
```

### 18.5 如何生成 VERIFY 的正反反馈

`<ACCEPT>` 是否有意义，取决于训练中是否存在值得拒绝的反馈。

构造流程：

1. 用 base student 对题目采样，得到 tentative solution；
2. 自动判分，划分 tentative correct / tentative incorrect；
3. 对 tentative incorrect，请 teacher 给出真实纠正，生成 `verify_accept_correction`；
4. 对 tentative correct，要求 teacher 生成确认反馈，生成 `verify_confirm`；
5. 对一部分 tentative correct 或 tentative incorrect，构造错误 correction，例如替换最终数值或让另一个采样错误轨迹充当 review，生成 `verify_reject_bad_feedback`；
6. 所有 feedback 均通过 gold answer 自动标记 `feedback_is_correct`。

注意：错误 teacher feedback 必须只用于训练采纳策略和鲁棒性测评，不能被混入正常正确解答数据中而未标注。

### 18.6 SFT loss mask

拼接后的 token 序列必须区分来源：

```text
user token:                  labels = -100
student reasoning token:     labels = token_id
<ASK>/<VERIFY>/<ACCEPT>:     labels = token_id
teacher help/review token:   labels = -100
student continuation token:  labels = token_id
padding token:               labels = -100
```

因此不应直接复用当前 `DataCollatorForCompletionOnlyLM`。它不足以可靠屏蔽夹在两个 assistant 段之间的 teacher 环境反馈。

### 18.7 从实测 rollout 回灌策略数据

六类 synthetic 样本只能解决协议冷启动，不能证明某个位置真的应该 `<ASK>` 或 `<VERIFY>`。仅靠自由策略采样也有盲点：如果当前 student 从不调用 teacher，成功的求助轨迹永远进不了 replay 集。因此 student 具备基本动作能力后，应对同一道题采集四类候选轨迹，再按实测 utility 选择 replay 样本：

| Collection mode | 轨迹含义 |
| --- | --- |
| `independent` | 禁止外部动作，测量 student 自力完成的基线 |
| `force_ask_first` | 在解题起点插入一次 trainable `<ASK>`，测试提示型帮助是否带来收益 |
| `force_verify_after_draft` | 先让 student 给出草稿，再插入一次 trainable `<VERIFY>`，测试审查/采纳是否有用 |
| `policy` | 由当前 student 自己选择动作，观察其已学会的行为 |

```text
U(trajectory) =
    final_score
    - teacher_cost_weight * teacher_tokens_used / teacher_token_budget
    - invalid_accept_weight * invalid_accept_count
    - denied_action_weight * denied_action_count
```

选择原则：

1. 同题独立答对与求助答对同时存在时，优先低成本的独立轨迹；
2. 独立轨迹失败、求助或验证轨迹成功时，保留成功协作轨迹；
3. 默认剔除 teacher 服务失败、非法 `<ACCEPT>`、student 伪造环境 wrapper、超出交互预算或 action token 与事件记录不一致的轨迹；
4. forced 轨迹只有在 outcome-cost utility 胜出时才作为 trainable action target 回灌；
5. teacher segments 原样保留为上下文，但继续保持 `loss=false`。

这里的候选轨迹必须来自训练池，并在结果中写入 `data_role=train`。仓库内置的 `math`、`gsm8k`、`minerva`、`aime*` 等 handler 指向评测数据，只能用于最终测量，不能用于构造 replay SFT；否则模型会通过回灌轨迹见到测试题。生成入口和 replay 构建器均对此做默认拦截。

这对应 `SFT_stage/build_hsp_outcome_sft.py`。它不是完整的反事实标注器：如果采样阶段从未探索到 `<ASK>` 或 `<VERIFY>`，它无法凭空知道该动作的价值。因此实际训练应使用 `mix_hsp_sft.py` 保留一部分 synthetic protocol 数据，并用较高采样多样性或后续 GRPO 提供行为探索。对 VERIFY replay，采集端会基于 `student_before_feedback` 中截至 VERIFY 的累计 student 可见草稿，以及实际注入上下文的 `teacher_context_text`，写入 `tentative_answer_correct`、`feedback_answer_correct` 与 `implicit_adoption_without_accept`。构建器默认拒绝接受错误 correction、未验证 correction、tentative 已正确时的冗余 accept、采用新 correction 却省略 `<ACCEPT>` 的轨迹、student 输出 `<TEACHER_*>` / `<ENVIRONMENT_NOTICE>` 的伪造轨迹，以及无法证明采用上述可见范围判定的旧格式 accepted 轨迹。

## 19. 在线 Rollout 改造方案

### 19.1 配置层

修改 `RL_stage/verl/workers/rollout/config.py`，加入可从 YAML/CLI 覆盖的字段：

```python
interaction_policy: str = "relay_call"  # relay_call | hsp
ask_token: str = "<ASK>"
verify_token: str = "<VERIFY>"
accept_token: str = "<ACCEPT>"
ask_budget_tokens: int = 64
verify_budget_tokens: int = 96
max_interactions: int = 3
teacher_help_temperature: float = 0.7
teacher_review_temperature: float = 0.0
```

实现中新增 `RL_stage/examples/config_hsp.yaml` 作为主奖励配置，新增 `config_hsp_shaped.yaml` 作为完整 shaping 消融配置，并保留旧 `config.yaml` 的 `<call>` 模式作为 baseline。`<TEACHER_HELP>` 与 `<TEACHER_REVIEW>` 由 rollout 作为固定 observation wrappers 注入，不属于 policy 配置动作。

### 19.2 Rollout 状态与事件

`help_vllm_rollout_spmd.py` 应保留原 RelayLLM 路径，新增 `interaction_policy == "hsp"` 分支。每个 request 增加：

```python
{
    "pending_action": None,              # None | "ask" | "verify"
    "pending_review": False,
    "interaction_count": 0,
    "events": [],
    "teacher_spans": [],
    "student_spans": [],
}
```

一次事件结构：

```python
{
    "action": "ask" | "verify",
    "student_prefix_text": "...",
    "teacher_text": "...",
    "teacher_tokens_used": 0,
    "accepted": False,
    "feedback_is_correct": None,
}
```

### 19.3 Student 生成逻辑

HSP 模式下，student SamplingParams 使用：

```python
stop=["<ASK>", "<VERIFY>"]
include_stop_str_in_output=True
```

不把 `<ACCEPT>` 设为 stop，因为它后面应紧跟修改后的 reasoning / final answer。

每次 student 输出后：

1. 将 student token 保留在当前 transcript；
2. 检测本段是否包含 `<ACCEPT>`，若前一事件为 review，则记录 `accepted=True`；
3. 若结束于 `<ASK>`，记录 ask event 并转到 teacher help；
4. 若结束于 `<VERIFY>`，记录 verify event 并转到 teacher review，其中 tentative draft 必须保存截至该动作的累计 student 可见输出，而不是仅保存最后一个生成片段；
5. 若自然结束，输出最终回答；
6. 若交互次数超限，禁止再次调用并让 student 完成答案。

### 19.4 Teacher 调用逻辑

teacher 的 prompt 不能再只是“从当前文本继续生成”。必须根据动作指定角色。

Help prompt 的要求：

```text
You are a mathematical reasoning consultant.
The student explicitly requested a short helpful next step.
Provide bounded assistance only. Do not claim to be the student and do not output control markers.
```

Review prompt 的要求：

```text
You are a verifier.
Review the student's reasoning and tentative answer.
Respond only in this format:
Verdict: correct | incorrect | uncertain
Issue: ...
Correction: ...
Suggested answer: \boxed{...} or None
```

teacher 返回后，由 rollout 加上 `<TEACHER_HELP>` 或 `<TEACHER_REVIEW>` 边界，再追加到 student 可见上下文。无论 teacher 是否因 EOS 停止，控制权都应回到 student；teacher 不直接终止整条轨迹。

### 19.5 Mask 与 DataProto 传递

在 `_pack_final_batch()` 中新增：

```text
valid_response_mask     # 所有实际内容，包括 teacher span
response_mask           # 仅 student span，用于 policy optimization
```

构造方式：

```python
valid_response_mask = get_response_mask(response_ids, eos_token_id)
response_mask = valid_response_mask.clone()
for start, end in teacher_spans:
    response_mask[i, start:end] = 0

attention_mask = concat(prompt_attention_mask, valid_response_mask)
```

同时将事件摘要放入 `non_tensor_batch`：

```python
{
    "ask_count": np.array(..., dtype=object),
    "verify_count": np.array(..., dtype=object),
    "accept_count": np.array(..., dtype=object),
    "invalid_accept_count": np.array(..., dtype=object),
    "invalid_protocol_count": np.array(..., dtype=object),
    "teacher_tokens_used": np.array(..., dtype=object),
    "teacher_help_tokens": np.array(..., dtype=object),
    "teacher_review_tokens": np.array(..., dtype=object),
    "full_transcript": np.array(..., dtype=object),
    "student_output_for_grading": np.array(..., dtype=object),
    "hsp_events": np.array(..., dtype=object)
}
```

在线 RL rollout 阶段不知道 ground truth 是否支持 teacher review，因此不应在这里提前计算 `feedback_is_correct`、`useful_accept_count` 或 `wrong_accept_count`。这些派生指标由 reward manager 在拿到 `ground_truth` 后，根据 raw `events` 计算，但只允许读取 student 实际看见的 `teacher_context_text`，不能读取可能因 budget 截断而不可见的完整 `teacher_text`。如果 teacher 回复包含任何 policy action 或环境 wrapper 保留标记，该回复只记录原文和真实调用成本，事件置错并以不含答案的 `<ENVIRONMENT_NOTICE>` 关闭本次动作；ASK 回复若泄露可直接解析的答案候选，也按同样方式失败处理。如果动作已经生成但剩余上下文连环境 observation 都无法放入，则不能再继续 student generation；eval 与 RL 均将事件标记为 `observation_status=omitted_no_context_budget` 并立即终止该轨迹，默认 replay 过滤会因事件错误将其剔除。如果 student 输出环境 wrapper，则计入 `invalid_protocol_count` 供 reward 惩罚。离线候选采集用于 replay SFT 筛选时已经持有 gold answer，因此会基于累计可见草稿与可见 review 上下文附加有效性元数据，并识别未输出 `<ACCEPT>` 却采用新 correction 的隐式采纳轨迹，避免错误、不可见 correction、协议绕过或伪造环境角色对应的轨迹被回灌。

实现中 `student_output_for_grading` 直接由 rollout 拼接各个 student span 保存，而不是在 reward 阶段通过不连续 token mask 临时恢复文本，从而避免移除 teacher span 后的 decode 边界歧义。

### 19.6 Reward manager 的必要修正

修改 `RL_stage/verl/workers/reward/function.py`：

1. 把 HSP 事件字段从 `data.non_tensor_batch` 传入 custom reward；
2. 保留 `full_transcript` 供日志分析，最终答案判分读取 rollout 提供的 `student_output_for_grading`，排除 teacher span，防止 review 中的答案泄漏到 accuracy；
3. reward 标量写入序列最后一个 student action token 位置，不能再简单使用 `sum(response_mask) - 1`，因为 teacher span 位于序列中间时该下标不再等于最后动作位置。

最后一点可以通过：

```python
last_policy_pos = torch.nonzero(data.batch["response_mask"][i], as_tuple=False)[-1].item()
reward_tensor[i, last_policy_pos] = score["overall"]
```

解决。

## 20. HSP Reward 的精确定义

### 20.1 为什么不能直接奖励 `<ACCEPT>`

`<ACCEPT>` 本身不是好行为；只有在 teacher review 正确且采纳后有助于最终正确时，它才是好行为。反之，接受错误 review 是可靠性失败。

因此所有 accept reward 都必须以 review validity 与 final correctness 为条件。

### 20.2 事件判定

对数学题，可以利用 gold answer 自动判断 review validity：

```text
teacher 提出 corrected / suggested answer 且与 gold 一致：
    feedback_is_correct = true

teacher 提出 corrected / suggested answer 且与 gold 不一致：
    feedback_is_correct = false

teacher 判定 original answer correct：
    结合 verify 前 student 的 tentative answer 与 gold 判断 verdict 是否正确
```

定义：

```text
useful_accept:
    student 输出 <ACCEPT>
    且 teacher 给出可解析、正确的 corrected / suggested answer
    且 verify 前 student 尚未给出正确答案
    且 final_answer_correct

wrong_accept:
    student 输出 <ACCEPT>
    且 feedback_is_correct 为 false

unsupported_accept:
    student 输出 <ACCEPT>
    且可见 review 没有提供可解析、正确的 corrected / suggested answer
    且不属于已识别的错误 correction

implicit_adoption_without_accept:
    student 未输出 <ACCEPT>
    且最终答案采用了可见 review 新提出的正确 suggested answer

wrong_implicit_adoption:
    student 未输出 <ACCEPT>
    且最终答案采用了可见 review 新提出的错误 suggested answer

missed_accept:
    feedback_is_correct
    且 verify 前答案错误
    且 student 未 accept
    且最终仍错误
```

### 20.3 基础 outcome-cost-trust reward

设计初期的候选形式为：

```text
R_base =
    1.00 * final_correct
  - 0.12 * ask_count
  - 0.08 * verify_count
  - 0.001 * teacher_tokens_used
  + 0.20 * useful_accept_count
  - 0.60 * wrong_accept_count
  - 0.10 * invalid_accept_count
 - 0.15 * missed_accept_count
```

这里不强行设定 `<VERIFY>` 一定比 `<ASK>` 便宜；总成本还受到 teacher 实际 token 数控制。上述 action penalty 只表达验证和求助均不是免费动作。

当前主实验配置 `config_hsp.yaml` 使用可解释性优先的 outcome-cost-trust 核心奖励：

```text
R_main =
    1.00 * final_correct
  - 0.15 * (teacher_tokens_used / 192)
  - 0.50 * wrong_accept_count
  - 0.05 * implicit_adoption_without_accept_count
  - 0.50 * wrong_implicit_adoption_count
  - 0.10 * unsupported_accept_count
```

协议约束仍以 guardrail 形式施加固定惩罚，不视为研究目标本身：

```text
R_guardrail =
  - 0.10 * invalid_accept_count
  - 0.20 * invalid_protocol_count
  - 0.05 * denied_action_count
```

其中 `192` 是成本归一化常数，不是截顶阈值；连续调用更多 teacher tokens 会持续增加 penalty。`unsupported_accept_count` 对接受 `uncertain`、被截断而不可见或未给出可验证 correction 的 review 施加较小惩罚，避免模型把 `<ACCEPT>` 当成无代价确认；可解析但错误的 correction 仍由更强的 `wrong_accept_count` 惩罚。`implicit_adoption_without_accept_count` 防止模型通过省略 `<ACCEPT>` 绕过选择性信任的可观测动作，错误的隐式采用与错误显式采纳使用同级惩罚。`invalid_protocol_count` 专门压制 student 伪造 `<TEACHER_HELP>`、`<TEACHER_REVIEW>` 或 `<ENVIRONMENT_NOTICE>` 的角色越权输出。

`useful_accept_count`、`resisted_bad_review_count`、`correct_without_interaction` 与 `teacher_error_count` 在主配置中继续统计，但对应 reward weight 设置为 `0`。前三项不加入主奖励，是因为正确性与 teacher 成本已经表达了优先行为，额外 bonus 会偏向 VERIFY/ACCEPT；`teacher_error_count` 不加入主奖励，是因为服务异常不是 student 的决策错误。对于需要验证额外 shaping 是否有益的消融实验，`config_hsp_shaped.yaml` 显式恢复这些权重：

```text
R_shaped = R_main + R_guardrail
  + 0.10 * useful_accept_count
  + 0.10 * resisted_bad_review_count
  + 0.05 * correct_without_interaction
  - 0.10 * teacher_error_count
```

两套配置均使用同一 `math_hsp_group.py`，所有实际生效或仅记录的权重均在 YAML 中显式声明。该 reward 配合 GRPO 运行，由 GRPO 在同题多条轨迹之间完成相对优势归一化。

### 20.4 Group-relative difficulty-aware reward

在同一道题的 `n` 条 rollout 内，按照能否无协作答对进行分层：

```text
Case 1: 存在无交互且正确的轨迹
    最优：无交互正确
    次优：交互后正确，但扣成本
    错误：不奖励

Case 2: 只有交互后正确的轨迹
    奖励成本较低、accept 行为正确的协作轨迹
    惩罚拒绝有效 review 导致错误的轨迹

Case 3: 所有轨迹均错误
    只保留 wrong_accept / 无效调用惩罚
    不奖励盲目调用 teacher
```

建议新建：

```text
RL_stage/examples/reward_function/math_hsp_group.py
```

而不是覆写原 `math_help_group.py`，以确保 RelayLLM baseline 可以复跑。

## 21. 逐文件改造清单

### 21.1 第一优先级：先跑通评估原型

| 文件 | 修改内容 |
| --- | --- |
| `eval/generate_withhelp.py` | 增加 `--interaction_policy hsp`；实现 ASK/VERIFY/ACCEPT 状态机；记录 events、真实成本和可 replay 的 segments；拒绝含保留标记或 ASK 答案泄露的 teacher 注入并记录 student 伪造环境 marker；离线 replay 候选记录可见范围内的 review/tentative 正确性和隐式采纳；禁用交互后出现的 ASK/VERIFY 计为 denied；支持同题多次采样与 collection modes |
| `eval/datasets_loader.py` | 增加 `local_json` 本地训练池入口，供受控轨迹采集读取 JSON/JSONL 的题目与 gold answer |
| `eval/evaluate_forhelp.bash` | 修正 `evaluation` / `eval` 模块路径；支持 HSP policy、每题采样数与 collection mode 参数；等待复核完成并拒绝静默忽略子任务失败；支持显式复核/覆盖开关 |
| `eval/collect_hsp_candidates.bash` | 针对一个训练数据集采集 independent / forced ASK / forced VERIFY / policy 四组同题候选轨迹，并拒绝内置测试任务 |
| `utils/vllm_service.py` | 支持 help/review 不同 temperature、stop 与可选 mode；或由 client 构造两类 prompt |

先在评估脚本上跑通若干手工 prompts，确认转移轨迹、review 格式和指标，再改分布式 RL rollout。这样调试成本最低。

### 21.2 第二优先级：SFT warm-up

| 新文件 / 修改文件 | 修改内容 |
| --- | --- |
| `SFT_stage/build_hsp_sft.py` | 产生六类结构化 transcript，并含反事实 review |
| `SFT_stage/build_hsp_outcome_sft.py` | 从标记为训练来源的真实 rollout 中按 outcome-cost utility 选择低成本成功 replay 数据，默认拒绝评测轨迹、错误/冗余/未验证的 `<ACCEPT>`、未显式 ACCEPT 的隐式采纳及协议非法 segments |
| `SFT_stage/mix_hsp_sft.py` | 控制 replay 占比并保留 cold-start action coverage |
| `SFT_stage/hsp_collator.py` | 对 student spans 训练，对 teacher spans mask loss；若交互轨迹在 observation/continuation 完成前被截断则拒绝训练 |
| `SFT_stage/train_hsp.py` | HSP SFT 训练入口，启动时强制执行数据协议校验并在 checkpoint 写入 `hsp_training_contract.json` 长度契约 |
| `SFT_stage/preflight_hsp.py` | 训练前严格校验 prompt/student/teacher/environment 角色状态机、loss mask、tokenizer/config 和 SFT/RL 最大可见长度契约 |
| `SFT_stage/add_command_upload.py` | 不改，留作 RelayLLM baseline |

### 21.3 第三优先级：RL rollout 与 reward

| 文件 | 修改内容 |
| --- | --- |
| `RL_stage/verl/workers/rollout/config.py` | 新增 HSP protocol、budget 与 interaction limit 配置 |
| `RL_stage/examples/config.yaml` | 新增 HSP 实验参数或另建 `config_hsp.yaml` |
| `RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py` | 增加 HSP 状态机、teacher wrappers、事件记录和双 mask；拒绝含保留标记或 ASK 答案泄露的 teacher 注入并统计 student 协议越权 |
| `RL_stage/verl/workers/reward/function.py` | 传递事件 metadata、使用正确 mask 定位 reward |
| `RL_stage/examples/reward_function/math_hsp_group.py` | HSP outcome-cost-trust reward，含错误/隐式采纳及协议越权惩罚，由 GRPO 完成同题组内相对归一化 |
| `RL_stage/verl/trainer/metrics.py` | 记录 ask/verify/accept/wrong-accept 与 teacher token 比率 |

### 21.4 实验可靠性修复

| 文件 | 修复内容 |
| --- | --- |
| `RL_stage/verl/utils/dataset.py` | 移除提前 return，修复 prompt token-length 过滤 |
| `RL_stage/verl/workers/rollout/vllm_rollout_spmd.py` | 删除调试用 `exit()`，恢复 non-collaborative baseline |
| `example.bash` | 修正 `scripts/model_merger.py` 路径，并添加 HSP 实验入口 |

## 22. 验证与消融实验

### 22.1 单元级验证

在大规模训练之前必须验证：

1. `<ASK>` 触发 help，不触发 review；
2. `<VERIFY>` 触发结构化 review，不自动 accept；
3. `<ACCEPT>` 仅在尚未被 student continuation 消费的最近一次 review 后计为合法；
4. teacher span 在 `attention_mask` 中可见，在 `response_mask` 中为 0；
5. reward 正确写入最后一个 student token，而不是 teacher span 内；
6. 错误 review 被接受时产生 `wrong_accept`，无可验证 correction 的 review 被接受时产生 `unsupported_accept`；
7. train rollout 与 eval rollout 对同一模拟输出产生一致 transcript。

建议新增测试文件：

```text
SFT_stage/tests/test_hsp_collator.py
RL_stage/tests/test_hsp_rollout_state.py
RL_stage/tests/test_hsp_reward.py
```

### 22.2 主要 baselines

```text
SLM only
LLM only
Always ASK
Always VERIFY then ACCEPT
RelayLLM <call>
Uncertainty-triggered relay（来自 docs/legacy/uncertainty_trust_relay_implementation.md，可作为系统触发 baseline）
HSP: <ASK>/<VERIFY>/<ACCEPT>
```

`uncertainty-triggered relay` 与 HSP 的核心区别是：

```text
前者：系统通过 logprob 替 student 做调用决定；
后者：student policy 主动学习何时求助和是否相信。
```

因此前者适合作为 baseline，而不应取代我们的主方法。

### 22.3 必要消融

```text
HSP without <VERIFY>                    # 只有能力求助
HSP without <ACCEPT>                    # VERIFY 后默认吸收 feedback
HSP without counterfactual bad reviews  # 测试是否退化为总是 accept
HSP without wrong_accept penalty         # 测试可靠性奖励作用
HSP with teacher tokens unmasked         # 只作错误实现诊断，不作为正式方法
```

最能证明研究价值的指标不是单纯 Accuracy，而是：

```text
Accuracy at equal teacher-token budget
Useful accept rate
Wrong accept rate
Recovery rate after correct review
Resistance rate against incorrect review
```

## 23. 推荐实施顺序

为了避免一次修改整个 RL 框架后无法定位问题，建议按以下顺序执行：

```text
Milestone 1: 修复现有评估路径和 baseline 可运行性
Milestone 2: 在 eval 中实现 HSP 状态机，用手工 student 输出验证协议
Milestone 3: 构造小规模 HSP SFT 数据并实现 loss mask
Milestone 4: SFT 一个小模型，验证其能稳定生成三种动作
Milestone 5: 将 HSP 状态机迁入 RL rollout，加入事件 metadata 与双 mask
Milestone 6: 实现 math_hsp_group reward，先跑短程 GRPO smoke test
Milestone 7: 正式对比 RelayLLM、uncertainty baseline 与 HSP
```

当前最重要的实现原则是：

> `<ASK>` 接受的是外部推理帮助，`<VERIFY>` 请求的是外部审查意见，`<ACCEPT>` 表达的是 student 对审查意见的选择性信任；teacher 的文本是环境观察，不能被错误地当作 student 的策略动作训练。

## 24. 当前落地状态（2026-05-26）

目前完成的是 Milestone 1、2、3、5 和 6 的代码基础：修复 baseline 的关键阻塞点，把 HSP 协议做成离线评估原型，实现结构化 SFT 数据/掩码/训练入口，并将 HSP 状态机、双 mask 和事件 reward 接入 RL 路径。当前仍缺少真实数据上的 SFT checkpoint 及 GPU GRPO smoke test，因此还不能声称模型已经学会了校准良好的 help-seeking policy。

### 24.1 已实现代码

| 文件 | 已实现内容 |
| --- | --- |
| `eval/generate_withhelp.py` | 保留原 `relay_call` 模式；新增 `--interaction_policy hsp`；实现 `<ASK>`、`<VERIFY>`、`<ACCEPT>` 交互状态机；加入 teacher help/review wrappers、interaction budget、事件记录与 teacher token 实际用量；拒绝注入含保留标记或 ASK 答案泄露的 teacher 回复并记录 student 伪造环境 marker；支持 `--samples_per_question` 与受控 collection modes |
| `eval/generate_withhelp.py` | 分离 `response` 和 `student_response_for_grading`，判分不读取 teacher span，防止 review 中的标准答案造成结果泄漏 |
| `eval/generate_withhelp.py` | teacher/student tokenizer 不同时，按 student context budget 截断注入的 feedback，同时保留完整调用成本；HSP 输出包含与 SFT loss mask 对齐的 `segments`、`data_role` 来源标记及 replay trust 校验元数据；tokenizer 不匹配时直接失败，不再静默替换 |
| `eval/evaluate_forhelp.bash` | 修复模块路径；第五个参数选择 `relay_call` 或 `hsp`，第六个参数控制每题轨迹采样数，第七个参数选择 collection mode；子任务失败时停止复核，复核同步完成后才返回；`OVERWRITE_RECHECK=1` 显式允许替换复核 sidecar |
| `eval/collect_hsp_candidates.bash` | 对单个训练 task 一次采集四类 counterfactual candidate 结果；拒绝用仓库内置 benchmark 构建 replay |
| `eval/results_recheck.py` | 支持读取 policy 与受控 collection 的 HSP 结果文件；HSP 复核使用 `student_response_for_grading`；通过 `OPENAI_API_KEY` 鉴权，失败时中止；保留 raw 结果并输出逐题 `*_rechecked.json` 及独立 `*_rechecked_summary.json`，同名 sidecar 默认拒绝覆盖 |
| `eval/summarize_hsp_results.py` | 汇总 action 出现率、teacher token 成本、交互/非交互得分、collection/data role 分布和 feedback 截断/失败；额外计算各 intervention 相对 independent 的同题配对效用差与隐式采纳数量 |
| `eval/__init__.py` | 将 `eval` 作为可被 `python -m eval...` 执行的包 |
| `utils/vllm_service.py` | teacher 服务返回实际生成 `token_count`，供 HSP 按真实 teacher 代价记录成本；默认不打印完整题目/回复 |
| `RL_stage/verl/utils/dataset.py` | 恢复 `format_prompt` / 多模态分支可达性；文本 prompt 长度以 tokenizer token 数过滤 |
| `RL_stage/verl/workers/rollout/vllm_rollout_spmd.py` | 删除普通 rollout 内残留的调试 `exit()` |
| `RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py` | 将 collated `raw_prompt_ids` 显式转为 Python list，避免等长 prompt 被 NumPy object 行表示后破坏 token 拼接 |
| `example.bash` | 修正模型合并脚本路径与参数 |
| `SFT_stage/build_hsp_sft.py` | 将含 `question` 与 gold solution 的 JSON/JSONL 转成六类结构化 `segments` 轨迹，移除 RelayLLM `<call>` 标记，输出明确标注为 `synthetic_protocol_seed` |
| `SFT_stage/build_hsp_source_splits.py` | 在协议扩展前拉取或复用快照化的保留数学评测题，执行 exact/high-similarity 去污染、源题等价去重与答案规范化，并以确定性哈希生成互斥 train/validation 原始池 |
| `SFT_stage/build_hsp_outcome_sft.py` | 读取 `results_*_hsp.json` 中的结构化轨迹，仅接受 `data_role=train`，按同题 `score - teacher/action cost` 选择 replay 数据，并拒绝错误/冗余/无元数据证明的 accept、隐式采纳、student 伪造环境 marker 与非法角色顺序轨迹 |
| `SFT_stage/mix_hsp_sft.py` | 保留全部 protocol seed，并限制 replay 在混合 SFT 数据中的最高比例 |
| `SFT_stage/hsp_collator.py` | 使用 fast tokenizer 的 offset mapping 生成 student-only labels；teacher/user/padding labels 均为 `-100`；拒绝 `teacher loss=true` 以及截断后不完整的交互输入 |
| `SFT_stage/train_hsp.py` | 使用结构化数据训练 student；训练启动前强制执行协议数据校验；注册 policy/observation tokens；输出 `hsp_training_contract.json` 供 RL 检查训练可见长度 |
| `SFT_stage/preflight_hsp.py` | 在 SFT / GRPO 前严格检查 role-aware segment 状态机、`<ACCEPT>` 时机、teacher loss 隔离、prompt/student/environment 标记越权、checkpoint 特殊 token、预算和 SFT/RL 长度契约 |
| `SFT_stage/tests/` | 校验六类轨迹构造、Relay `<call>` 清洗、action labels、teacher masking 与 preflight 契约 |
| `RL_stage/verl/workers/rollout/config.py` | 新增 `interaction_policy=hsp`、动作 token、interaction budget 与 teacher 温度配置 |
| `RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py` | 新增 HSP 在线状态机；直接保留 `student_output_for_grading` 和 `full_transcript`；输出双 mask；传递 `hsp_events`；拒绝注入含保留标记或 ASK 答案泄露的 teacher 回复并统计 student 伪造环境 marker；预算拒绝后的动作计入 denied penalty |
| `RL_stage/verl/workers/reward/function.py` | reward 接口传递 HSP metadata；按 student 文本判分；将 scalar reward 写入最后一个 student policy token |
| `RL_stage/examples/reward_function/math_hsp_group.py` | 实现 outcome-cost-trust 主 reward 与 shaped 消融共用的指标/打分函数；主 reward 不重复奖励有效采纳或系统失败事件 |
| `RL_stage/verl/trainer/metrics.py` | 分别统计可见 response 长度、policy response 长度和 observation token 长度 |
| `RL_stage/examples/config_hsp.yaml` | HSP GRPO 主配置，使用精简 outcome-cost-trust reward 与协议 guardrail |
| `RL_stage/examples/config_hsp_shaped.yaml` | HSP GRPO 消融配置，恢复有效采纳、拒绝坏审查、独立正确和 teacher error shaping 权重 |
| `RL_stage/examples/qwen3_hsp_grpo.sh` | HSP GRPO 启动入口；启动训练前自动执行 tokenizer 与配置 preflight |
| `RL_stage/examples/qwen3_4b_math_grpo.sh` | RelayLLM baseline 启动入口使用仓库现有 `math_help_group.py` reward，保留可复跑对照组 |
| `RL_stage/tests/test_math_hsp_reward.py` | 校验主奖励、shaped 奖励、错误采纳、显式信任约束、实际 teacher token 成本和失败调用指标 |

HSP 结果文件中与研究分析直接相关的字段为：

```text
response                       # 含 teacher span 的完整交互 transcript
student_response_for_grading   # 仅 student 输出，用于答案正确性判定
segments                       # user/student/teacher 分段轨迹，可回灌 HSP SFT，loss 标记已明确
events                         # ask / verify、teacher 文本、是否 accept、实际 budget
student_before_feedback         # VERIFY 时截止动作位置的累计 student 可见草稿
teacher_context_text            # student 实际收到的 teacher 文本，用于 ACCEPT 正确性判断
collection_mode                # policy / independent / force_ask_first / force_verify_after_draft
collection_error               # 受控收集协议污染时的拒收原因
data_role / dataset_name       # train/validation/test 来源标记与可追踪的数据集名称
ask_count / verify_count / accept_count
invalid_accept_count / denied_action_count
invalid_protocol_count         # student 输出保留环境 marker 的次数
teacher_tokens_used
teacher_context_tokens / feedback_truncated  # teacher 成本与实际注入上下文长度的差异
```

### 24.2 启动评估

先在一张或多张供 teacher 使用的 GPU 上启动现有服务：

```bash
cd /path/to/InfoBuy
python utils/vllm_service.py \
  --model_path ${INFOBUY_TEACHER_MODELS}/qwen3-8b-main \
  --port 7778 \
  --tensor_parallel_size 1 \
  --trust_remote_code
```

单个数据集的 HSP 评估：

```bash
cd /path/to/InfoBuy
python -m eval.generate_withhelp \
  --small_model ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --dataset math \
  --larger_model teacher_name \
  --large_model_url http://127.0.0.1:7778/generate \
  --interaction_policy hsp \
  --max_interactions 3 \
  --ask_budget_tokens 64 \
  --verify_budget_tokens 96 \
  --samples_per_question 8
```

批量运行仓库既有任务集合：

```bash
cd /path/to/InfoBuy
bash eval/evaluate_forhelp.bash ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft teacher_name 7778 "0 1" hsp 8
```

该批量入口默认对规则判错的样本执行外部 judge 复核，因此需设置 `OPENAI_API_KEY`。若当前仅做不含外部复核的 deterministic smoke test，必须显式设置 `SKIP_LLM_RECHECK=1`；重复运行同名生成结果时需显式设置 `OVERWRITE_RESULTS=1`，重复生成同名复核 sidecar 时还需显式设置 `OVERWRITE_RECHECK=1`，更建议设置 `OUTPUT_TAG=round_001` 产生新的结果文件。生成过程会在逐题 `results_*.json` 旁写入对应的 `results_*_summary.json`，避免并行任务共同追加一个全局汇总文件。复核不会覆盖原始逐题结果，而是在同目录输出 `results_*_rechecked.json` 与 `results_*_rechecked_summary.json`；需报告复核后行为分组时，应把逐题 rechecked 文件传给汇总工具。

为了给 replay selection 提供没有被当前策略探索到的动作候选，在一个训练数据集上采集四类同题轨迹：

```bash
cd /path/to/InfoBuy
bash eval/collect_hsp_candidates.bash \
  ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft local_json teacher_name 7778 8 \
  --name ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --output_tag pilot_r1 \
  --max_interactions 3 \
  --ask_budget_tokens 64 \
  --verify_budget_tokens 96
```

HSP 的逐题结果输出为：

```text
$STORAGE_PATH/evaluation/<student_model>_<teacher_name>/results_local_json_hsp_math_train_<hash>_pilot_r1_hsp.json
$STORAGE_PATH/evaluation/<student_model>_<teacher_name>/results_local_json_hsp_math_train_<hash>_pilot_r1_hsp_independent.json
$STORAGE_PATH/evaluation/<student_model>_<teacher_name>/results_local_json_hsp_math_train_<hash>_pilot_r1_hsp_force_ask_first.json
$STORAGE_PATH/evaluation/<student_model>_<teacher_name>/results_local_json_hsp_math_train_<hash>_pilot_r1_hsp_force_verify_after_draft.json
```

汇总一个或多个 HSP 结果文件中的策略行为：

```bash
python -m eval.summarize_hsp_results \
  $STORAGE_PATH/evaluation/<student_model>_<teacher_name>/results_math_hsp_rechecked.json \
  $STORAGE_PATH/evaluation/<student_model>_<teacher_name>/results_gsm8k_hsp_rechecked.json \
  --output $STORAGE_PATH/summaries/hsp_action_summary.json
```

重点观察 `groups.no_interaction.mean_score`、`groups.any_interaction.mean_score`、`actions_per_example` 和 `teacher_tokens_mean`：前两项说明是否只在需要时调用，后两项说明代价是否失控。

在 student 尚未经过 HSP SFT 之前，上述入口主要用于协议调试和人工/受控 action 轨迹测试；普通 student 不一定会主动输出新的 action markers。`--samples_per_question` 产生的同题多轨迹用于 replay 筛选和策略行为统计，不应将同题重复样本当成独立测试题报告显著性。

`eval/collect_hsp_candidates.bash` 专门服务于训练回放，自动设置 `data_role=train` 并拒绝仓库中绑定到评测集的内置 dataset handlers。`local_json` 读取 JSON/JSONL 训练文件，记录需含 `question`/`problem` 与 `gold_answer`/`answer`；如训练池位于 Hugging Face，也可以使用 `mydataset --name <dataset_id>`。结果文件名自动含训练来源标识和短哈希；同名输出存在时默认报错，防止轨迹被静默覆盖。对 `math`、`gsm8k` 等最终 benchmark 的 HSP 行为分析，应使用 `eval.generate_withhelp` 或 `evaluate_forhelp.bash` 的默认 `data_role=test` 输出，且不得送入 replay SFT。

### 24.3 构建并训练 HSP SFT 数据

原仓库 `SFT_stage/train.py` 默认指向的 `HINT-lab/sft_Qwen_Qwen3-0.6B` 数据集当前为 gated 数据集。使用其数据前需要在 Hugging Face 获得访问权限并导出，或者换用自有的 `question` / `gold_solution` 数据。

#### 24.3.1 已确定的 v0 协议冷启动题库

截至 2026-05-26，第一版 `hsp_protocol_seed.jsonl` 的原始题库已固定为公开数据集 [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) 的 `default/train` 分片，仅选取 `source=synthetic_math`。该数据集为 Apache-2.0 许可证，所选记录同时提供 `problem` 与完整 `solution`，可直接用于构造协议监督。

选择理由是隔离优先于规模优先。`NuminaMath-CoT/train` 的来源统计中，`synthetic_math` 有 167,874 条；本轮没有使用标记为 `math`、`gsm8k`、`synthetic_amc`、`amc_aime` 或 `olympiads` 的记录。候选 `open-r1/OpenR1-Math-220k/default` 中 `olympiads` 占 68,089 / 93,733 条，与当前保留的竞赛型评测目标过近，因此不作为 v0 源题池。

本研究主线后续的 rollout collection、outcome-selected replay 与 policy optimization 也继续从该 NuminaMath 训练池派生，不要求额外引入 DAPO。DAPO 仅用于可选的 RelayLLM 对照分支，不应与主线训练结果混合汇报。

当前目录约定如下：

```text
${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl
${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_seed_v0_1000.manifest.json
${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_seed.jsonl
${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.json
${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl
${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl
${INFOBUY_GENERATED_DATA}/replay/                         # 后续真实 student rollout 筛选结果
```

`SFT_stage/fetch_hsp_source_dataset.py` 通过 Hugging Face Dataset Viewer 分页读取源题，保留题目级 provenance，并对 API 限流执行等待重试。实际生成命令为：

```bash
cd /path/to/InfoBuy
python -m SFT_stage.fetch_hsp_source_dataset \
  --max_records 1000 \
  --request_interval_seconds 2 \
  --output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
  --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_seed_v0_1000.manifest.json

python -m SFT_stage.build_hsp_sft \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_seed.jsonl \
  --emit_all_types

python -m SFT_stage.preflight_hsp \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_seed.jsonl \
  --require_all_types
```

本次运行扫描了 5,155 条源记录，得到 1,000 道去重基题；`--emit_all_types` 产生 6,000 条协议轨迹，六类样本各 1,000 条，`<ASK>` / `<VERIFY>` / `<ACCEPT>` 的出现次数分别为 1,000 / 4,000 / 1,000。该集合的用途是让模型学会状态机与动作语义，不是学习何时调用的最终策略。

由于每道基题对应六条动作轨迹，后续 train/validation 切分必须在协议扩展前进行，并先按规范化 question 去除格式等价的源题；仅检查 `source_id` 不足以防止改写或空白差异造成泄漏。禁止在轨迹层面随机切分，否则同一题目的 solution 会同时进入训练集和验证集。

#### 24.3.2 Pilot v1 去污染切分与协议数据

`SFT_stage/build_hsp_source_splits.py` 在任何协议扩展之前工作，避免同一基题的不同 action 版本跨越训练集和验证集。它从仓库评测入口对应的数据源拉取以下保留问题：

| 保留评测源 | 问题数 |
|---|---:|
| MATH-500 | 500 |
| GSM8K test | 1,319 |
| AMC23 test | 40 |
| Minerva test | 272 |
| OlympiadBench test | 675 |
| AIME-2024 | 30 |
| AIME-2025 | 30 |

对于 `pilot_v1`，脚本对 1,000 道 `synthetic_math` 基题执行规范化 exact-match 与字符 5-gram Jaccard `>=0.85` 的近重复筛查，并在切分前去除规范化 question 相同的内部重复。修复候选扫描提前退出后重新运行，held-out exact/near-duplicate、内部重复与无有效答案记录均移除 `0` 条；随后通过 `sha256(seed:source_id)` 确定性切分为 `800` 道训练题与 `200` 道验证题。脚本将 held-out 原文快照及其规范化 SHA256 写在 manifest 邻接文件中，并在原始 split 输出中补齐经过句末标点清理的 `gold_answer`。首次拉取执行命令为：

```bash
python -m SFT_stage.build_hsp_source_splits \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
  --train_size 800 \
  --validation_size 200 \
  --seed 42 \
  --train_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --validation_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
  --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.json

python -m SFT_stage.build_hsp_sft \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --variants_per_problem 1 \
  --seed 42

python -m SFT_stage.build_hsp_sft \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl \
  --emit_all_types \
  --seed 42
```

需要严格复现该次去污染结果时，不再在线读取可能变化的 held-out 数据，而是增加：

```bash
python -m SFT_stage.build_hsp_source_splits \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
  --train_size 800 --validation_size 200 --seed 42 \
  --train_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --validation_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
  --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.json \
  --heldout_snapshot_input ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.heldout_snapshot.json
```

训练集得到 `800` 条加权抽样协议轨迹；验证集得到 `1,200` 条六类全覆盖轨迹。两者均通过 `preflight_hsp --require_all_types`，其 `source_id` 与规范化 question 集合均不交叉。该数据足以完成 SFT 代码链路和小规模范式学习验证，但不代表正式规模训练完成。

扩大到正式训练规模时，先从相同的 `NuminaMath-CoT/synthetic_math` 来源抽取更大的原始池，再重新执行上述去污染切分流程；不得仅扩大协议扩展后的 JSONL，也不得复用未经重新筛查的新题目。正式抽取使用随机页模式并默认创建断点文件，网络中断后可续跑：

```bash
python -m SFT_stage.fetch_hsp_source_dataset \
  --output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_pool_v1.jsonl \
  --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_pool_v1.manifest.json \
  --version v1_formal_pool --max_records 6500 \
  --sampling_mode shuffled_pages --seed 42

# 中断后使用相同参数并增加 --resume。
```

输入至少需包含题目与可作为正确监督的解题过程：

```json
{"id":"m1","question":"Compute ...","gold_solution":"... \\boxed{42}.","gold_answer":"42"}
```

生成结构化 cold-start 数据：

```bash
cd /path/to/InfoBuy
python -m SFT_stage.build_hsp_sft \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --variants_per_problem 1 \
  --seed 42
```

`--variants_per_problem 1` 按文档中的六类比例采样。协议单测或数据检查时可用 `--emit_all_types`，为每题产生全部六类轨迹。

在投入 SFT 之前先检查结构化轨迹。该检查会拒绝 teacher `loss=true`、review 前出现 `<ACCEPT>`、未由动作触发的 teacher/environment 段、observation 后缺少 student continuation、用户或 student 伪造保留标记、请求与 observation 不匹配等协议污染，并报告三类 action 的覆盖数：

```bash
python -m SFT_stage.preflight_hsp \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl
```

对 `--emit_all_types` 生成的小型协议检查集，可增加 `--require_all_types` 强制六类轨迹齐全。

若 Hugging Face 账户已获得 RelayLLM gated 数据权限，可直接从 Hub 读取：

```bash
python -m SFT_stage.build_hsp_sft \
  --dataset_name HINT-lab/sft_Qwen_Qwen3-0.6B \
  --dataset_split train \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_from_relayllm.jsonl \
  --seed 42
```

没有 gated 数据权限时，同一入口可接入公开、带正确 solution 的数学数据集；构造器支持常见的 `question`/`problem` 与 `gold_solution`/`solution`/`answer` 字段。

开始 SFT：

```bash
python -m SFT_stage.train_hsp \
  --model_name Qwen/Qwen3-0.6B \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --output_dir ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --max_seq_length 12288 \
  --bf16
```

这里真正的 policy actions 只有三个：`<ASK>`、`<VERIFY>`、`<ACCEPT>`。`<TEACHER_HELP>` 与 `<TEACHER_REVIEW>` 是环境观察边界；训练入口可把它们注册为 tokenizer 中的单 token 以缩短上下文，但它们永远不进入 loss labels。当前 GRPO 配置允许 `4096` prompt tokens 加 `8192` visible response tokens，因此 SFT 示例使用 `12288`；训练入口将该值保存进 checkpoint，联合 `--model_path` 与 `--rl_config` 的 preflight 会在两者不兼容时直接报错。

SFT 输出 checkpoint 后，可单独确认 tokenizer 契约；GRPO 需要这一步通过，否则动作 marker 仍可能被拆成普通文本 token：

```bash
python -m SFT_stage.preflight_hsp \
  --model_path ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --require_context_tokens
```

协议 cold-start 之后，可运行 `eval/collect_hsp_candidates.bash` 对同一批题收集自由策略和受控干预轨迹，基于真实成功率和 teacher 成本构造 replay SFT 数据：

```bash
python -m SFT_stage.build_hsp_outcome_sft \
  --input \
    ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp.json \
    ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp_independent.json \
    ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp_force_ask_first.json \
    ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp_force_verify_after_draft.json \
  --output ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
  --teacher_token_budget 192 \
  --teacher_cost_weight 0.15 \
  --min_score 1.0

python -m SFT_stage.preflight_hsp \
  --dataset ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl

python -m SFT_stage.mix_hsp_sft \
  --protocol_data ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --replay_data ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/replay/hsp_sft_mixed_pilot_r1.jsonl \
  --max_replay_fraction 0.50
```

Replay 数据只会复现采样阶段实际出现且成功的动作，因此应使用混合后的 `${INFOBUY_GENERATED_DATA}/replay/hsp_sft_mixed_pilot_r1.jsonl` 进行下一轮 SFT，不能直接用稀疏 replay 替换 cold-start 集。`--max_replay_fraction 0.50` 的含义是保留全部 protocol seed，并让 replay 至多占最终混合集的一半。

`build_hsp_outcome_sft.py` 默认只接受结果中 `data_role=train` 的轨迹。对于 `<ACCEPT>` 轨迹，它要求 tentative 判定来自累计 student 可见草稿、correction 判定来自实际可见的 teacher 上下文，并拒绝错误或冗余采纳。若 student 没有输出 `<ACCEPT>` 却最终采用 review 新给出的答案，轨迹会标注为隐式采纳并从 replay 排除，防止 SFT 学会绕过采纳动作。任何 student 伪造环境 wrapper 或 role 状态机非法的轨迹也会被拒绝。修复前采集的 accepted/non-accepted verification replay 需重新采集后再回灌。

### 24.4 运行 HSP GRPO

RL 训练使用独立配置，保留 RelayLLM baseline 配置不变。teacher 服务启动方式与评估一致，然后在 `RL_stage` 目录执行：

```bash
cd /path/to/InfoBuy/RL_stage
MODEL_PATH=${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
SAVE_PATH=${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_main \
bash examples/qwen3_hsp_grpo.sh
```

该命令默认运行主奖励。运行 shaped 消融时显式设置不同配置与输出目录：

```bash
cd /path/to/InfoBuy/RL_stage
HSP_CONFIG=examples/config_hsp_shaped.yaml \
MODEL_PATH=${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
SAVE_PATH=${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_shaped \
bash examples/qwen3_hsp_grpo.sh
```

该启动脚本会先运行 `SFT_stage/preflight_hsp.py`，检查 checkpoint 的 action/context tokens 与所选配置的 GRPO、budget、response 长度和显式 reward 权重契约；检查失败时不会进入训练。`config_hsp.yaml` 与 `config_hsp_shaped.yaml` 均读取上述去污染 NuminaMath raw train/validation split，字段为 `question` 与 `gold_answer`，不再读取旧的远程训练集或 AIME 验证集；在未手工指定 `SAVE_PATH` 时，两套配置也会写入不同 checkpoint 目录。初版配置明确使用 `adv_estimator: grpo`。HSP 的 teacher span 是中途 observation；对 GAE/critic 形式的价值传播还需额外实验验证，不作为当前默认训练路径。

### 24.5 已验证行为

本轮实现已做以下轻量验证：

```text
Python 源文件语法编译检查通过
Bash 入口语法检查通过
模拟轨迹 VERIFY -> TEACHER_REVIEW -> ACCEPT 可运行
模拟轨迹 ASK -> TEACHER_HELP -> student finish 可运行
上述两条轨迹中 teacher 内容均不会进入 student_response_for_grading
模拟 tokenizer 不一致导致 feedback 膨胀时，注入文本被 context budget 截断且成本仍单独保留
SFT 构造器对两道 toy 数学题成功产生六类轨迹，共 12 条
SFT collator 测试确认仅 student spans 与三个动作进入 labels，teacher spans 被屏蔽
SFT collator 对 teacher segment 误设 loss=true 的数据直接报错
HSP preflight 测试确认 review 前 `<ACCEPT>`、teacher loss 污染、prompt/student/environment 伪造 marker、错序 observation、SFT/RL 可见长度不一致和不支持的 RL 配置会被拒绝
HSP 结果汇总测试确认交互分组正确率、动作频率、teacher token 成本、反馈失败/截断以及 intervention 相对 independent 的配对增益与隐式采纳可以稳定统计
HSP 评估输出保留 segment-level replay 轨迹并记录 VERIFY trust 元数据；outcome replay 测试确认同题优先选择低成本成功行为、剔除失败反馈、非法动作、错误/冗余 ACCEPT，以及缺少可见范围证明的旧格式 accepted 轨迹
受控采集测试确认 forced ASK / forced VERIFY 会产生 trainable action segments，且需满足 interaction budget
数据来源保护测试确认内置 benchmark 不可标记为训练轨迹，且非训练来源不会进入 replay SFT
HSP 数据混合测试确认 replay 占比可受控且 protocol seed 不被移除
RL HSP stub 状态测试确认多轮后 VERIFY 记录累计 student 可见草稿，且 teacher observation 仍为非策略 span
HSP reward 单元测试确认主奖励记录但不额外奖励 useful_accept 或独立正确，shaped 消融可显式恢复这些 bonus；wrong_accept/错误隐式采纳、正确但省略 ACCEPT、无证据 correction 与累计 teacher-token 成本惩罚保持有效
HSP reward 单元测试确认主奖励记录但不惩罚 teacher 系统失败，shaped 消融可恢复该惩罚；student 伪造环境 marker 即使最终答对仍会产生协议 guardrail 惩罚
HSP rollout 在 response budget 用尽时不再把环境终止提示加入 student policy 输出
HSP eval 与 rollout 在动作后没有足够 context 容纳 observation 时均显式终止并记录 `omitted_no_context_budget`，不再产生未闭合交互后的 student continuation
HSP 推理显式保留 special action tokens，且 preflight/rollout 拒绝评测 override 将其隐藏，避免动作标记在解码时无法触发状态机
SFT collator 会剔除在 teacher observation 或后续 student continuation 完成前被截断的交互样本，避免只训练发起动作
teacher 回复若包含协议保留标记或 ASK 答案泄露会被记录为失败；原回复不注入上下文，改由不含答案的环境失败通知闭合状态转移；student 输出环境 marker 会被统计、惩罚并从 replay 中排除
外部答案复核以 `*_rechecked.json` 保留逐题审计结果，并用邻接的 `*_rechecked_summary.json` 保存该次汇总；raw 采集轨迹不会被覆盖，复核 sidecar 也需显式覆盖授权，且不再追加共享汇总文件
批量评估入口拒绝参数缺失或空 GPU 队列，并在生成或外部复核失败时返回非零状态，不会挂起或打印伪成功结果
```

### 24.6 尚未完成的实验部分

以下内容仍属于后续 Milestone，不能用当前评估原型替代：

```text
Milestone 4: 在真实数学 solution 数据上构造数据，对 student 做 HSP SFT，并测量三种动作是否稳定出现
Milestone 6 runtime: 在具备 torch/vLLM/GPU 和 teacher 服务的环境中跑短程 GRPO smoke test
Milestone 7: 与 RelayLLM / uncertainty-triggered baseline 的预算对齐实验
```

代码已实现训练侧 student-only `response_mask`，并用 `valid_response_mask` 保证 teacher observation 对后续 student 可见。尚未完成的是在真实分布式运行中检查 batch shape、吞吐与 reward 统计是否符合预期。
