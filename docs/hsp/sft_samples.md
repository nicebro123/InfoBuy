# HSP SFT 数据集 — 6 种样本类型样例

版本：v0.1
日期：2026-05-29

以下是从 `${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl`（800 条）中选取的 6 种类型各一条典型样例，以交互对话形式展示。

---

## 类型 1: `normal` — 正常推理，无交互

**权重**: 45% | **目的**: 防止模型学会"每道题都求助"

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User:
Find a quadratic with rational coefficients and quadratic term $x^2$ that has
$\sqrt{5}-3$ as a root.
Please reason step by step, and put your final answer within \boxed{}.

Assistant:
Given that one of the roots is $\sqrt{5}-3$, the other root must be its
conjugate, which is $-\sqrt{5}-3$ (since the coefficients of the quadratic
must be rational).

**Step 1: Calculate the sum of the roots.**
\[
(\sqrt{5}-3) + (-\sqrt{5}-3) = -6
\]

**Step 2: Calculate the product of the roots.**
\[
(\sqrt{5}-3)(-\sqrt{5}-3) = (\sqrt{5})^2 - 3^2 = 5 - 9 = -4
\]

**Step 3: Use Vieta's formulas.**
The quadratic equation can be formed using the sum and product of the roots:
\[
x^2 - (\text{sum of roots}) \cdot x + (\text{product of roots}) = x^2 + 6x - 4
\]

**Conclusion:**
The quadratic equation with rational coefficients that has $\sqrt{5}-3$ as a
root is $\boxed{x^2 + 6x - 4}$.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**SFT 可训练内容**：全部 Assistant 文本（loss=true）
`gold_answer`: `"x^2 + 6x - 4"`

---

## 类型 2: `ask_help` — 能力不足，请求帮助

**权重**: 15% | **目的**: 训练模型在推理卡住时使用 `<ASK>N</ASK>`

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User:
How many positive integers less than 201 are multiples of either 4 or 9,
but not both at once?
Please reason step by step, and put your final answer within \boxed{}.

Assistant:
I need a useful next reasoning step before completing this solution.
<ASK>96</ASK>

Teacher:
<TEACHER_HELP>
A useful next step is: 1. Calculate the number of multiples of 4 less than 201:
⌊200/4⌋ = 50.
</TEACHER_HELP>

Assistant:
1. Calculate the number of multiples of 4 less than 201: ⌊200/4⌋ = 50.
2. Calculate the number of multiples of 9 less than 201: ⌊200/9⌋ = 22.
3. Find their least common multiple (LCM) of 4 and 9: 4 × 9 = 36.
4. Calculate the number of multiples of 36 less than 201: ⌊200/36⌋ = 5.
5. Determine the numbers that are multiples of 4 or 9, but not both:
   (50 - 5) + (22 - 5) = 45 + 17 = 62.

The total number of positive integers less than 201 that are multiples of 4
or 9, but not both, is \boxed{62}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**SFT 可训练内容**：`<ASK>96</ASK>` + 最终 Assistant 推理（loss=true）
**不可训练**：Teacher 的 `<TEACHER_HELP>...</TEACHER_HELP>`（loss=false）
`gold_answer`: `"62"` | `feedback_is_correct`: `null`

---

## 类型 3: `verify_confirm` — 答案正确，验证确认

**权重**: 10% | **目的**: 训练模型在已有答案但不确定时请求审查

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User:
Find the positive real number x such that (x-6)/11 = 6/(x-11).
Please reason step by step, and put your final answer within \boxed{}.

Assistant:
Start by cross-multiplying to eliminate the fractions:
\[(x-6)(x-11) = 6 \cdot 11\]

Expanding and simplifying the left-hand side:
\[x^2 - 17x + 66 = 66\]

Subtract 66 from both sides:
\[x^2 - 17x = 0\]

Factoring out the common term, x:
\[x(x-17) = 0\]

Setting each factor equal to zero gives the possible solutions:
\[x = 0 \quad \text{or} \quad x = 17\]

Since the problem asks for the positive real number, we have x = \boxed{17}.
<VERIFY>128</VERIFY>

Teacher:
<TEACHER_REVIEW>
Verdict: correct
Issue: None
Correction: None
Suggested answer: \boxed{17}
</TEACHER_REVIEW>

Assistant:
The checked final answer remains \boxed{17}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**SFT 可训练内容**：推理 + `<VERIFY>128</VERIFY>` + 最终确认文本（loss=true）
**不可训练**：Teacher `<TEACHER_REVIEW>...</TEACHER_REVIEW>`（loss=false）
`gold_answer`: `"17"` | `feedback_is_correct`: `true`

---

## 类型 4: `verify_accept_correction` — 答案错误，接受纠正

**权重**: 15% | **目的**: 训练模型在 teacher 指出有效错误时使用 `<ACCEPT>`

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User:
Find the number of ordered pairs (a, b) of complex numbers such that
a^4 b^6 = a^8 b^3 = 1.
Please reason step by step, and put your final answer within \boxed{}.

Assistant:
A tentative calculation gives \boxed{13}.
<VERIFY>128</VERIFY>

Teacher:
<TEACHER_REVIEW>
Verdict: incorrect
Issue: The tentative result 13 does not match the corrected derivation.
Correction: The corrected result is \boxed{12}.
Suggested answer: \boxed{12}
</TEACHER_REVIEW>

Assistant:
<ACCEPT>
From the equation a^4 b^6 = 1, we obtain a^8 b^12 = 1. From the equation
a^8 b^3 = 1, multiplying both sides by itself gives a^16 b^6 = 1.

Dividing the first equation a^8 b^12 = 1 by a^16 b^6 = 1 gives:
a^{-8} b^6 = 1 → b^6 = a^8.
This implies b = a^{8/6} = a^{4/3}.

Now, substituting back to check the values of a:
a^4 (a^{4/3})^6 = a^4 a^8 = a^{12} = 1.
Thus, a must be a 12th root of unity.

Substituting b = a^{4/3} into a^8 b^3 = 1, we get:
a^8 (a^{4/3})^3 = a^8 a^4 = a^{12} = 1.
This is consistent and does not introduce any new restrictions on a.

Therefore, the solutions (a, b) are of the form (ω, ω^{4/3}), where ω is a
12th root of unity, giving us \boxed{12} solutions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**SFT 可训练内容**：错误回答 + `<VERIFY>128</VERIFY>` + `<ACCEPT>` + 完整推理（loss=true）
**不可训练**：Teacher `<TEACHER_REVIEW>...</TEACHER_REVIEW>`（loss=false）
`gold_answer`: `"12"` | `feedback_is_correct`: `true`（teacher 的反馈是*正确*的——准确指出了错误）

---

## 类型 5: `verify_reject_bad_feedback` — 老师给错误反馈，拒绝

**权重**: 10% | **目的**: 训练模型**不要盲目信任**大模型，选择性采纳

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User:
When a right triangle is rotated about one leg, the volume of the cone produced
is 972π cm^3. When the triangle is rotated about the other leg, the volume of
the cone produced is 1458π cm^3. What is the length (in cm) of the hypotenuse
of the triangle?
Please reason step by step, and put your final answer within \boxed{}.

Assistant:
Let one leg of the triangle have length x and let the other leg have length y.
When we rotate around the leg of length x, the result is a cone of height x
and radius y, and so of volume (1/3)π·y·x² = 972π. Likewise, when we rotate
around the leg of length y, we get a cone of height y and radius x, and so of
volume (1/3)π·x·y² = 1458π.

Dividing these equations gives:
[(1/3)π·x·y²] / [(1/3)π·y·x²] = 1458/972 = 3/2
Thus, y²/x² = 3/2, implying y/x = √(3/2), so y = √(3/2)·x.

Substituting y = √(3/2)·x into (1/3)π·y·x² = 972π ...

[... detailed algebra ...]

h = 12 √(2 + 3) = 12 √5
\boxed{12 \sqrt{5}} \text{ cm}
<VERIFY>96</VERIFY>

Teacher:
<TEACHER_REVIEW>
Verdict: incorrect
Issue: Replace the final value with (12 √5)+1.
Correction: The final value is \boxed{(12 \sqrt{5})+1}.
Suggested answer: \boxed{(12 \sqrt{5})+1}
</TEACHER_REVIEW>

Assistant:
The proposed correction conflicts with the derivation, so I retain
\boxed{12 \sqrt{5}}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**SFT 可训练内容**：正确推理 + `<VERIFY>96</VERIFY>` + 拒绝并坚持原答案（loss=true）
**不可训练**：Teacher `<TEACHER_REVIEW>...</TEACHER_REVIEW>`（loss=false）
**关键点**：Student 是正确的（`12√5`），Teacher 故意给错误反馈（`(12√5)+1`）
`gold_answer`: `"12 \sqrt{5}"` | `feedback_is_correct`: `false`（teacher 的反馈是*错误*的）

---

## 类型 6: `verify_uncertain` — 老师不确定，独立判断

**权重**: 5% | **目的**: 训练模型在 teacher 也无法确定时，仍能独立完成推理

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User:
What is the greatest common divisor of 12345 and 6789?
Please reason step by step, and put your final answer within \boxed{}.

Assistant:
I have a tentative derivation but want an external check before finalizing.
<VERIFY>128</VERIFY>

Teacher:
<TEACHER_REVIEW>
Verdict: uncertain
Issue: The available check is inconclusive.
Correction: None
Suggested answer: None
</TEACHER_REVIEW>

Assistant:
We apply the Euclidean Algorithm to find the greatest common divisor of these
two integers.

gcd(12345, 6789)
= gcd(6789, 12345 - 6789)
= gcd(6789, 5556)
= gcd(5556, 6789 - 5556)
= gcd(5556, 1233)
= gcd(1233, 5556 - 4·1233)
= gcd(1233, 624)
= gcd(624, 1233 - 624)
= gcd(624, 609)
= gcd(609, 624 - 609)
= gcd(609, 15)
= gcd(15, 609 - 40·15)
= gcd(15, 9)
= gcd(9, 15 - 9)
= gcd(9, 6)
= gcd(6, 9 - 6)
= gcd(6, 3)
= gcd(3, 6 - 2·3)
= gcd(3, 0)

The remainder is now zero, thus the greatest common divisor is \boxed{3}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**SFT 可训练内容**：`<VERIFY>128</VERIFY>` + 独立完成的完整推理（loss=true）
**不可训练**：Teacher `<TEACHER_REVIEW>...</TEACHER_REVIEW>`（loss=false）
`gold_answer`: `"3"` | `feedback_is_correct`: `null`（teacher 没有给出确定性判断）

---

## 总结

| 类型 | 权重 | 核心教学目的 | 有 `<ASK>` | 有 `<VERIFY>` | 有 `<ACCEPT>` | Teacher 反馈 |
|---|---|---|---|---|---|---|
| normal | 45% | 独立解题，不求助 | — | — | — | 无 |
| ask_help | 15% | 求助能力不足时 | :white_check_mark: | — | — | 有帮助 |
| verify_confirm | 10% | 答对了不要慌 | — | :white_check_mark: | — | 确认正确 |
| verify_accept_correction | 15% | 答错了要接受纠正 | — | :white_check_mark: | :white_check_mark: | 指出错误 |
| verify_reject_bad_feedback | 10% | 老师错了不要信 | — | :white_check_mark: | — | **故意给错** |
| verify_uncertain | 5% | 老师不确定就靠自己 | — | :white_check_mark: | — | 无法判断 |

**统一规则**：
- 所有 Teacher 反馈（`<TEACHER_HELP>`, `<TEACHER_REVIEW>`）均为 `loss=false`，**不参与梯度更新**
- 所有 Student 文本（含 `<ASK>`, `<VERIFY>`, `<ACCEPT>` 等动作 token）均为 `loss=true`
- `feedback_is_correct` 标记的是Teacher 反馈是否正确，而非 Student 原答案是否正确
