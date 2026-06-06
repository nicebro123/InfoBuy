# InfoBuy 存储与目录规划

目标：代码仓库只放代码、配置和小文档；所有模型权重、Hugging Face 缓存、预下载数据集、生成数据、训练 checkpoint、评测结果都放到仓库外的 `$INFOBUY_STORE`。

## 1. 总体原则

```text
/Users/quanquan/Desktop/InfoBuy/        代码仓库，不放大文件
/Users/quanquan/Desktop/InfoBuy_store/  统一实验存储根目录
```

原则：

```text
代码       -> InfoBuy/
模型权重   -> InfoBuy_store/models/
HF 缓存    -> InfoBuy_store/huggingface/
数据集     -> InfoBuy_store/datasets/
训练权重   -> InfoBuy_store/checkpoints/
评测输出   -> InfoBuy_store/eval/
日志       -> InfoBuy_store/logs/
历史备份   -> InfoBuy_store/backups/
临时文件   -> InfoBuy_store/tmp/
```

## 2. 外部 Store 目录结构

```text
$INFOBUY_STORE/
├── huggingface/
│   ├── cache/                    # HF_HOME
│   ├── hub/                      # HF_HUB_CACHE
│   └── datasets_cache/           # HF_DATASETS_CACHE
├── models/
│   ├── pretrained/
│   │   ├── Qwen3-0.6B/           # student base
│   │   └── Qwen3-8B/             # main teacher
│   └── teachers/
│       └── qwen3-8b-main -> ../pretrained/Qwen3-8B
├── datasets/
│   ├── hf_downloads/
│   │   ├── AI-MO__NuminaMath-CoT/
│   │   └── optional_baselines/
│   │       ├── dapo17k_raw/
│   │       └── dapo17k_8b_filtered/
│   ├── benchmarks/
│   │   ├── math500/
│   │   ├── gsm8k/
│   │   ├── aime2024/
│   │   ├── aime2025/
│   │   ├── amc23/
│   │   ├── minerva/
│   │   └── olympiadbench/
│   └── infobuy/
│       ├── raw/                  # 去污染 NuminaMath raw split
│       ├── protocol/             # Ask / Verify / Accept 协议 SFT 数据
│       ├── flat/                 # flattened SFT 数据
│       ├── replay/               # outcome-selected rollout replay
│       ├── trust/                # 反馈采纳/拒绝可信数据
│       ├── purchase/             # 信息购买策略数据
│       ├── manifests/            # provenance / split / decontamination
│       └── splits/
├── checkpoints/
│   ├── sft/                      # SFT 输出
│   ├── rl/                       # GRPO 输出
│   ├── merged/                   # 合并后的 HF 格式模型
│   ├── intermediate/
│   └── filter/
├── eval/
│   ├── evaluation/               # eval/generate_withhelp.py 原始输出
│   ├── raw_results/
│   ├── rechecked/
│   ├── summaries/
│   └── reports/
├── logs/
│   ├── sft/
│   ├── rl/
│   ├── vllm/
│   ├── eval/
│   └── wandb/
├── services/
│   ├── teacher_vllm/
│   └── student_vllm/
├── backups/                     # 从代码仓库移出的历史数据备份
└── tmp/
    ├── downloads/
    ├── extraction/
    └── debug/
```

## 3. 环境变量

统一从仓库根目录加载：

```bash
cd /Users/quanquan/Desktop/InfoBuy
export INFOBUY_STORE=/Users/quanquan/Desktop/InfoBuy_store
source setup/env.sh
```

`setup/env.sh` 会导出：

```bash
INFOBUY_STORE=$INFOBUY_STORE
INFOBUY_MODELS=$INFOBUY_STORE/models
INFOBUY_DATASETS=$INFOBUY_STORE/datasets
INFOBUY_CKPT=$INFOBUY_STORE/checkpoints
INFOBUY_GENERATED_DATA=$INFOBUY_STORE/datasets/infobuy
INFOBUY_HF_DOWNLOADS=$INFOBUY_STORE/datasets/hf_downloads
INFOBUY_BENCHMARKS=$INFOBUY_STORE/datasets/benchmarks
INFOBUY_BACKUPS=$INFOBUY_STORE/backups
STORAGE_PATH=$INFOBUY_STORE/eval

HF_HOME=$INFOBUY_STORE/huggingface/cache
HF_HUB_CACHE=$INFOBUY_STORE/huggingface/hub
HF_DATASETS_CACHE=$INFOBUY_STORE/huggingface/datasets_cache
```

为兼容旧脚本，`setup/env.sh` 也会导出 `HSP_STORE`、`HSP_CKPT`、`HSP_GENERATED_DATA` 等别名，但新文档和新脚本优先使用 `INFOBUY_*`。

## 4. 仓库内 data 的规则

默认架构不要求仓库内存在 `data/`。

主线配置应直接读取：

```text
$INFOBUY_GENERATED_DATA/raw/...
$INFOBUY_GENERATED_DATA/protocol/...
$INFOBUY_GENERATED_DATA/replay/...
```

`setup/link_data.sh` 只作为旧命令兼容桥接。只有当某个旧命令硬编码 `data/...` 时，才手动执行：

```bash
source setup/env.sh
bash setup/link_data.sh
```

该脚本会创建：

```text
InfoBuy/data -> $INFOBUY_STORE/datasets
```

这不是默认要求。

## 5. 当前主线数据

InfoBuy 主线使用去污染后的 NuminaMath-CoT `synthetic_math`：

```text
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
$INFOBUY_GENERATED_DATA/protocol/hsp_protocol_train_pilot_v1.jsonl
$INFOBUY_GENERATED_DATA/protocol/hsp_protocol_validation_pilot_v1.jsonl
```

当前 GRPO 配置读取：

```yaml
data:
  train_files: ${oc.env:INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
  val_files: ${oc.env:INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
  prompt_key: question
  answer_key: gold_answer
```

DAPO 不作为主线训练数据，只放在可选 baseline / ablation：

```text
$INFOBUY_HF_DOWNLOADS/optional_baselines/
```

## 6. 脚本职责

| 脚本 | 职责 |
|---|---|
| `setup/env.sh` | 导出 InfoBuy 外部目录变量 |
| `setup/make_dirs.sh` | 创建完整 `$INFOBUY_STORE` 目录树 |
| `setup/link_data.sh` | 可选：为旧 `data/...` 命令建立兼容软链接 |
| `setup/download_models.sh` | 下载 student / teacher 权重到 `$INFOBUY_PRETRAINED_MODELS` |
| `setup/download_data.sh` | 下载 HF 数据快照与 benchmark 缓存 |

## 7. 目录检查清单

```text
[ ] InfoBuy/ 只放代码、配置、文档、小测试 fixture
[ ] INFOBUY_STORE 指向外部大文件目录
[ ] HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE 都在 INFOBUY_STORE 下
[ ] 预训练权重在 INFOBUY_STORE/models/pretrained/
[ ] teacher alias 在 INFOBUY_STORE/models/teachers/
[ ] 主线数据在 INFOBUY_STORE/datasets/infobuy/
[ ] SFT checkpoint 在 INFOBUY_STORE/checkpoints/sft/
[ ] RL checkpoint 在 INFOBUY_STORE/checkpoints/rl/
[ ] 评测输出在 INFOBUY_STORE/eval/
[ ] DAPO 没有被误设为主线 InfoBuy RL 数据
```
