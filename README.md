# Boxed-Error Reweighting for On-Policy Self-Distillation

Sun YuXin*, Shen Lixin*, Kong Jing*, and Chen Yitong*  
College of Computing and Data Science, Nanyang Technological University  
*Equal contribution.

Course-project extension of [On-Policy Self-Distillation (OPSD)](https://arxiv.org/abs/2601.18734) (Zhao et al., 2026). One small language model is both **student** (problem only) and **teacher** (problem + gold solution). We add a cheap, verifiable **outcome gate**: after each student rollout, extract `\boxed{}`, check it with `math-verify`, and upweight the distillation loss **only** when the model produced a complete answer that is wrong.

This repository is meant to be pushed to GitHub as-is: **code, configs, paper LaTeX, and metric summaries**. Base-model weights and trained checkpoints are downloaded or produced locally (see below).

<p align="center">
<a href="https://arxiv.org/pdf/2601.18734v3"><img src="https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg" alt="arXiv"></a>
<a href="https://siyan-zhao.github.io/blog/2026/opsd/"><img src="https://img.shields.io/badge/Blog-OPSD-blue.svg" alt="OPSD blog"></a>
</p>

---

## What is in this repo

| Path | Role |
|---|---|
| `opsd_trainer.py` | `OPSDTrainer`: token-level on-policy distillation + sequence weights |
| `opsd_train.py` | Training entry (`--outcome_gate_lambda`, `--outcome_gate_mode`) |
| `outcome_reward.py` | Boxed-answer extract + `math-verify` reward \(R \in \{0,1\}\) |
| `data_collator.py` | Student / teacher prompts; gold answers for the gate |
| `eval/evaluate_math.py` | vLLM eval (AIME24/25, HMMT25, …) |
| `scripts/run_opsd_1b_local.sh` | 1-GPU LoRA reproduction + BER-OPSD launcher |
| `paper/` | AAAI-style write-up (`ogopsd.tex`) |
| `eval_results/*comparison*.json` | Metric tables from the local 100-step runs |
| `OPSD.pdf` | Original OPSD paper |

**Download or train locally** (too large for git):

- Base LM weights (`Qwen3-1.7B` ≈ 3–4 GB)
- LoRA adapters (≈ 1.7 GB per checkpoint)
- Merged eval models (≈ 3.3 GB)
- Full generation JSON dumps (≈ 20–24 MB per suite)

---

## Download the base model

Training and eval start from **Qwen3-1.7B**. Pull the weights from Hugging Face once:

```bash
# Hugging Face repo: https://huggingface.co/Qwen/Qwen3-1.7B
pip install -U "huggingface_hub[cli]"

huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B
```

Transformers will also download the repo on the fly if you pass the Hub id:

```bash
export MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B
# or a local snapshot
export MODEL_NAME_OR_PATH=./models/Qwen3-1.7B
```

Larger paper variants (optional, not used in the course runs):

| Model | Hub id | Typical use |
|---|---|---|
| Qwen3-1.7B | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) | Main BER-OPSD / OPSD reproduction |
| Qwen3-4B | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | Upstream non-thinking / 4B scripts |
| Qwen3-8B | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Upstream 8B scripts |

License follows the Hub card (Qwen3 is Apache-2.0). You need a Hugging Face account only if the repo requires it; Qwen3-1.7B is public.

### Training data (auto-download)

The trainer calls:

```python
load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
```

Hub: [`siyanzhao/Openthoughts_math_30k_opsd`](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd) (29,434 math traces). First run caches it under `HF_HOME`.

### Evaluation data (auto-download)

| Flag | Hub dataset |
|---|---|
| `--dataset aime24` | [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| `--dataset aime25` | [`yentinglin/aime_2025`](https://huggingface.co/datasets/yentinglin/aime_2025) |
| `--dataset hmmt25` | [`MathArena/hmmt_feb_2025`](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) |

### Trained checkpoints (train them yourself)

After `bash scripts/run_opsd_1b_local.sh`, adapters land in `runs/<run_config>/checkpoint-{25,50,75,100}/`. Merge before paper-protocol eval if you want a standalone model. Those files are gitignored on purpose.

---

## Method (BER-OPSD)

OPSD matches the student token distribution to a privileged teacher **on the student’s own rollout**. The original loss ignores correctness; with thinking off and a 1,024-token completion cap, many traces truncate without a boxed answer.

BER-OPSD leaves the KL / JSD term unchanged and multiplies each sequence by a weight:

\[
w = 1 + \lambda \cdot g \cdot (1 - R)
\]

- \(R \in \{0,1\}\): `math-verify` of the boxed student answer vs gold
- **`boxed_error` (main):** \(g=1\) only if a `\boxed{}` exists **and** \(R=0\). Truncated / unboxed traces stay at weight 1
- **`all_error` (ablation):** \(g=1\) for every incorrect rollout, including truncations
- \(\lambda=0\) recovers vanilla OPSD

The gate only reweights the existing KL / JSD term. There is no extra teacher model, group sampling, or advantage.

---

## Local results (Qwen3-1.7B, thinking eval)

Protocol matches the OPSD paper: temperature 1.0, thinking on, `max_new_tokens=38912`, `val_n=12`. Training: 100 LoRA steps, rank 64 / \(\alpha=128\), 1×A800, effective batch 4, student thinking **off**, completion 1024, \(\beta=0\), clip 0.05.

**Avg@12**

| Method | AIME24 | AIME25 | HMMT25 | Macro |
|---|---:|---:|---:|---:|
| Base Qwen3-1.7B | 50.28 | 33.89 | 23.06 | 35.74 |
| OPSD (reproduced) | 56.11 | 42.22 | 22.78 | 40.37 |
| BER-OPSD All-Error \(\lambda=1\) | 57.50 | 40.00 | 23.61 | 40.37 |
| **BER-OPSD Boxed-Error \(\lambda=1\)** | 55.28 | 38.33 | **27.78** | 40.46 |
| Paper OPSD @100 (4×H100) | 57.2 | 41.1 | 29.2 | — |

Boxed-Error lifts HMMT25 Avg@12 by **+5.0** vs local OPSD (Pass@12 43.3→56.7, Maj@12 26.7→33.3). AIME stays at or below reproduced OPSD; all three suites are in the table. Machine-readable numbers: [`eval_results/ogopsd_full_comparison.json`](eval_results/ogopsd_full_comparison.json).

---

## Install

Python 3.10, a recent CUDA GPU, and the packages in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate opsd

# Optional, if you have a matching CUDA toolkit + nvcc:
pip install flash-attn==2.8.3 --no-build-isolation
```

If FlashAttention is unavailable, the local launcher defaults to PyTorch SDPA (`ATTN_IMPLEMENTATION=sdpa`).

The trainer builds on TRL’s experimental GOLD trainer.

---

## Training

### Vanilla OPSD (1 GPU, 100 steps)

```bash
export MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B   # or ./models/Qwen3-1.7B
bash scripts/run_opsd_1b_local.sh
```

This is \(\lambda=0\), i.e. original OPSD. On 1×A800 it finishes in about 20 minutes.

Paper-style 4-GPU launch (same hyperparameters as `scripts/run_opsd_1b.sh`):

```bash
CUDA_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
  PER_DEVICE_TRAIN_BATCH_SIZE=4 \
  VLLM_GPU_MEMORY_UTILIZATION=0.60 \
  ATTN_IMPLEMENTATION=flash_attention_2 \
  bash scripts/run_opsd_1b_local.sh
```

### BER-OPSD (boxed-error gate)

```bash
export MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B
OUTCOME_GATE_LAMBDA=1 \
OUTCOME_GATE_MODE=boxed_error \
RUN_CONFIG=ogopsd_boxederr_l1_100 \
bash scripts/run_opsd_1b_local.sh
```

Ablation (upweight every error, including truncations):

```bash
OUTCOME_GATE_LAMBDA=1 \
OUTCOME_GATE_MODE=all_error \
RUN_CONFIG=ogopsd_lambda1_100 \
bash scripts/run_opsd_1b_local.sh
```

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--outcome_gate_lambda` | `0` | \(0\) = vanilla OPSD |
| `--outcome_gate_mode` | `all_error` | `boxed_error` = main method |
| `--fixed_teacher` | on in scripts | Teacher stays at step-0 LoRA |
| `--beta` | `0` | Forward KL |
| `--jsd_token_clip` | `0.05` | Per-token clip |
| `--max_completion_length` | `1024` | Student rollout cap |
| `--use_peft` / `--lora_r 64` | on in scripts | LoRA on q/k/v/o/gate/up/down |

SFT / GRPO baselines: `scripts/run_sft.sh`, `scripts/run_grpo.sh`.

---

## Evaluation

Paper protocol (thinking mode):

```bash
cd eval
python evaluate_math.py \
  --base_model ../models/Qwen3-1.7B \
  --dataset aime24 \
  --val_n 12 \
  --temperature 1.0 \
  --tensor_parallel_size 1 \
  --checkpoint_dir ../runs/<run>/checkpoint-100
```

Repeat with `--dataset aime25` and `--dataset hmmt25`. The evaluator uses `max_new_tokens=38912` and thinking on unless you pass `--no_thinking`.

---

## Repository layout

```
.
├── opsd_trainer.py / opsd_train.py / data_collator.py / outcome_reward.py
├── sft_train.py / grpo_train.py
├── accelerate.yaml / accelerate_local.yaml
├── environment.yml / requirements-lock.txt
├── scripts/                 # launchers (Hub ids, local output dirs)
├── eval/                    # vLLM math eval
├── eval_results/            # Avg@12 / Pass@12 / Maj@12 summaries
├── paper/                   # ogopsd.tex + AAAI 2026 style
├── backups/original-ae7d251 # pre-gate snapshot of trainer files
└── OPSD.pdf                 # original paper
```

Cluster-specific reproduction log: [`REPRODUCTION_NOTES.md`](REPRODUCTION_NOTES.md). Rollback to vanilla OPSD files: [`backups/ROLLBACK.md`](backups/ROLLBACK.md).

---

## Upstream OPSD

The distillation core follows Zhao et al. Original resources:

- Paper: <https://arxiv.org/abs/2601.18734>
- Blog: <https://siyan-zhao.github.io/blog/2026/opsd/>
- Implementation based on [TRL GOLD](https://huggingface.co/docs/trl/gold_trainer)

---

## Citation

Original OPSD:

```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```

This repository’s variant:

```bibtex
@article{sun2026beropsd,
  title={Boxed-Error Reweighting for On-Policy Self-Distillation},
  author={Sun, YuXin and Shen, Lixin and Kong, Jing and Chen, Yitong},
  note={Equal contribution. College of Computing and Data Science, Nanyang Technological University},
  year={2026}
}
```

The write-up is in `paper/ogopsd.tex`. Wang et al., arXiv:2610.05070, use the name OG-OPSD for a different outcome-guided method.
