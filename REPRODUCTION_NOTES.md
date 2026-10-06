# OPSD local reproduction notes

Cluster log from the 1×A800 reproduction (paths below are machine-specific).
For a portable setup — Hub model ids, install, and launch commands — use [`README.md`](README.md).

The paper used for cross-checking is [`OPSD.pdf`](OPSD.pdf). The base model
is [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B); on the original
host it lived at `/home/aochang/kongjing/model/Qwen3-1.7B`.

## Code and framework

The training entry points are `opsd_train.py` (OPSD), `sft_train.py` (SFT baseline),
and `grpo_train.py` (GRPO baseline). `OPSDTrainer` subclasses TRL's experimental
`GOLD`/`SFTTrainer`. Training uses PyTorch, Transformers, TRL, Accelerate and
DeepSpeed; vLLM supplies on-policy rollouts; PEFT/LoRA supplies the trainable
parameters. `eval/evaluate_math.py` uses vLLM and `math-verify` for evaluation.

The environment is installed at `/data/shared/users/changao/envs/opsd` (Python 3.10,
PyTorch 2.8.0+cu128, Transformers 4.57.1, TRL 0.26.0, Accelerate 1.11.0,
DeepSpeed 0.18.2, vLLM 0.11.0, PEFT 0.17.1, Datasets 3.6.0). Use the environment
explicitly when launching, for example:

```bash
cd /home/aochang/kongjing/OPSD
PATH=/data/shared/users/changao/envs/opsd/bin:$PATH \
  HF_HOME=/data/shared/users/changao/cache/opsd-hf \
  HF_DATASETS_CACHE=/data/shared/users/changao/cache/opsd-hf/datasets \
  bash scripts/run_opsd_1b_local.sh
```

The launcher defaults to the verified single-GPU settings (`CUDA_DEVICES=0`,
`NUM_PROCESSES=1`, batch 2, vLLM utilization 0.30). On a node with four idle
GPUs, the paper-like launch can be selected explicitly, e.g.
`CUDA_DEVICES=0,1,2,3 NUM_PROCESSES=4 PER_DEVICE_TRAIN_BATCH_SIZE=4
VLLM_GPU_MEMORY_UTILIZATION=0.60 bash scripts/run_opsd_1b_local.sh`.

The local launcher defaults to `accelerate_local.yaml`, which keeps optimizer state
on GPU. The upstream `accelerate.yaml` enables CPU optimizer offload; on this host
that attempts to compile DeepSpeed CPUAdam and fails because the installed toolkit is
CUDA 13.0 while the PyTorch wheel is cu128. FlashAttention 2 could not be installed
because `nvcc` is unavailable, so the local launcher defaults to Transformers SDPA;
set `ATTN_IMPLEMENTATION=flash_attention_2` only when a matching wheel is available.

## Runs completed

All runs use the downloaded `siyanzhao/Openthoughts_math_30k_opsd` dataset (29,434
examples) and the main paper setting (student thinking off, teacher thinking on,
fixed teacher, LoRA rank 64/alpha 128, forward KL `beta=0`, pointwise clip 0.05).

* `runs/smoke_no_offload`: one-step smoke test, completion limit 16, max length 512,
  one GPU, batch 1. vLLM rollout, backward pass and adapter checkpoint completed;
  `train_loss=1.013671875`.
* `runs/short_5steps`: five-step check, completion limit 128, max length 2048;
  `train_runtime=14.6753 s`, `train_loss=0.00223484`.
* `runs/repro_100_gpu0`: 100-step run on the only idle GPU (GPU 0), batch 2
  (effective batch 4), completion limit 1024, max length 20,000. It completed in
  `1154.8 s` (~19 min 15 s), with `train_loss=-0.0006760`; checkpoints 25, 50, 75
  and 100 plus rollout JSON files are present.

The single-GPU run is an executable reproduction of the algorithm, not a strict
reproduction of the paper's distributed setting (the paper used 8 A100/H100 GPUs,
effective batch 32, FlashAttention 2, and reports the best checkpoint through step
100).

## Evaluation checks

Early smoke tests (2 AIME24 problems, `val_n=2`, 8,192-token cap) only showed that
the evaluator could load the base model and the 100-step adapter. Those runs are
not accuracy comparisons.

A full paper-protocol evaluation was then run for Qwen3-1.7B **Base** and the local
**100-step OPSD** merge (`runs/repro_100_gpu0/merged-100`) on AIME24, AIME25 and
HMMT25. Protocol matches the README thinking-mode settings: temperature 1.0,
thinking enabled, `max_new_tokens=38912`, `val_n=12`, `top_k` disabled, `min_p=0`,
presence penalty 0. The eval script auto-sets `top_p=0.95` when the flag is omitted,
same as `eval/run_eval.sh`. Chunks were evaluated on leftover GPU memory (do not
stop other users' jobs) and merged with `eval/merge_chunk_results.py`. Summary
files:

* `eval_results/aime24_base_merged.json` / `aime24_opsd100_merged.json`
* `eval_results/aime25_base_merged.json` / `aime25_opsd_merged.json`
* `eval_results/hmmt25_base_merged.json` / `hmmt25_opsd_merged.json`
* `eval_results/paper_table_comparison.json`

Paper thinking-mode Qwen3-1.7B Avg@12 from the README (step 100, and the best
checkpoint when it differs):

| Benchmark | Paper Base | Local Base | Paper @100 | Local OPSD-100 | Paper best |
|---|---:|---:|---:|---:|---:|
| AIME24 | 51.5 | 50.28 (181/360) | 57.2 | 56.11 (202/360) | 57.2 @100 |
| AIME25 | 36.7 | 33.89 (122/360) | 41.1 | 42.22 (152/360) | 43.9 @50 |
| HMMT25 | 23.1 | 23.06 (83/360) | 29.2 | 22.78 (82/360) | 29.2 @100 |

Local Base vs OPSD-100 Pass@12 / majority-vote@12:

| Benchmark | Base Pass@12 | OPSD-100 Pass@12 | Base Maj@12 | OPSD-100 Maj@12 |
|---|---:|---:|---:|---:|
| AIME24 | 73.3 | 76.7 | 70.0 | 70.0 |
| AIME25 | 66.7 | 66.7 | 43.3 | 60.0 |
| HMMT25 | 46.7 | 43.3 | 26.7 | 26.7 |

Per-problem correct-count (out of 12) vs Base: AIME24 improved 11 / worse 6 /
same 13 (net +21 solutions); AIME25 improved 15 / worse 3 / same 12 (net +30);
HMMT25 improved 6 / worse 6 / same 18 (net −1).

### What this does and does not show

* The training + eval pipeline runs end-to-end on this machine.
* Local Base is within about 0–3 Avg@12 points of the paper Base (HMMT25 matches
  to 0.04). That is evidence the eval protocol and datasets are aligned.
* Local 100-step OPSD recovers the paper's AIME24 and AIME25 step-100 numbers
  within about 1 point, despite training on 1×A800 with effective batch 4, SDPA,
  and a 1,024-token completion cap instead of the paper's 4×H100 / effective
  batch 32 / FlashAttention 2 setup.
* HMMT25 did **not** reproduce the paper's +6.1 gain at step 100. With a single
  decoding seed the README already warns that Avg@12 will move around; the local
  run also used a much smaller effective batch, so this gap is not by itself a
  code bug.
* The 43.9 AIME25 figure in some earlier notes is the paper's **step 50** best,
  not step 100 (41.1). This local job only evaluated the 100-step merge.
