# Evaluation summaries

This folder keeps **metric tables only**. Full per-problem generation dumps
(`aime24_*.json`, `aime25_*.json`, `hmmt25_*.json`, typically 20–24 MB each)
are not in git; they live on the training machine under the same filenames.

| File | Contents |
|---|---|
| `paper_table_comparison.json` | Local Base / OPSD-100 vs paper Qwen3-1.7B thinking-mode Avg@12 |
| `ogopsd_v1_comparison.json` | All-Error gate (`lambda=1`) vs reproduced OPSD |
| `ogopsd_full_comparison.json` | Base, OPSD-100, All-Error (v1), Boxed-Error (v2) |

Re-run evaluation with `eval/evaluate_math.py` after training. Protocol:

```
temperature=1.0, thinking on, max_new_tokens=38912, val_n=12, top_p=0.95
```
