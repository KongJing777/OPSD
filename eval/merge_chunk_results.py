#!/usr/bin/env python
"""Merge contiguous evaluate_math.py chunk JSON files into one summary."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Chunk JSON files")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    chunks = [json.loads(Path(name).read_text()) for name in args.inputs]
    results = [item for chunk in chunks for item in chunk.get("results", [])]
    results.sort(key=lambda x: int(x["problem_id"]) if str(x.get("problem_id", "")).isdigit() else str(x.get("problem_id", "")))
    total = sum(int(x.get("val_n", 1)) for x in results)
    correct = sum(int(x.get("num_correct", 0)) for x in results)
    passed = sum(bool(x.get("pass_at_n", False)) for x in results)
    majority = sum(bool(x.get("majority_vote_correct", False)) for x in results)
    formatted = sum(sum(bool(g.get("formatted", False)) for g in x.get("generations", [])) for x in results)
    out = {k: chunks[0].get(k) for k in ("base_model", "dataset", "enable_thinking", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "max_new_tokens", "val_n")}
    out.update({
        "num_problems": len(results), "total_solutions": total,
        "pass_at_n": passed, "pass_at_n_pct": 100 * passed / len(results),
        "average_at_n": correct, "average_at_n_pct": 100 * correct / total,
        "majority_vote_at_n": majority, "majority_vote_at_n_pct": 100 * majority / len(results),
        "formatted_count": formatted, "format_rate": 100 * formatted / total,
        "source_chunks": args.inputs, "results": results,
    })
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in out if k not in ("results", "source_chunks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
