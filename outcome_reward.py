"""Sequence-level verifiable correctness used by OG-OPSD.

Default OPSD does not call this path. When --outcome_gate_lambda > 0, student
rollouts are graded against the dataset gold answer with math_verify, and the
per-sequence distillation loss is scaled by (1 + lambda * (1 - R)).
"""

from __future__ import annotations

import re
from typing import Optional

from math_verify import parse, verify


def extract_boxed_answer(text: Optional[str]) -> Optional[str]:
    """Extract the last \\boxed{...} answer, searching after </think> when present."""
    if not text:
        return None
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        # last boxed in the whole string (some rollouts omit </think>)
        idx = search_text.rfind(r"\boxed{")
        if idx == -1:
            return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def grade_answer(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """Grade a predicted boxed answer against gold using math_verify, then string fallback."""
    if predicted is None or ground_truth is None:
        return False
    pred = str(predicted).strip()
    gold = str(ground_truth).strip()
    if not pred or not gold:
        return False

    gold_parsed = parse(gold)
    pred_parsed = parse(_preprocess_for_parse(pred))
    if gold_parsed is not None and pred_parsed is not None:
        try:
            if verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass

    # evaluate_math-style $ wrapping, in case parse() wanted latex math mode
    try:
        pred_d = pred if "$" in pred else f"${pred}$"
        gold_d = gold if "$" in gold else f"${gold}$"
        pred_parsed = parse(pred_d, fallback_mode="no_fallback")
        gold_parsed = parse(gold_d, fallback_mode="no_fallback")
        if gold_parsed is not None and pred_parsed is not None:
            if verify(gold_parsed, pred_parsed, timeout_seconds=5):
                return True
    except Exception:
        pass

    pred_norm = re.sub(r"\s+", "", pred).replace("$", "").lower()
    gold_norm = re.sub(r"\s+", "", gold).replace("$", "").lower()
    return bool(pred_norm) and pred_norm == gold_norm


def outcome_reward(completion: str, gold: str) -> tuple[float, bool]:
    """Return (R in {0,1}, whether a boxed answer was found)."""
    pred = extract_boxed_answer(completion)
    boxed = pred is not None
    if not boxed:
        return 0.0, False
    return (1.0 if grade_answer(pred, gold) else 0.0), True
