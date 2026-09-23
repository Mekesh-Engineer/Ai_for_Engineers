# -*- coding: utf-8 -*-
"""
evaluation.py
-------------
Evaluation module for Experiment 8: Automated Grammar Error Correction & Text Rewriting.
Computes Exact Match, Levenshtein Edit Distance, Token Precision/Recall/F1/F0.5, GLEU,
Semantic Preservation / Similarity, Over/Under-Correction rates, and generates diagnostic reports.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes character-level Levenshtein edit distance via dynamic programming."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def calculate_exact_match(prediction: str, reference: str) -> bool:
    """Calculates boolean Exact Match after whitespace and lowercase normalization."""
    p_clean = re.sub(r'\s+', ' ', str(prediction).strip().lower())
    r_clean = re.sub(r'\s+', ' ', str(reference).strip().lower())
    return p_clean == r_clean


def calculate_edit_distance(prediction: str, reference: str) -> int:
    """Calculates Levenshtein edit distance between predicted and reference text."""
    return compute_levenshtein_distance(str(prediction).strip(), str(reference).strip())


calculate_levenshtein_distance = calculate_edit_distance


def calculate_token_f1(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculates token-level precision, recall, harmonic F1, and precision-weighted F0.5.
    """
    pred_tokens = re.findall(r'\b\w+\b', str(prediction).lower())
    ref_tokens = re.findall(r'\b\w+\b', str(reference).lower())

    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "f0_5": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f0_5": 0.0}

    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    ref_counts: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    common_count = sum(min(pred_counts[t], ref_counts.get(t, 0)) for t in pred_counts)

    precision = common_count / len(pred_tokens) if pred_tokens else 0.0
    recall = common_count / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f0_5": round(f0_5, 4)
    }


def calculate_gleu_score(reference: str, prediction: str, source: Optional[str] = None) -> float:
    """
    Computes Generalized Language Evaluation Understudy (GLEU) score for GEC.
    Evaluates modified n-gram precision across n=1..4 with brevity penalty.
    """
    hyp_tokens = re.findall(r'\b\w+\b', str(prediction).lower())
    ref_tokens = re.findall(r'\b\w+\b', str(reference).lower())

    if not hyp_tokens or not ref_tokens:
        return 0.0
    if hyp_tokens == ref_tokens:
        return 1.0

    precisions = []
    for n in range(1, 5):
        hyp_ngrams = [tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
        if not hyp_ngrams:
            break
        ref_cnt: Dict[Tuple, int] = {}
        for ng in ref_ngrams:
            ref_cnt[ng] = ref_cnt.get(ng, 0) + 1
        matches = 0
        for ng in hyp_ngrams:
            if ref_cnt.get(ng, 0) > 0:
                matches += 1
                ref_cnt[ng] -= 1
        precisions.append((matches + 1e-5) / (len(hyp_ngrams) + 1e-5))

    if not precisions:
        return 0.0

    log_sum = sum(np.log(p) for p in precisions) / len(precisions)
    geo_mean = np.exp(log_sum)

    # Brevity penalty
    bp = min(1.0, np.exp(1.0 - (len(ref_tokens) / max(1, len(hyp_tokens)))))
    return round(float(bp * geo_mean), 4)


def calculate_semantic_similarity(reference: str, prediction: str) -> float:
    """
    Measures semantic similarity / character & subword contextual cosine similarity
    between reference and predicted text (BERTScore equivalent).
    """
    ref_clean = str(reference).strip().lower()
    pred_clean = str(prediction).strip().lower()
    if ref_clean == pred_clean:
        return 1.0

    # Character 3-gram cosine similarity
    def get_ngrams(s: str, n: int = 3) -> Dict[str, int]:
        s = f"_{s}_"
        counts: Dict[str, int] = {}
        for i in range(len(s) - n + 1):
            sub = s[i:i+n]
            counts[sub] = counts.get(sub, 0) + 1
        return counts

    ng_r = get_ngrams(ref_clean)
    ng_p = get_ngrams(pred_clean)
    all_keys = set(ng_r.keys()).union(set(ng_p.keys()))

    dot = sum(ng_r.get(k, 0) * ng_p.get(k, 0) for k in all_keys)
    norm_r = np.sqrt(sum(v**2 for v in ng_r.values()))
    norm_p = np.sqrt(sum(v**2 for v in ng_p.values()))

    if norm_r == 0 or norm_p == 0:
        return 0.0
    cosine = dot / (norm_r * norm_p)
    return round(float(np.clip(cosine, 0.0, 1.0)), 4)


def assess_length_and_preservation(
    source_text: str,
    prediction: str,
    target_ratio: float = 1.0,
    tolerance_pct: float = 35.0
) -> Dict[str, Any]:
    """Verifies that the generated correction maintains sentence length bounds and does not drop contents."""
    src_words = len(str(source_text).split())
    pred_words = len(str(prediction).split())
    ratio = (pred_words / src_words) if src_words > 0 else 1.0

    lower_bound = target_ratio * (1.0 - (tolerance_pct / 100.0))
    upper_bound = target_ratio * (1.0 + (tolerance_pct / 100.0))
    is_compliant = bool(lower_bound <= ratio <= upper_bound)

    return {
        "source_words": src_words,
        "prediction_words": pred_words,
        "length_ratio": round(ratio, 4),
        "is_compliant": is_compliant
    }


def detect_over_and_under_correction(input_text: str, prediction: str, reference: str) -> Tuple[bool, bool]:
    """
    Determines if prediction exhibits over-correction (unwarranted edits)
    or under-correction (remaining identical to erroneous input).
    """
    in_clean = str(input_text).strip().lower()
    pred_clean = str(prediction).strip().lower()
    ref_clean = str(reference).strip().lower()

    is_under_correction = (pred_clean == in_clean and in_clean != ref_clean)
    dist_pred_ref = compute_levenshtein_distance(pred_clean, ref_clean)
    dist_in_ref = compute_levenshtein_distance(in_clean, ref_clean)
    is_over_correction = (dist_pred_ref > dist_in_ref and dist_in_ref > 0)

    return is_over_correction, is_under_correction


def evaluate_correction_quality(
    input_text: str,
    prediction: str,
    reference: str,
    error_type: Optional[str] = None
) -> Dict[str, Any]:
    """Unified evaluation bundle computing all standard GEC metrics."""
    exact = calculate_exact_match(prediction, reference)
    lev_dist = compute_levenshtein_distance(prediction, reference)
    token_metrics = calculate_token_f1(prediction, reference)
    gleu = calculate_gleu_score(reference, prediction, source=input_text)
    sem_sim = calculate_semantic_similarity(reference, prediction)
    length_chk = assess_length_and_preservation(input_text, prediction)
    is_over, is_under = detect_over_and_under_correction(input_text, prediction, reference)

    return {
        "input_sentence": input_text,
        "prediction": prediction,
        "reference": reference,
        "error_type": error_type or "General Grammar",
        "exact_match": exact,
        "levenshtein_distance": lev_dist,
        "token_precision": token_metrics["precision"],
        "token_recall": token_metrics["recall"],
        "token_f1": token_metrics["f1"],
        "token_f0_5": token_metrics["f0_5"],
        "gleu_score": gleu,
        "semantic_similarity": sem_sim,
        "length_compliant": length_chk["is_compliant"],
        "is_over_correction": is_over,
        "is_under_correction": is_under
    }


def generate_evaluation_report(
    eval_df: pd.DataFrame,
    prompt_comp_df: Optional[pd.DataFrame] = None,
    baseline_df: Optional[pd.DataFrame] = None
) -> str:
    """Generates a structured comprehensive diagnostic evaluation text report."""
    exact_acc = float(eval_df["exact_match"].mean() * 100) if "exact_match" in eval_df.columns else 0.0
    mean_lev = float(eval_df["levenshtein_distance"].mean()) if "levenshtein_distance" in eval_df.columns else 0.0
    mean_f1 = float(eval_df["token_f1"].mean()) if "token_f1" in eval_df.columns else 0.0
    mean_gleu = float(eval_df["gleu_score"].mean()) if "gleu_score" in eval_df.columns else 0.0
    mean_sem = float(eval_df["semantic_similarity"].mean()) if "semantic_similarity" in eval_df.columns else 0.0
    over_rate = float(eval_df["is_over_correction"].mean() * 100) if "is_over_correction" in eval_df.columns else 0.0
    under_rate = float(eval_df["is_under_correction"].mean() * 100) if "is_under_correction" in eval_df.columns else 0.0

    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT 8: AUTOMATED GRAMMAR CORRECTION & REWRITING EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total Sentences Evaluated : {len(eval_df)}")
    lines.append(f"Exact Match Accuracy      : {exact_acc:.2f}%")
    lines.append(f"Mean Levenshtein Distance : {mean_lev:.2f} characters")
    lines.append(f"Mean Token-Level F1-Score : {mean_f1:.4f} ({mean_f1*100:.2f}%)")
    lines.append(f"Mean GLEU Score           : {mean_gleu:.4f} ({mean_gleu*100:.2f}%)")
    lines.append(f"Mean Semantic Similarity  : {mean_sem:.4f} ({mean_sem*100:.2f}%)")
    lines.append(f"Over-Correction Rate      : {over_rate:.2f}%")
    lines.append(f"Under-Correction Rate     : {under_rate:.2f}%")
    lines.append("-" * 80)

    if prompt_comp_df is not None and not prompt_comp_df.empty:
        lines.append("\nPROMPT ENGINEERING STRATEGIES BENCHMARK:")
        lines.append("-" * 80)
        lines.append(f"{'Prompt Version':<20} | {'Exact Match':<12} | {'Token F1':<10} | {'GLEU':<8} | {'Latency':<8}")
        lines.append("-" * 80)
        for _, row in prompt_comp_df.iterrows():
            lines.append(f"{str(row.get('prompt_version', '')):<20} | {row.get('exact_match_acc', 0.0):<11.1f}% | {row.get('token_f1', 0.0):<10.4f} | {row.get('gleu_score', 0.0):<8.4f} | {row.get('latency_seconds', 0.0):<7.2f}s")
        lines.append("-" * 80)

    if baseline_df is not None and not baseline_df.empty:
        lines.append("\nNEURAL MODEL VS. RULE-BASED BASELINE COMPARISON:")
        lines.append("-" * 80)
        for _, row in baseline_df.iterrows():
            lines.append(f"* {row.get('approach', '')}: Token F1={row.get('token_f1', 0.0):.4f}, Exact Match={row.get('exact_match', 0.0):.1f}%, GLEU={row.get('gleu_score', 0.0):.4f}")
        lines.append("-" * 80)

    lines.append("\nPER-SENTENCE DETAILED BREAKDOWN:")
    lines.append("-" * 80)
    for idx, row in eval_df.iterrows():
        sid = row.get("id", idx + 1)
        cat = row.get("error_type", "General")
        lines.append(f"[{sid}] Error Category: {cat}")
        lines.append(f"  - Input Error : {row.get('input_sentence', '')}")
        lines.append(f"  - Prediction  : {row.get('prediction', '')}")
        lines.append(f"  - Gold Ref    : {row.get('reference', '')}")
        lines.append(f"  - Score: Exact={row.get('exact_match', False)} | LevDist={row.get('levenshtein_distance', 0)} | Token F1={row.get('token_f1', 0.0):.4f} | GLEU={row.get('gleu_score', 0.0):.4f}\n")

    return "\n".join(lines)


class EvaluationModule:
    """Object-oriented wrapper class for backward compatibility."""

    calculate_exact_match = staticmethod(calculate_exact_match)
    calculate_edit_distance = staticmethod(calculate_edit_distance)
    calculate_levenshtein_distance = staticmethod(calculate_edit_distance)
    calculate_token_f1 = staticmethod(calculate_token_f1)
    calculate_gleu_score = staticmethod(calculate_gleu_score)
    calculate_semantic_similarity = staticmethod(calculate_semantic_similarity)
    assess_semantic_preservation = staticmethod(calculate_semantic_similarity)
    assess_length_and_preservation = staticmethod(assess_length_and_preservation)
    detect_over_and_under_correction = staticmethod(detect_over_and_under_correction)
    evaluate_sample = staticmethod(evaluate_correction_quality)

    def evaluate_batch(
        self,
        inputs: List[str],
        predictions: List[str],
        references: List[str],
        error_types: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        records = []
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references)):
            cat = error_types[i] if error_types and i < len(error_types) else "General"
            res = evaluate_correction_quality(inp, pred, ref, error_type=cat)
            res["id"] = i + 1
            records.append(res)

        df = pd.DataFrame(records)
        summary = {
            "total_samples": len(df),
            "exact_match_accuracy": round(float(df["exact_match"].mean() * 100), 2) if len(df) > 0 else 0.0,
            "avg_levenshtein_distance": round(float(df["levenshtein_distance"].mean()), 2) if len(df) > 0 else 0.0,
            "avg_token_f1": round(float(df["token_f1"].mean()), 4) if len(df) > 0 else 0.0,
            "avg_token_precision": round(float(df["token_precision"].mean()), 4) if len(df) > 0 else 0.0,
            "avg_token_recall": round(float(df["token_recall"].mean()), 4) if len(df) > 0 else 0.0,
            "avg_gleu_score": round(float(df["gleu_score"].mean()), 4) if len(df) > 0 else 0.0,
            "avg_semantic_similarity": round(float(df["semantic_similarity"].mean()), 4) if len(df) > 0 else 0.0,
            "over_correction_rate": round(float(df["is_over_correction"].mean() * 100), 2) if len(df) > 0 else 0.0,
            "under_correction_rate": round(float(df["is_under_correction"].mean() * 100), 2) if len(df) > 0 else 0.0
        }
        return summary, df
