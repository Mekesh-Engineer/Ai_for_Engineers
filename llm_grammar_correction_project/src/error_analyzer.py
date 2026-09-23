# -*- coding: utf-8 -*-
"""
error_analyzer.py
-----------------
ErrorAnalyzerModule for Experiment 8: Automated Grammar Error Correction & Text Rewriting.
Analyzes model performance across error categories, classifies error taxonomy,
and generates diagnostic breakdown reports.
"""

from typing import Dict, List, Optional, Any
import pandas as pd


def analyze_by_error_type(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Computes metric aggregations grouped by error category."""
    if evaluation_df is None or evaluation_df.empty:
        return pd.DataFrame()

    col = "error_type" if "error_type" in evaluation_df.columns else ("category" if "category" in evaluation_df.columns else None)
    if not col:
        return pd.DataFrame()

    agg_dict = {
        "sample_count": ("id", "count"),
        "exact_match_pct": ("exact_match", lambda s: round(s.mean() * 100, 1)) if "exact_match" in evaluation_df.columns else ("id", "count"),
        "avg_lev_distance": ("levenshtein_distance", lambda s: round(s.mean(), 2)) if "levenshtein_distance" in evaluation_df.columns else ("id", "count"),
        "avg_token_f1": ("token_f1", lambda s: round(s.mean(), 4)) if "token_f1" in evaluation_df.columns else ("id", "count"),
        "avg_gleu_score": ("gleu_score", lambda s: round(s.mean(), 4)) if "gleu_score" in evaluation_df.columns else ("id", "count"),
        "over_correction_pct": ("is_over_correction", lambda s: round(s.mean() * 100, 1)) if "is_over_correction" in evaluation_df.columns else ("id", "count"),
        "under_correction_pct": ("is_under_correction", lambda s: round(s.mean() * 100, 1)) if "is_under_correction" in evaluation_df.columns else ("id", "count")
    }

    # Clean agg_dict based on existing columns
    valid_aggs = {k: v for k, v in agg_dict.items() if v[0] in evaluation_df.columns}
    grouped = evaluation_df.groupby(col).agg(**valid_aggs).reset_index()
    if col != "error_type":
        grouped = grouped.rename(columns={col: "error_type"})

    return grouped.sort_values(by="sample_count", ascending=False)


def generate_diagnostic_summary(evaluation_df: pd.DataFrame) -> str:
    """Produces a formatted textual diagnostic report."""
    summary_by_type = analyze_by_error_type(evaluation_df)
    under_count = int(evaluation_df["is_under_correction"].sum()) if "is_under_correction" in evaluation_df.columns else 0
    over_count = int(evaluation_df["is_over_correction"].sum()) if "is_over_correction" in evaluation_df.columns else 0
    non_exact = int((~evaluation_df["exact_match"]).sum()) if "exact_match" in evaluation_df.columns else 0

    lines = [
        "=" * 75,
        "EXPERIMENT 8: GRAMMAR ERROR CORRECTION & REWRITING DIAGNOSTIC REPORT",
        "=" * 75,
        "",
        "1. PERFORMANCE BREAKDOWN BY ERROR CATEGORY:",
        "-" * 75
    ]

    if not summary_by_type.empty:
        for _, row in summary_by_type.iterrows():
            f1_val = row.get('avg_token_f1', 0.0)
            gleu_val = row.get('avg_gleu_score', 0.0)
            lines.append(
                f"• {row['error_type']:<35} | Count: {row['sample_count']:<2} | "
                f"Match: {row.get('exact_match_pct', 0.0)}% | Lev Dist: {row.get('avg_lev_distance', 0.0)} | Token F1: {f1_val:.4f} | GLEU: {gleu_val:.4f}"
            )
    else:
        lines.append("No error categories provided in evaluation DataFrame.")

    lines.extend([
        "",
        "2. SUMMARY OF FAILURE & EDIT PATTERNS:",
        "-" * 75,
        f"• Under-Correction Count : {under_count}",
        f"• Over-Correction Count  : {over_count}",
        f"• Non-Exact Matches      : {non_exact}",
        "",
        "3. RECOMMENDATIONS FOR PROMPT OPTIMIZATION:",
        "-" * 75,
        "• Use 'minimal' prompt template when preserving domain identifiers and exact terminology is critical.",
        "• Use 'standard' prompt template for general technical report proofreading to balance accuracy and fluency.",
        "• Use 'rewrite' / 'academic' prompt template when sentence restructuring and formal scientific style are desired.",
        "=" * 75
    ])

    return "\n".join(lines)


class ErrorAnalyzerModule:
    """Object-oriented wrapper for error analysis."""

    def analyze_by_error_type(self, evaluation_df: pd.DataFrame) -> pd.DataFrame:
        return analyze_by_error_type(evaluation_df)

    def extract_failure_cases(self, evaluation_df: pd.DataFrame, max_cases: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        under_cases = evaluation_df[evaluation_df["is_under_correction"]].to_dict(orient="records")[:max_cases] if "is_under_correction" in evaluation_df.columns else []
        over_cases = evaluation_df[evaluation_df["is_over_correction"]].to_dict(orient="records")[:max_cases] if "is_over_correction" in evaluation_df.columns else []
        high_dist = evaluation_df[~evaluation_df["exact_match"]].sort_values(by="levenshtein_distance", ascending=False).to_dict(orient="records")[:max_cases] if ("exact_match" in evaluation_df.columns and "levenshtein_distance" in evaluation_df.columns) else []

        return {
            "under_corrections": under_cases,
            "over_corrections": over_cases,
            "high_divergence_cases": high_dist
        }

    def generate_diagnostic_summary(self, evaluation_df: pd.DataFrame) -> str:
        return generate_diagnostic_summary(evaluation_df)
