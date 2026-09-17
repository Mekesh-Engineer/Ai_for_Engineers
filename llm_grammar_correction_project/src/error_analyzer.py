"""
ErrorAnalyzerModule for Experiment 8: Automated Grammar Correction & Text Rewriting
Analyzes model performance across error categories and diagnoses failure patterns.
"""

from typing import Dict, List, Optional
import pandas as pd

class ErrorAnalyzerModule:
    def __init__(self):
        pass

    def analyze_by_error_type(self, evaluation_df: pd.DataFrame) -> pd.DataFrame:
        """Computes metric aggregations grouped by error category."""
        if "error_type" not in evaluation_df.columns:
            return pd.DataFrame()

        grouped = evaluation_df.groupby("error_type").agg(
            sample_count=("id", "count"),
            exact_match_pct=("exact_match", lambda s: round(s.mean() * 100, 1)),
            avg_lev_distance=("levenshtein_distance", lambda s: round(s.mean(), 2)),
            avg_token_f1=("token_f1", lambda s: round(s.mean(), 4)),
            over_correction_pct=("is_over_correction", lambda s: round(s.mean() * 100, 1)),
            under_correction_pct=("is_under_correction", lambda s: round(s.mean() * 100, 1))
        ).reset_index()

        return grouped.sort_values(by="sample_count", ascending=False)

    def extract_failure_cases(self, evaluation_df: pd.DataFrame, max_cases: int = 5) -> Dict[str, List[Dict]]:
        """Extracts representative under-correction, over-correction, and high-edit failure cases."""
        under_cases = evaluation_df[evaluation_df["is_under_correction"]].to_dict(orient="records")[:max_cases]
        over_cases = evaluation_df[evaluation_df["is_over_correction"]].to_dict(orient="records")[:max_cases]
        
        # High edit distance cases where exact match failed
        high_dist = evaluation_df[~evaluation_df["exact_match"]].sort_values(by="levenshtein_distance", ascending=False).to_dict(orient="records")[:max_cases]

        return {
            "under_corrections": under_cases,
            "over_corrections": over_cases,
            "high_divergence_cases": high_dist
        }

    def generate_diagnostic_summary(self, evaluation_df: pd.DataFrame) -> str:
        """Produces a formatted textual diagnostic report."""
        summary_by_type = self.analyze_by_error_type(evaluation_df)
        failures = self.extract_failure_cases(evaluation_df)

        lines = [
            "=" * 70,
            "GRAMMAR ERROR CORRECTION & REWRITING DIAGNOSTIC REPORT",
            "=" * 70,
            "",
            "1. PERFORMANCE BREAKDOWN BY ERROR CATEGORY:",
            "-" * 70
        ]

        if not summary_by_type.empty:
            for _, row in summary_by_type.iterrows():
                lines.append(
                    f"• {row['error_type']:<35} | Count: {row['sample_count']:<2} | "
                    f"Match: {row['exact_match_pct']}% | Lev Dist: {row['avg_lev_distance']} | F1: {row['avg_token_f1']}"
                )
        else:
            lines.append("No error categories provided in evaluation DataFrame.")

        lines.extend([
            "",
            "2. SUMMARY OF FAILURE PATTERNS:",
            "-" * 70,
            f"• Under-Correction Count : {len(evaluation_df[evaluation_df['is_under_correction']])}",
            f"• Over-Correction Count  : {len(evaluation_df[evaluation_df['is_over_correction']])}",
            f"• Non-Exact Matches      : {len(evaluation_df[~evaluation_df['exact_match']])}",
            "",
            "3. RECOMMENDATIONS FOR PROMPT OPTIMIZATION:",
            "-" * 70,
            "• Use 'minimal' prompt template when preserving domain identifiers and exact terminology is critical.",
            "• Use 'standard' prompt template for general technical report proofreading to balance accuracy and fluency.",
            "• Use 'rewrite' / 'academic' prompt template when sentence restructuring and formal passive voice are desired.",
            "=" * 70
        ])

        return "\n".join(lines)
