"""
EvaluationModule for Experiment 8: Automated Grammar Correction & Text Rewriting
Computes Exact Match, Levenshtein Distance, Token F1, Over/Under-correction, and Semantic Preservation.
"""

import re
from typing import Dict, List, Tuple, Union
import pandas as pd

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes character-level Levenshtein edit distance between two strings."""
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


class EvaluationModule:
    @staticmethod
    def calculate_exact_match(prediction: str, reference: str) -> bool:
        """Calculates boolean Exact Match after whitespace and lowercase normalization."""
        p_clean = re.sub(r'\s+', ' ', prediction.strip().lower())
        r_clean = re.sub(r'\s+', ' ', reference.strip().lower())
        return p_clean == r_clean

    @staticmethod
    def calculate_edit_distance(prediction: str, reference: str) -> int:
        """Calculates Levenshtein edit distance between predicted and reference text."""
        return compute_levenshtein_distance(prediction.strip(), reference.strip())

    calculate_levenshtein_distance = calculate_edit_distance

    @staticmethod
    def calculate_token_f1(prediction: str, reference: str) -> Dict[str, float]:
        """Calculates token-level precision, recall, and F1 score."""
        pred_tokens = re.findall(r'\w+', prediction.lower())
        ref_tokens = re.findall(r'\w+', reference.lower())

        if not pred_tokens and not ref_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not pred_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        pred_counts = {}
        for t in pred_tokens:
            pred_counts[t] = pred_counts.get(t, 0) + 1

        ref_counts = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1

        common_count = sum(min(pred_counts[t], ref_counts.get(t, 0)) for t in pred_counts)

        precision = common_count / len(pred_tokens) if pred_tokens else 0.0
        recall = common_count / len(ref_tokens) if ref_tokens else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    @staticmethod
    def assess_semantic_preservation(input_text: str, prediction: str) -> float:
        """Measures keyword preservation ratio between raw input and predicted correction."""
        in_tokens = set(re.findall(r'\w+', input_text.lower()))
        pred_tokens = set(re.findall(r'\w+', prediction.lower()))

        if not in_tokens:
            return 1.0
        intersection = in_tokens.intersection(pred_tokens)
        return round(len(intersection) / len(in_tokens), 4)

    @staticmethod
    def detect_over_and_under_correction(input_text: str, prediction: str, reference: str) -> Tuple[bool, bool]:
        """
        Determines if prediction shows over-correction (excessive alterations beyond reference)
        or under-correction (remaining identical to erroneous input).
        """
        in_clean = input_text.strip().lower()
        pred_clean = prediction.strip().lower()
        ref_clean = reference.strip().lower()

        is_under_correction = (pred_clean == in_clean and in_clean != ref_clean)
        
        # Over-correction heuristic: edit distance to ref is significantly larger than input to ref
        dist_pred_ref = compute_levenshtein_distance(pred_clean, ref_clean)
        dist_in_ref = compute_levenshtein_distance(in_clean, ref_clean)
        is_over_correction = (dist_pred_ref > dist_in_ref and dist_in_ref > 0)

        return is_over_correction, is_under_correction

    def evaluate_sample(self, input_text: str, prediction: str, reference: str) -> Dict:
        """Evaluates a single prediction against its reference correction and input text."""
        exact = self.calculate_exact_match(prediction, reference)
        lev_dist = self.calculate_edit_distance(prediction, reference)
        token_metrics = self.calculate_token_f1(prediction, reference)
        sem_pres = self.assess_semantic_preservation(input_text, prediction)
        is_over, is_under = self.detect_over_and_under_correction(input_text, prediction, reference)

        return {
            "exact_match": exact,
            "levenshtein_distance": lev_dist,
            "token_precision": token_metrics["precision"],
            "token_recall": token_metrics["recall"],
            "token_f1": token_metrics["f1"],
            "semantic_preservation": sem_pres,
            "is_over_correction": is_over,
            "is_under_correction": is_under
        }

    def evaluate_batch(
        self,
        inputs: List[str],
        predictions: List[str],
        references: List[str],
        error_types: Optional[List[str]] = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """Evaluates a full batch of predictions and aggregates metrics."""
        records = []
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references)):
            res = self.evaluate_sample(inp, pred, ref)
            res["id"] = i + 1
            res["input_sentence"] = inp
            res["prediction"] = pred
            res["reference"] = ref
            if error_types and i < len(error_types):
                res["error_type"] = error_types[i]
            records.append(res)

        df = pd.DataFrame(records)
        
        summary = {
            "total_samples": len(df),
            "exact_match_accuracy": round(float(df["exact_match"].mean() * 100), 2),
            "avg_levenshtein_distance": round(float(df["levenshtein_distance"].mean()), 2),
            "avg_token_f1": round(float(df["token_f1"].mean()), 4),
            "avg_token_precision": round(float(df["token_precision"].mean()), 4),
            "avg_token_recall": round(float(df["token_recall"].mean()), 4),
            "avg_semantic_preservation": round(float(df["semantic_preservation"].mean()), 4),
            "over_correction_rate": round(float(df["is_over_correction"].mean() * 100), 2),
            "under_correction_rate": round(float(df["is_under_correction"].mean() * 100), 2)
        }

        return summary, df
