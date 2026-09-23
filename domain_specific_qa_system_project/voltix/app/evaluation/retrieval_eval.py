from typing import List

class RetrievalMetrics:
    """Computes Hit@K, Precision@K, Recall@K, and MRR (Mean Reciprocal Rank)."""

    @staticmethod
    def hit_at_k(retrieved_docs: List[str], ground_truth_doc: str, k: int = 5) -> float:
        top_k = retrieved_docs[:k]
        return 1.0 if ground_truth_doc in top_k else 0.0

    @staticmethod
    def reciprocal_rank(retrieved_docs: List[str], ground_truth_doc: str) -> float:
        if ground_truth_doc in retrieved_docs:
            rank = retrieved_docs.index(ground_truth_doc) + 1
            return 1.0 / rank
        return 0.0
