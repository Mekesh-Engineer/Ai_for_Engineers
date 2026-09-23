class GenerationMetrics:
    """Computes basic text quality metrics such as ROUGE-L or Exact Match."""

    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

    @staticmethod
    def token_overlap(prediction: str, reference: str) -> float:
        pred_set = set(prediction.lower().split())
        ref_set = set(reference.lower().split())
        if not ref_set:
            return 0.0
        intersection = pred_set.intersection(ref_set)
        return len(intersection) / len(ref_set)
