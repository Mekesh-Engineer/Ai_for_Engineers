import numpy as np
from typing import List, Union
from app.utils.logger import get_logger

logger = get_logger("voltix.embedder")

class EmbeddingGenerator:
    """Wraps SentenceTransformers / BGE for 384d / 768d text embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                # Fallback to lightweight model
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        self._load_model()
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)
