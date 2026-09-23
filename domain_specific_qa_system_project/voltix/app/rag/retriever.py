from typing import List, Dict, Any, Optional
from app.rag.embedder import EmbeddingGenerator
from app.rag.vector_store import FAISSVectorStore
from app.utils.logger import get_logger

logger = get_logger("voltix.retriever")

class RAGRetriever:
    """Top-K semantic retriever with project isolation and score thresholding."""

    def __init__(self, embedder: EmbeddingGenerator, vector_store: FAISSVectorStore, similarity_threshold: float = 0.5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str, top_k: int = 5, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query_vector = self.embedder.encode(query)
        results = self.vector_store.search(query_vector, top_k=top_k, project_id=project_id)

        retrieved = []
        for score, meta in results:
            if score >= self.similarity_threshold:
                item = dict(meta)
                item["similarity_score"] = round(score, 4)
                retrieved.append(item)

        logger.info(f"Retrieved {len(retrieved)} chunks (threshold={self.similarity_threshold}, project={project_id})")
        return retrieved
