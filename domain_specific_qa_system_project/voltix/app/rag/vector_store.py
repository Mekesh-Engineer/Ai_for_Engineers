import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from app.utils.logger import get_logger

logger = get_logger("voltix.vector_store")

class FAISSVectorStore:
    """Persistent FAISS Index and metadata storage manager with project isolation and document removal."""

    def __init__(self, index_path: str, metadata_path: str, dimension: int = 384):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._load_or_create()

    def _load_or_create(self):
        import faiss
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors) from {self.index_path}")
                return
            except Exception as e:
                logger.warning(f"Error reading FAISS index, creating new: {e}")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info(f"Initialized new FAISS IndexFlatIP (dim={self.dimension})")

    def add_vectors(self, embeddings: np.ndarray, meta_list: List[Dict[str, Any]]):
        import faiss
        if embeddings.shape[1] != self.dimension:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(meta_list)
        self.save()
        logger.info(f"Added {len(embeddings)} vectors to FAISS index. Total: {self.index.ntotal}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        project_id: Optional[str] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Perform similarity search with optional project isolation."""
        import faiss
        if self.index is None or self.index.ntotal == 0:
            return []

        faiss.normalize_L2(query_vector)
        # Fetch more candidates if filtering by project
        search_k = min(max(top_k * 4, 20), self.index.ntotal)
        scores, indices = self.index.search(query_vector, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                meta = self.metadata[idx]
                # Filter by project if specified
                if project_id is not None:
                    if meta.get("project_id") != project_id:
                        continue
                results.append((float(score), meta))
                if len(results) >= top_k:
                    break

        return results

    def remove_document(self, doc_id: str, embedder=None):
        """Removes all chunks associated with doc_id and rebuilds the FAISS index."""
        import faiss
        new_metadata = [m for m in self.metadata if m.get("doc_id") != doc_id]
        if len(new_metadata) == len(self.metadata):
            logger.info(f"No vectors found for doc_id {doc_id}")
            return

        logger.info(f"Removing doc_id {doc_id}. Remaining chunks: {len(new_metadata)}")

        if not new_metadata or embedder is None:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.save()
            return

        # Recompute embeddings for remaining metadata
        texts = [m["content"] for m in new_metadata]
        embeddings = embedder.encode(texts)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = new_metadata
        self.save()
        logger.info(f"FAISS index successfully rebuilt with {self.index.ntotal} vectors.")

    def save(self):
        import faiss
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved FAISS index to {self.index_path}")
