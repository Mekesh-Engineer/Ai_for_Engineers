from typing import List, Dict, Any

class SemanticChunker:
    """Overlapping character/token chunker for document pages."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        chunks = []
        chunk_idx = 0

        for page in pages:
            text = page["text"]
            page_num = page.get("page_number", 1)
            source = page.get("source", "Unknown Document")

            if len(text) <= self.chunk_size:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "page_number": page_num,
                    "source": source,
                    "content": text
                })
                chunk_idx += 1
            else:
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        chunks.append({
                            "doc_id": doc_id,
                            "chunk_index": chunk_idx,
                            "page_number": page_num,
                            "source": source,
                            "content": chunk_text
                        })
                        chunk_idx += 1
                    start += (self.chunk_size - self.chunk_overlap)

        return chunks
