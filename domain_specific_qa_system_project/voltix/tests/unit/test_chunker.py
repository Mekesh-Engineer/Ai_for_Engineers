import pytest
from app.rag.chunker import SemanticChunker

def test_semantic_chunker():
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        {"page_number": 1, "source": "test.pdf", "text": "This is a short page test."},
        {"page_number": 2, "source": "test.pdf", "text": "A" * 250}
    ]
    chunks = chunker.chunk_pages(pages, doc_id="doc-123")
    assert len(chunks) > 1
    assert chunks[0]["doc_id"] == "doc-123"
    assert chunks[0]["page_number"] == 1
