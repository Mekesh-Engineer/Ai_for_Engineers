import pytest
from app.rag.citation import CitationFormatter

def test_citation_formatter():
    passages = [
        {"source": "Transformer_Theory.pdf", "page_number": 42, "similarity_score": 0.89, "content": "Transformer turns ratio formula is V1/V2 = N1/N2."}
    ]
    citations = CitationFormatter.format_citations(passages)
    assert len(citations) == 1
    assert citations[0]["document"] == "Transformer_Theory.pdf"
    assert citations[0]["page"] == 42
    assert citations[0]["score"] == 0.89

    context_str = CitationFormatter.build_context_string(passages)
    assert "UNTRUSTED SOURCE CONTENT" in context_str
    assert "Transformer_Theory.pdf" in context_str
