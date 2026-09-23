import os
import tempfile
import pytest
from pathlib import Path
from app.tools import EECalculator, SymPySolver, UnitConverter, MatlabScriptGenerator
from app.services.router import QueryRouter
from app.services.prompt_builder import PromptBuilder
from app.rag import DocumentLoader, SemanticChunker, FAISSVectorStore, RAGRetriever, CitationFormatter
from app.models.conversation import Conversation, Message
from app.models.document import Document
from app.extensions import db

def test_acceptance_1_eee_concept():
    """Test 1 — EEE Concept: Explain the working principle of a three-phase induction motor."""
    query = "Explain the working principle of a three-phase induction motor."
    routing = QueryRouter.classify_intent(query)
    assert routing["agent"] == "MachinesAgent"
    prompt = PromptBuilder.build_system_prompt(routing["agent"], "", academic_mode="Learn")
    assert "MachinesAgent" in prompt or "Electrical Machines" in prompt
    assert "KaTeX" in prompt

def test_acceptance_2_numerical_slip():
    """Test 2 — Numerical: Ns = 1500 rpm, N = 1440 rpm => s = 4%."""
    res = EECalculator.induction_motor_slip(ns=1500, n=1440)
    assert res["success"] is True
    assert res["result"]["slip_pu"] == 0.04
    assert res["result"]["slip_percent"] == 4.0

def test_acceptance_3_engineering_calculator_3phase_current():
    """Test 3 — Calculator: 5 kW, 415 V, 3-phase, 0.8 pf => Current calculation."""
    res = EECalculator.three_phase_power(p=5000, vl=415, il=None, pf=0.8)
    assert res["success"] is True
    assert abs(res["result"]["line_current_a"] - 8.696) < 0.01

def test_acceptance_4_and_5_rag_and_document_lifecycle(app):
    """Test 4, 5 & 10 — RAG Document Ingestion, Retrieval, Summary, Citations, and Deletion Lifecycle."""
    with app.app_context():
        # 1. Create a dummy test EEE document
        temp_dir = Path(tempfile.mkdtemp())
        doc_file = temp_dir / "Transformer_Protection_Guide.txt"
        doc_file.write_text("Differential protection (87T) operates on the principle of Kirchhoff's Current Law. Biased differential relays provide slope characteristics to prevent false tripping during CT saturation.", encoding="utf-8")

        pages = DocumentLoader.load_document(str(doc_file))
        assert len(pages) == 1

        doc_record = Document(filename=doc_file.name, file_path=str(doc_file), page_count=1)
        db.session.add(doc_record)
        db.session.commit()

        chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_pages(pages, doc_id=doc_record.id)
        assert len(chunks) >= 1

        index_path = temp_dir / "test.index"
        meta_path = temp_dir / "test_meta.pkl"

        class DummyEmbedder:
            def encode(self, texts):
                import numpy as np
                # Return constant normalized dummy vector of dimension 384
                vecs = np.ones((len(texts), 384), dtype=np.float32)
                for i in range(len(texts)):
                    vecs[i] = vecs[i] / np.linalg.norm(vecs[i])
                return vecs

        dummy_embedder = DummyEmbedder()
        embeddings = dummy_embedder.encode([c["content"] for c in chunks])

        vstore = FAISSVectorStore(str(index_path), str(meta_path), dimension=384)
        vstore.add_vectors(embeddings, chunks)
        assert vstore.index.ntotal >= 1

        retriever = RAGRetriever(dummy_embedder, vstore, similarity_threshold=0.1)
        results = retriever.retrieve("What is differential protection 87T?")
        assert len(results) >= 1
        assert "Differential protection" in results[0]["content"]

        citations = CitationFormatter.format_citations(results)
        assert len(citations) >= 1
        assert citations[0]["document"] == "Transformer_Protection_Guide.txt"

        # Test Deletion Lifecycle
        vstore.remove_document(doc_record.id, embedder=dummy_embedder)
        assert vstore.index.ntotal == 0

def test_acceptance_6_matlab_torque_speed():
    """Test 6 — MATLAB: Generate MATLAB code to plot torque-slip characteristics."""
    script = MatlabScriptGenerator.generate_induction_motor_torque_speed_script(v_phase=230, f=50, poles=4)
    assert "clc; clear; close all;" in script
    assert "plot(Nr, T" in script
    assert "Ns = (120 * f) / P;" in script

def test_acceptance_7_memory_and_persistence(app):
    """Test 7 & 9 — Multi-turn Conversation Memory & SQLite Persistence."""
    with app.app_context():
        conv = Conversation(title="Synchronous Machines Discussion", model_used="qwen2.5:7b")
        db.session.add(conv)
        db.session.commit()

        # Turn 1
        msg1_user = Message(conversation_id=conv.id, sender="user", content="Explain synchronous motor.")
        msg1_asst = Message(conversation_id=conv.id, sender="assistant", content="A synchronous motor runs at synchronous speed Ns = 120f/P.")
        db.session.add_all([msg1_user, msg1_asst])
        db.session.commit()

        # Turn 2
        msg2_user = Message(conversation_id=conv.id, sender="user", content="What is its starting problem?")
        db.session.add(msg2_user)
        db.session.commit()

        # Verify history retrieval
        msgs = Message.query.filter_by(conversation_id=conv.id).order_by(Message.created_at.asc()).all()
        assert len(msgs) == 3
        assert msgs[0].content == "Explain synchronous motor."
        assert msgs[1].content == "A synchronous motor runs at synchronous speed Ns = 120f/P."
        assert msgs[2].content == "What is its starting problem?"

def test_acceptance_11_offline_mathematics():
    """Test 11 — Offline-first deterministic solvers (Ohm, SymPy, Pint) execute without cloud/internet."""
    # SymPy
    res_sympy = SymPySolver.solve_equation("s**2 + 4 = 0", "s")
    assert res_sympy["success"] is True

    # Pint
    res_pint = UnitConverter.convert(2.5, "kV", "V")
    assert res_pint["success"] is True
    assert res_pint["result_value"] == 2500.0

    # EE Calc
    res_pf = EECalculator.power_factor_correction(active_power_kw=100, initial_pf=0.7, target_pf=0.95, voltage_v=415)
    assert res_pf["success"] is True
    assert res_pf["result"]["qc_kvar"] > 0
