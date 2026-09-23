import os
import json
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from app.extensions import db
from app.models.document import Document, DocumentChunk
from app.rag import DocumentLoader, SemanticChunker, EmbeddingGenerator, FAISSVectorStore
from app.utils.validators import allowed_file, sanitize_filename
from app.utils.logger import get_logger

docs_bp = Blueprint("documents", __name__, url_prefix="/api/documents")
logger = get_logger("voltix.docs_route")

@docs_bp.route("/", methods=["GET"])
def list_documents():
    """List indexed RAG documents, optionally filtered by project_id."""
    project_id = request.args.get("project_id")
    query = Document.query
    if project_id:
        query = query.filter_by(project_id=project_id)
    documents = query.order_by(Document.uploaded_at.desc()).all()
    return jsonify([doc.to_dict() for doc in documents])

@docs_bp.route("/upload", methods=["POST"])
def upload_document():
    """Upload PDF/DOCX/TXT/MD/CSV file, parse, chunk, compute embeddings, and index into FAISS."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported. Allowed: PDF, DOCX, TXT, MD, CSV"}), 400

    project_id = request.form.get("project_id")
    filename = sanitize_filename(file.filename)
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    file_path = upload_dir / filename
    file.save(file_path)

    try:
        # 1. Load document pages
        pages = DocumentLoader.load_document(str(file_path))
        page_count = len(pages)

        # 2. Save Document model entity
        doc_entity = Document(
            filename=filename,
            file_path=str(file_path),
            page_count=page_count,
            project_id=project_id
        )
        db.session.add(doc_entity)
        db.session.commit()

        # 3. Chunk text
        chunker = SemanticChunker(
            chunk_size=current_app.config["RAG_CHUNK_SIZE"],
            chunk_overlap=current_app.config["RAG_CHUNK_OVERLAP"]
        )
        chunks_data = chunker.chunk_pages(pages, doc_id=doc_entity.id)
        for c in chunks_data:
            c["project_id"] = project_id

        # 4. Generate embeddings
        embedder = EmbeddingGenerator(model_name=current_app.config["EMBEDDING_MODEL_NAME"])
        texts = [c["content"] for c in chunks_data]
        embeddings = embedder.encode(texts)

        # 5. Add to FAISS Vector Store
        vector_store = FAISSVectorStore(
            index_path=current_app.config["FAISS_INDEX_PATH"],
            metadata_path=current_app.config["FAISS_METADATA_PATH"],
            dimension=embeddings.shape[1]
        )
        vector_store.add_vectors(embeddings, chunks_data)

        # 6. Save Chunk DB Entities
        start_faiss_id = len(vector_store.metadata) - len(chunks_data)
        for idx, c in enumerate(chunks_data):
            chunk_entity = DocumentChunk(
                document_id=doc_entity.id,
                project_id=project_id,
                faiss_id=start_faiss_id + idx,
                chunk_index=c["chunk_index"],
                page_number=c["page_number"],
                content=c["content"],
                metadata_json=json.dumps({"source": filename, "project_id": project_id})
            )
            db.session.add(chunk_entity)

        doc_entity.total_chunks = len(chunks_data)
        db.session.commit()

        return jsonify({
            "message": "Document successfully uploaded and indexed",
            "document": doc_entity.to_dict()
        }), 201

    except Exception as e:
        logger.error(f"Failed to process uploaded document {filename}: {e}")
        return jsonify({"error": f"Document processing failed: {str(e)}"}), 500

@docs_bp.route("/<string:doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Delete document from database, remove file from disk, and remove vectors from FAISS index."""
    doc = Document.query.get_or_404(doc_id)
    file_path = Path(doc.file_path)

    # 1. Remove from FAISS index
    try:
        embedder = EmbeddingGenerator(model_name=current_app.config["EMBEDDING_MODEL_NAME"])
        vector_store = FAISSVectorStore(
            index_path=current_app.config["FAISS_INDEX_PATH"],
            metadata_path=current_app.config["FAISS_METADATA_PATH"]
        )
        vector_store.remove_document(doc_id, embedder=embedder)
    except Exception as e:
        logger.warning(f"Error removing vectors from FAISS for doc {doc_id}: {e}")

    # 2. Delete file from storage if exists
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            logger.warning(f"Error deleting document file {file_path}: {e}")

    # 3. Delete from SQLite DB
    db.session.delete(doc)
    db.session.commit()
    logger.info(f"Deleted document {doc.filename} (ID: {doc_id})")

    return jsonify({"message": f"Document '{doc.filename}' deleted and unindexed successfully"})
