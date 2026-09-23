import uuid
from datetime import datetime, timezone
from app.extensions import db

class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = db.Column(db.String(36), db.ForeignKey("projects.id"), nullable=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_hash = db.Column(db.String(64), nullable=True)
    total_chunks = db.Column(db.Integer, default=0)
    page_count = db.Column(db.Integer, default=0)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    chunks = db.relationship("DocumentChunk", backref="document", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "total_chunks": self.total_chunks,
            "page_count": self.page_count,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
        }


class DocumentChunk(db.Model):
    __tablename__ = "document_chunks"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = db.Column(db.String(36), db.ForeignKey("documents.id"), nullable=False)
    project_id = db.Column(db.String(36), db.ForeignKey("projects.id"), nullable=True)
    faiss_id = db.Column(db.Integer, nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    page_number = db.Column(db.Integer, default=1)
    content = db.Column(db.Text, nullable=False)
    metadata_json = db.Column(db.Text, nullable=True)

    def to_dict(self):
        import json
        metadata = {}
        if self.metadata_json:
            try:
                metadata = json.loads(self.metadata_json)
            except Exception:
                metadata = {}

        return {
            "id": self.id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "faiss_id": self.faiss_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "content": self.content,
            "metadata": metadata,
        }
