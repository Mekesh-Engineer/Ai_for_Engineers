import uuid
import json
from datetime import datetime, timezone
from app.extensions import db

class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = db.Column(db.String(36), db.ForeignKey("projects.id"), nullable=True)
    title = db.Column(db.String(255), nullable=False, default="New Conversation")
    model_used = db.Column(db.String(100), nullable=True)
    academic_mode = db.Column(db.String(50), default="Learn")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    messages = db.relationship("Message", backref="conversation", lazy=True, cascade="all, delete-orphan", order_by="Message.created_at")

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "model_used": self.model_used,
            "academic_mode": self.academic_mode,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages) if self.messages else 0,
        }

    def export_markdown(self) -> str:
        lines = [f"# {self.title}", f"**Model:** {self.model_used or 'Default'} | **Mode:** {self.academic_mode} | **Date:** {self.created_at}\n", "---"]
        for msg in self.messages:
            role = "User" if msg.sender == "user" else "VOLTIX Assistant"
            lines.append(f"### 💬 {role}")
            lines.append(f"{msg.content}\n")
        return "\n".join(lines)

    def export_json(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "model_used": self.model_used,
            "academic_mode": self.academic_mode,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "messages": [m.to_dict() for m in self.messages]
        }


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = db.Column(db.String(36), db.ForeignKey("conversations.id"), nullable=False)
    sender = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    metadata_json = db.Column(db.Text, nullable=True)  # JSON for citations, tokens, tool execution info
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        metadata = {}
        if self.metadata_json:
            try:
                metadata = json.loads(self.metadata_json)
            except Exception:
                metadata = {}

        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "sender": self.sender,
            "content": self.content,
            "metadata": metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
