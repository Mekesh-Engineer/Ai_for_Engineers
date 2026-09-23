import uuid
from datetime import datetime, timezone
from app.extensions import db

class ToolLog(db.Model):
    __tablename__ = "tool_logs"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = db.Column(db.String(36), db.ForeignKey("messages.id"), nullable=True)
    tool_name = db.Column(db.String(100), nullable=False)
    query = db.Column(db.Text, nullable=False)
    inputs_json = db.Column(db.Text, nullable=True)
    result_json = db.Column(db.Text, nullable=True)
    execution_time_ms = db.Column(db.Float, default=0.0)
    executed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "query": self.query,
            "inputs": json.loads(self.inputs_json) if self.inputs_json else {},
            "result": json.loads(self.result_json) if self.result_json else {},
            "execution_time_ms": self.execution_time_ms,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }
