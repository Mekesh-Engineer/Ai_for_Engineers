from datetime import datetime, timezone
from app.extensions import db

class UserSettings(db.Model):
    __tablename__ = "user_settings"

    id = db.Column(db.Integer, primary_key=True, default=1)
    default_model = db.Column(db.String(100), default="qwen2.5:7b")
    temperature = db.Column(db.Float, default=0.2)
    top_p = db.Column(db.Float, default=0.9)
    top_k = db.Column(db.Integer, default=40)
    enable_rag = db.Column(db.Boolean, default=True)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "default_model": self.default_model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_rag": self.enable_rag,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
