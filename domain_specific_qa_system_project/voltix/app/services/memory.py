from typing import List, Dict, Any
from app.models.conversation import Message

class ConversationMemoryManager:
    """Multi-turn conversation history manager fetching recent context windows."""

    @staticmethod
    def get_recent_history(conversation_id: str, limit: int = 6) -> List[Dict[str, str]]:
        if not conversation_id:
            return []

        recent_msgs = (
            Message.query.filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )

        history = []
        for msg in reversed(recent_msgs):
            history.append({
                "sender": msg.sender,
                "content": msg.content
            })

        return history
