from app.routes.main import main_bp
from app.routes.chat import chat_bp
from app.routes.documents import docs_bp
from app.routes.conversations import conv_bp
from app.routes.models import models_bp
from app.routes.tools import tools_bp
from app.routes.projects import projects_bp
from app.routes.agent import agent_bp

__all__ = [
    "main_bp",
    "chat_bp",
    "docs_bp",
    "conv_bp",
    "models_bp",
    "tools_bp",
    "projects_bp",
    "agent_bp",
]
