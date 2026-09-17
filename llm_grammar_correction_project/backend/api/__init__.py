from backend.api.models import router as models_router
from backend.api.chat import router as chat_router
from backend.api.files import router as files_router
from backend.api.project import router as project_router
from backend.api.settings import router as settings_router

__all__ = [
    "models_router",
    "chat_router",
    "files_router",
    "project_router",
    "settings_router"
]
