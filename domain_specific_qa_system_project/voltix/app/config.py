import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if available
load_dotenv(BASE_DIR / ".env")

# Load YAML configuration
CONFIG_YAML_PATH = BASE_DIR / "config.yaml"
yaml_config = {}
if CONFIG_YAML_PATH.exists():
    with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f) or {}


class Config:
    """Base application configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "voltix-eee-prod-secret-key-2026")
    
    # Storage & SQLite
    DB_DIR = BASE_DIR / "data" / "db"
    DB_PATH = DB_DIR / yaml_config.get("storage", {}).get("db_filename", "voltix.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload & Directories
    UPLOAD_FOLDER = BASE_DIR / "data" / "documents"
    VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
    FAISS_INDEX_PATH = VECTOR_STORE_DIR / yaml_config.get("storage", {}).get("faiss_index_filename", "faiss.index")
    FAISS_METADATA_PATH = VECTOR_STORE_DIR / yaml_config.get("storage", {}).get("faiss_metadata_filename", "metadata.pkl")
    MAX_CONTENT_LENGTH = yaml_config.get("storage", {}).get("upload_max_size_mb", 32) * 1024 * 1024

    # Ollama & LLM
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", yaml_config.get("ollama", {}).get("base_url", "http://localhost:11434"))
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", yaml_config.get("ollama", {}).get("default_model", "qwen2.5:7b"))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # RAG Settings
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", yaml_config.get("rag", {}).get("embedding_model", "BAAI/bge-small-en-v1.5"))
    RAG_CHUNK_SIZE = yaml_config.get("rag", {}).get("chunk_size", 512)
    RAG_CHUNK_OVERLAP = yaml_config.get("rag", {}).get("chunk_overlap", 64)
    RAG_TOP_K = int(os.getenv("TOP_K", yaml_config.get("rag", {}).get("top_k", 5)))
    RAG_SIMILARITY_THRESHOLD = yaml_config.get("rag", {}).get("similarity_threshold", 0.65)


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
