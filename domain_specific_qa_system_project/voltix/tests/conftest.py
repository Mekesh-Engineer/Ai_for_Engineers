import os
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.config import Config
from app.extensions import db

class TestConfig(Config):
    TESTING = True
    SECRET_KEY = "test-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = Path(tempfile.mkdtemp())
    VECTOR_STORE_DIR = Path(tempfile.mkdtemp())
    FAISS_INDEX_PATH = VECTOR_STORE_DIR / "test_faiss.index"
    FAISS_METADATA_PATH = VECTOR_STORE_DIR / "test_metadata.pkl"

@pytest.fixture
def app():
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def runner(app):
    return app.test_cli_runner()
