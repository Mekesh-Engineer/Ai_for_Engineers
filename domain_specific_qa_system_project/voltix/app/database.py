from app.extensions import db

def init_db(app):
    """Ensure database tables and data directories exist."""
    app.config["DB_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["VECTOR_STORE_DIR"].mkdir(parents=True, exist_ok=True)
    
    with app.app_context():
        db.create_all()
