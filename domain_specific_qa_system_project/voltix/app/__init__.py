import os
from flask import Flask
from app.config import Config, DevelopmentConfig, ProductionConfig
from app.extensions import db, cors
from app.database import init_db
from app.routes import main_bp, chat_bp, docs_bp, conv_bp, models_bp, tools_bp, projects_bp, agent_bp
from app.utils.logger import get_logger

logger = get_logger("voltix.app")

def create_app(config_class=None):
    """Application Factory initializing Flask app, DB extensions, and Blueprints."""
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static"
    )

    if config_class is None:
        env = os.getenv("FLASK_ENV", "development")
        config_class = ProductionConfig if env == "production" else DevelopmentConfig

    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize Database & Directories
    init_db(app)

    # Register Blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(docs_bp)
    app.register_blueprint(conv_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(tools_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(agent_bp)

    logger.info("VOLTIX Flask application initialized successfully.")
    return app
