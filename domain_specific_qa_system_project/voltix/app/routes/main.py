from flask import Blueprint, render_template, current_app

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    """Renders the main ChatGPT-style web UI."""
    return render_template("chat.html", app_name=current_app.config.get("SECRET_KEY", "VOLTIX"))
