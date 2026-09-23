from flask import Blueprint, request, jsonify, abort
from app.extensions import db
from app.models.project import Project
from app.models.conversation import Conversation
from app.models.document import Document
from app.utils.logger import get_logger

projects_bp = Blueprint("projects", __name__, url_prefix="/api/projects")
logger = get_logger("voltix.projects_route")

@projects_bp.route("/", methods=["GET"])
def list_projects():
    """List all user engineering projects."""
    projects = Project.query.order_by(Project.updated_at.desc()).all()
    return jsonify([p.to_dict() for p in projects])

@projects_bp.route("/", methods=["POST"])
def create_project():
    """Create a new engineering project workspace."""
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Project name cannot be empty"}), 400

    project = Project(
        name=name,
        description=data.get("description", ""),
        notes=data.get("notes", "")
    )
    db.session.add(project)
    db.session.commit()
    logger.info(f"Created project: {project.name} (ID: {project.id})")
    return jsonify(project.to_dict()), 201

@projects_bp.route("/<string:project_id>", methods=["GET"])
def get_project_detail(project_id):
    """Retrieve full project metadata with conversations and documents."""
    project = db.session.get(Project, project_id)
    if not project:
        abort(404, description="Project not found")

    convs = Conversation.query.filter_by(project_id=project_id).order_by(Conversation.updated_at.desc()).all()
    docs = Document.query.filter_by(project_id=project_id).order_by(Document.uploaded_at.desc()).all()

    res = project.to_dict()
    res["conversations"] = [c.to_dict() for c in convs]
    res["documents"] = [d.to_dict() for d in docs]
    return jsonify(res)

@projects_bp.route("/<string:project_id>", methods=["PUT", "PATCH"])
def update_project(project_id):
    """Update project name, description, or notes."""
    project = db.session.get(Project, project_id)
    if not project:
        abort(404, description="Project not found")

    data = request.get_json() or {}

    if "name" in data and data["name"].strip():
        project.name = data["name"].strip()
    if "description" in data:
        project.description = data["description"]
    if "notes" in data:
        project.notes = data["notes"]

    db.session.commit()
    return jsonify(project.to_dict())

@projects_bp.route("/<string:project_id>", methods=["DELETE"])
def delete_project(project_id):
    """Delete project and associated resources."""
    project = db.session.get(Project, project_id)
    if not project:
        abort(404, description="Project not found")

    db.session.delete(project)
    db.session.commit()
    logger.info(f"Deleted project ID {project_id}")
    return jsonify({"message": "Project deleted successfully"})
