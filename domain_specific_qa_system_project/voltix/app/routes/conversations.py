from flask import Blueprint, jsonify, request, Response, abort
from app.extensions import db
from app.models.conversation import Conversation, Message

conv_bp = Blueprint("conversations", __name__, url_prefix="/api/conversations")

@conv_bp.route("/", methods=["GET"])
def get_conversations():
    """List chat sessions, optionally filtered by project_id."""
    project_id = request.args.get("project_id")
    query = Conversation.query
    if project_id:
        query = query.filter_by(project_id=project_id)
    conversations = query.order_by(Conversation.updated_at.desc()).all()
    return jsonify([c.to_dict() for c in conversations])

@conv_bp.route("/search", methods=["GET"])
def search_conversations():
    """Search conversations by title or message content."""
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])

    matched_conv_ids = set()
    for c in Conversation.query.filter(Conversation.title.ilike(f"%{q}%")).all():
        matched_conv_ids.add(c.id)

    for m in Message.query.filter(Message.content.ilike(f"%{q}%")).all():
        matched_conv_ids.add(m.conversation_id)

    results = Conversation.query.filter(Conversation.id.in_(matched_conv_ids)).order_by(Conversation.updated_at.desc()).all()
    return jsonify([c.to_dict() for c in results])

@conv_bp.route("/", methods=["POST"])
def create_conversation():
    """Create a new conversation session."""
    data = request.get_json() or {}
    title = data.get("title", "New Conversation")
    model_used = data.get("model", "qwen2.5:7b")
    project_id = data.get("project_id")
    academic_mode = data.get("academic_mode", "Learn")

    conv = Conversation(title=title, model_used=model_used, project_id=project_id, academic_mode=academic_mode)
    db.session.add(conv)
    db.session.commit()
    return jsonify(conv.to_dict()), 201

@conv_bp.route("/<string:conv_id>", methods=["GET"])
def get_conversation_detail(conv_id):
    """Get detail and message history of a specific conversation."""
    conv = db.session.get(Conversation, conv_id)
    if not conv:
        abort(404, description="Conversation not found")

    messages = Message.query.filter_by(conversation_id=conv_id).order_by(Message.created_at.asc()).all()

    res = conv.to_dict()
    res["messages"] = [m.to_dict() for m in messages]
    return jsonify(res)

@conv_bp.route("/<string:conv_id>", methods=["PATCH", "PUT"])
def rename_conversation(conv_id):
    """Rename conversation title or change academic mode."""
    conv = db.session.get(Conversation, conv_id)
    if not conv:
        abort(404, description="Conversation not found")

    data = request.get_json() or {}

    if "title" in data and data["title"].strip():
        conv.title = data["title"].strip()
    if "academic_mode" in data:
        conv.academic_mode = data["academic_mode"]

    db.session.commit()
    return jsonify(conv.to_dict())

@conv_bp.route("/<string:conv_id>/export", methods=["GET"])
def export_conversation(conv_id):
    """Export conversation in Markdown or JSON format."""
    conv = db.session.get(Conversation, conv_id)
    if not conv:
        abort(404, description="Conversation not found")

    export_fmt = request.args.get("format", "markdown").lower()

    if export_fmt == "json":
        return jsonify(conv.export_json())
    else:
        md_text = conv.export_markdown()
        return Response(
            md_text,
            mimetype="text/markdown",
            headers={"Content-Disposition": f"attachment;filename=voltix_chat_{conv.id[:8]}.md"}
        )

@conv_bp.route("/<string:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    """Delete a conversation session."""
    conv = db.session.get(Conversation, conv_id)
    if not conv:
        abort(404, description="Conversation not found")

    db.session.delete(conv)
    db.session.commit()
    return jsonify({"message": "Conversation deleted successfully"})
