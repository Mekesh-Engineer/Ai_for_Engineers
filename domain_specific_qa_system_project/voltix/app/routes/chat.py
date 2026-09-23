import json
from flask import Blueprint, request, Response, stream_with_context, current_app
from app.extensions import db
from app.models.conversation import Conversation, Message
from app.services.orchestrator import Orchestrator
from app.utils.logger import get_logger

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")
logger = get_logger("voltix.chat_route")

@chat_bp.route("/stream", methods=["POST"])
def stream_chat():
    """SSE Streaming Chat API Endpoint supporting ReAct reasoning, tools, academic modes, and project isolation."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    conversation_id = data.get("conversation_id")
    model_name = data.get("model")
    academic_mode = data.get("academic_mode", "Learn")
    project_id = data.get("project_id")
    enable_rag = data.get("enable_rag", True)

    if not prompt:
        return json.dumps({"error": "Prompt cannot be empty"}), 400

    # Ensure conversation exists
    with current_app.app_context():
        conv = None
        if conversation_id:
            conv = db.session.get(Conversation, conversation_id)

        if not conv:
            conv = Conversation(
                title=prompt[:40] + ("..." if len(prompt) > 40 else ""),
                model_used=model_name or current_app.config.get("DEFAULT_MODEL", "qwen2.5:7b"),
                academic_mode=academic_mode,
                project_id=project_id
            )
            db.session.add(conv)
            db.session.commit()
            conversation_id = conv.id
        else:
            conv.academic_mode = academic_mode
            if project_id and not conv.project_id:
                conv.project_id = project_id
            db.session.commit()

        # Save user message
        user_msg = Message(
            conversation_id=conversation_id,
            sender="user",
            content=prompt
        )
        db.session.add(user_msg)
        db.session.commit()

    orchestrator = Orchestrator(current_app.config)

    def generate_events():
        full_assistant_response = ""
        citations = []
        thought_log = []

        try:
            for event in orchestrator.process_and_stream(
                prompt=prompt,
                conversation_id=conversation_id,
                model_name=model_name,
                academic_mode=academic_mode,
                project_id=project_id,
                enable_rag=enable_rag
            ):
                ev_type = event.get("type")

                if ev_type == "meta":
                    citations = event.get("citations", [])
                    meta_payload = {
                        "type": "meta",
                        "conversation_id": conversation_id,
                        "agent": event.get("agent"),
                        "intent": event.get("intent"),
                        "model": event.get("model"),
                        "academic_mode": event.get("academic_mode", "Learn"),
                        "citations": citations
                    }
                    yield f"data: {json.dumps(meta_payload)}\n\n"

                elif ev_type == "thought":
                    thought_log.append({"type": "thought", "content": event.get("content")})
                    yield f"data: {json.dumps(event)}\n\n"

                elif ev_type == "tool_call":
                    thought_log.append({"type": "tool_call", "name": event.get("name"), "args": event.get("args")})
                    yield f"data: {json.dumps(event)}\n\n"

                elif ev_type == "tool_result":
                    thought_log.append({"type": "tool_result", "name": event.get("name"), "output": event.get("output")})
                    yield f"data: {json.dumps(event)}\n\n"

                elif ev_type == "token":
                    full_assistant_response += event["content"]
                    token_payload = {"type": "token", "content": event["content"]}
                    yield f"data: {json.dumps(token_payload)}\n\n"

                elif ev_type == "done":
                    # Save completed assistant message to SQLite
                    try:
                        with current_app.app_context():
                            asst_msg = Message(
                                conversation_id=conversation_id,
                                sender="assistant",
                                content=full_assistant_response,
                                metadata_json=json.dumps({
                                    "citations": citations,
                                    "thought_log": thought_log
                                })
                            )
                            db.session.add(asst_msg)
                            db.session.commit()
                    except Exception as db_err:
                        logger.error(f"Error saving assistant message: {db_err}")

                    done_payload = {
                        "type": "done",
                        "conversation_id": conversation_id,
                        "citations": citations
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            logger.error(f"Streaming exception: {e}")
            err_payload = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return Response(
        stream_with_context(generate_events()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )
