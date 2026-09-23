import json
import uuid
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.services.llm_service import ChatMessage, get_llm_provider
from backend.services.project_analyzer import project_analyzer
from backend.utils.logger import studio_logger

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# In-memory conversation store
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    prompt: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None
    attached_files: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    profile: Optional[str] = None
    mode: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    context_files: Optional[List[str]] = Field(default=[], description="List of relative file paths to inject into context")

class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    mode: str
    model: str
    message_count: int

def build_context_augmented_messages(req: ChatRequest) -> List[ChatMessage]:
    """Inject system prompt, profile instructions, and project file contexts."""
    base_sys_prompt = req.system_prompt
    if not base_sys_prompt and req.profile:
        base_sys_prompt = settings.get_profile_prompt(req.profile)
    if not base_sys_prompt:
        base_sys_prompt = settings.active_system_prompt

    context_blocks = []
    if req.context_files:
        for rel_path in req.context_files:
            try:
                content = project_analyzer.read_file_content(rel_path, max_chars=6000)
                context_blocks.append(f"--- File: {rel_path} ---\n{content}")
            except Exception as e:
                context_blocks.append(f"--- File: {rel_path} (Error reading: {e}) ---")

    augmented_sys_prompt = base_sys_prompt
    if context_blocks:
        context_str = "\n\n".join(context_blocks)
        augmented_sys_prompt += f"\n\n[PROJECT CONTEXT - The following files from the project are provided for reference]:\n{context_str}\n[END OF PROJECT CONTEXT]"

    final_messages = [ChatMessage(role="system", content=augmented_sys_prompt)]

    # 1. If structured messages list was provided
    if req.messages:
        for m in req.messages:
            if m.role != "system":
                final_messages.append(m)
        return final_messages

    # 2. Process attachments
    attached_blocks = []
    if req.attached_files:
        for af in req.attached_files:
            fname = af.get("filename", "file.txt")
            cnt = af.get("content", "")
            attached_blocks.append(f"--- Attached File: {fname} ---\n{cnt}")

    user_text = req.prompt or ""
    if attached_blocks:
        attachment_str = f"[ATTACHED FILES: {len(attached_blocks)}]\n" + "\n\n".join(attached_blocks) + "\n[END OF ATTACHMENTS]\n\n"
        user_text = attachment_str + user_text

    # 3. If history was provided, add past turns
    if req.history:
        for h in req.history:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role != "system" and content:
                final_messages.append(ChatMessage(role=role, content=content))

    # 4. Append the current user prompt if not already the last message in history
    if user_text:
        if not req.history or (req.history and req.history[-1].get("content") != user_text):
            final_messages.append(ChatMessage(role="user", content=user_text))

    return final_messages

@router.post("")
async def chat_endpoint(req: ChatRequest):
    """Standard non-streaming chat endpoint."""
    provider = get_llm_provider(mode=req.mode, model_name_or_path=req.model)
    final_messages = build_context_augmented_messages(req)

    conv_id = req.conversation_id or str(uuid.uuid4())
    res = await provider.chat(
        messages=final_messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )

    now = time.time()
    chat_history_msgs = [m.model_dump() for m in final_messages if m.role != "system"]
    chat_history_msgs.append({"role": "assistant", "content": res.text})

    title_source = req.prompt or (req.messages[0].content if req.messages else "New Chat")
    conv_title = (title_source.strip().split("\n")[0])[:40]

    if conv_id not in CONVERSATIONS:
        CONVERSATIONS[conv_id] = {
            "id": conv_id,
            "title": conv_title or "New Chat",
            "created_at": now,
            "updated_at": now,
            "mode": provider.mode_name,
            "model": res.model,
            "messages": []
        }

    CONVERSATIONS[conv_id]["updated_at"] = now
    CONVERSATIONS[conv_id]["messages"] = chat_history_msgs

    return {
        "conversation_id": conv_id,
        "response": res.model_dump()
    }

@router.post("/stream")
async def stream_chat_endpoint(req: ChatRequest):
    """Server-Sent Events (SSE) streaming chat endpoint."""
    provider = get_llm_provider(mode=req.mode, model_name_or_path=req.model)
    final_messages = build_context_augmented_messages(req)
    conv_id = req.conversation_id or str(uuid.uuid4())

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': conv_id, 'model': req.model or 'active'})}\n\n"
        accumulated_text = []

        try:
            async for chunk in provider.stream_chat(
                messages=final_messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens
            ):
                if chunk.text:
                    accumulated_text.append(chunk.text)
                
                payload = {
                    "type": "token",
                    "text": chunk.text,
                    "token": chunk.text,
                    "done": chunk.done,
                    "model": chunk.model,
                    "finish_reason": chunk.finish_reason,
                    "duration": chunk.total_duration_seconds
                }
                yield f"data: {json.dumps(payload)}\n\n"

            now = time.time()
            full_reply = "".join(accumulated_text)
            
            chat_history_msgs = [m.model_dump() for m in final_messages if m.role != "system"]
            chat_history_msgs.append({"role": "assistant", "content": full_reply})

            title_source = req.prompt or (req.messages[0].content if req.messages else "New Chat")
            conv_title = (title_source.strip().split("\n")[0])[:40]

            if conv_id not in CONVERSATIONS:
                CONVERSATIONS[conv_id] = {
                    "id": conv_id,
                    "title": conv_title or "New Chat",
                    "created_at": now,
                    "updated_at": now,
                    "mode": provider.mode_name,
                    "model": req.model or "active",
                    "messages": []
                }
            CONVERSATIONS[conv_id]["updated_at"] = now
            CONVERSATIONS[conv_id]["messages"] = chat_history_msgs

            yield f"data: {json.dumps({'type': 'end', 'conversation_id': conv_id})}\n\n"
        except Exception as e:
            studio_logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/conversations")
async def list_conversations():
    """List all active conversation sessions."""
    res = []
    for cid, c in sorted(CONVERSATIONS.items(), key=lambda x: x[1]["updated_at"], reverse=True):
        res.append({
            "id": c["id"],
            "title": c["title"],
            "created_at": c["created_at"],
            "updated_at": c["updated_at"],
            "mode": c["mode"],
            "model": c["model"],
            "message_count": len(c["messages"])
        })
    return res

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieve full history of a conversation."""
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return CONVERSATIONS[conversation_id]

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id in CONVERSATIONS:
        del CONVERSATIONS[conversation_id]
        return {"status": "deleted", "id": conversation_id}
    raise HTTPException(status_code=404, detail="Conversation not found")
