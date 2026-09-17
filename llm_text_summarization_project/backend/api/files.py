from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend.services.file_service import file_service
from backend.services.summarizer import document_summarizer
from backend.services.llm_service import get_llm_provider
from backend.config.settings import settings
from backend.utils.logger import studio_logger

router = APIRouter(prefix="/api/files", tags=["Files"])

class SummarizeTextRequest(BaseModel):
    text: str = Field(..., description="Document text to process")
    filename: Optional[str] = "document"
    task: Optional[str] = "summarize"
    mode: Optional[str] = None
    custom_instructions: Optional[str] = None
    system_prompt: Optional[str] = None

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and extract text from project or document file (.txt, .md, .py, .pdf, .docx, etc.)."""
    try:
        content_bytes = await file.read()
        if len(content_bytes) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds maximum allowed 50MB limit.")

        doc = file_service.extract_text_from_bytes(file.filename, content_bytes)
        chunks = file_service.chunk_text(
            doc.content,
            max_chunk_tokens=settings.max_chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens
        )

        return {
            "status": "success",
            "document": doc.to_dict(),
            "content_preview": doc.content[:1500] + ("..." if len(doc.content) > 1500 else ""),
            "full_content": doc.content,
            "chunk_count": len(chunks),
            "is_hierarchical_needed": len(chunks) > 1
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        studio_logger.error(f"Upload error for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@router.post("/summarize")
async def summarize_document_endpoint(req: SummarizeTextRequest):
    """Execute single-pass or hierarchical multi-chunk summarization on text."""
    try:
        result = await document_summarizer.summarize_document(
            text=req.text,
            filename=req.filename or "document",
            mode=req.mode,
            task=req.task or "summarize",
            system_prompt=req.system_prompt,
            custom_instructions=req.custom_instructions
        )
        return result
    except Exception as e:
        studio_logger.error(f"Summarization endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
