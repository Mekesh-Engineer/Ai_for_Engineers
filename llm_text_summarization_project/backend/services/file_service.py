import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from backend.utils.logger import studio_logger
from backend.utils.security import validate_extension, sanitize_filename

class ExtractedDocument:
    def __init__(self, filename: str, content: str, file_type: str, size_bytes: int, metadata: Optional[Dict[str, Any]] = None):
        self.filename = filename
        self.content = content
        self.file_type = file_type
        self.size_bytes = size_bytes
        self.metadata = metadata or {}
        self.char_count = len(content)
        self.word_count = len(content.split())
        self.estimated_tokens = int(self.word_count * 1.3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "estimated_tokens": self.estimated_tokens,
            "metadata": self.metadata
        }

class FileService:
    """Extracts, cleans, estimates tokens, and chunks documents across multiple formats."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from raw text."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove null bytes
        text = text.replace("\x00", "")
        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def extract_text_from_bytes(cls, filename: str, file_bytes: bytes) -> ExtractedDocument:
        """Extract text from raw byte content based on extension."""
        ext = Path(filename).suffix.lower()
        size_bytes = len(file_bytes)
        metadata = {"extension": ext}

        if not validate_extension(filename):
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .txt, .md, .py, .js, .ts, .html, .css, .json, .yaml, .csv, .pdf, .docx")

        extracted_text = ""

        try:
            if ext in [".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".yaml", ".yml", ".c", ".cpp", ".h", ".java", ".sh", ".bat", ".ps1", ".sql", ".toml"]:
                try:
                    extracted_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    extracted_text = file_bytes.decode("latin-1", errors="replace")

            elif ext == ".json":
                try:
                    raw_str = file_bytes.decode("utf-8")
                    data = json.loads(raw_str)
                    extracted_text = json.dumps(data, indent=2)
                except Exception:
                    extracted_text = file_bytes.decode("utf-8", errors="replace")

            elif ext == ".csv":
                try:
                    raw_str = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    raw_str = file_bytes.decode("latin-1", errors="replace")
                
                reader = csv.reader(raw_str.splitlines())
                rows = list(reader)
                metadata["total_rows"] = len(rows)
                metadata["total_columns"] = len(rows[0]) if rows else 0
                
                # Format CSV neatly for LLM consumption
                formatted_lines = []
                for i, row in enumerate(rows[:200]):  # cap initial table preview rows if massive
                    formatted_lines.append(" | ".join(row))
                extracted_text = "\n".join(formatted_lines)

            elif ext == ".pdf":
                try:
                    import pypdf
                    import io
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    pages_text = []
                    for page_idx, page in enumerate(reader.pages):
                        page_content = page.extract_text() or ""
                        pages_text.append(f"--- Page {page_idx + 1} ---\n{page_content}")
                    metadata["total_pages"] = len(reader.pages)
                    extracted_text = "\n\n".join(pages_text)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse PDF document: {str(e)}")

            elif ext == ".docx":
                try:
                    import docx
                    import io
                    doc = docx.Document(io.BytesIO(file_bytes))
                    paras = [p.text for p in doc.paragraphs if p.text.strip()]
                    for table in doc.tables:
                        for row in table.rows:
                            paras.append(" | ".join(cell.text.strip() for cell in row.cells))
                    extracted_text = "\n\n".join(paras)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse DOCX document: {str(e)}")

            else:
                extracted_text = file_bytes.decode("utf-8", errors="replace")

        except Exception as e:
            studio_logger.error(f"Error extracting text from '{filename}': {e}")
            raise

        cleaned = cls.clean_text(extracted_text)
        if not cleaned:
            raise ValueError(f"Extracted content from '{filename}' is empty or unreadable.")

        return ExtractedDocument(
            filename=filename,
            content=cleaned,
            file_type=ext[1:].upper() if ext.startswith(".") else ext.upper(),
            size_bytes=size_bytes,
            metadata=metadata
        )

    @classmethod
    def chunk_text(
        cls,
        text: str,
        max_chunk_tokens: int = 1500,
        overlap_tokens: int = 150
    ) -> List[Dict[str, Any]]:
        """Split large text into overlapping chunks based on paragraph and sentence boundaries."""
        words = text.split()
        total_words = len(words)
        
        # Approximate words per chunk
        target_chunk_words = int(max_chunk_tokens / 1.3)
        overlap_words = int(overlap_tokens / 1.3)

        if total_words <= target_chunk_words:
            return [{
                "chunk_index": 0,
                "text": text,
                "word_count": total_words,
                "estimated_tokens": int(total_words * 1.3),
                "is_single_chunk": True
            }]

        chunks = []
        # First split into paragraphs
        paragraphs = text.split("\n\n")
        current_chunk_paragraphs = []
        current_word_count = 0
        chunk_idx = 0

        for para in paragraphs:
            para_words = len(para.split())
            if current_word_count + para_words > target_chunk_words and current_chunk_paragraphs:
                chunk_str = "\n\n".join(current_chunk_paragraphs)
                chunks.append({
                    "chunk_index": chunk_idx,
                    "text": chunk_str,
                    "word_count": len(chunk_str.split()),
                    "estimated_tokens": int(len(chunk_str.split()) * 1.3),
                    "is_single_chunk": False
                })
                chunk_idx += 1
                # Retain last paragraph for context overlap if feasible
                if len(current_chunk_paragraphs) > 1 and len(current_chunk_paragraphs[-1].split()) <= overlap_words:
                    current_chunk_paragraphs = [current_chunk_paragraphs[-1], para]
                    current_word_count = len(current_chunk_paragraphs[0].split()) + para_words
                else:
                    current_chunk_paragraphs = [para]
                    current_word_count = para_words
            else:
                current_chunk_paragraphs.append(para)
                current_word_count += para_words

        if current_chunk_paragraphs:
            chunk_str = "\n\n".join(current_chunk_paragraphs)
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_str,
                "word_count": len(chunk_str.split()),
                "estimated_tokens": int(len(chunk_str.split()) * 1.3),
                "is_single_chunk": False
            })

        return chunks

file_service = FileService()
