import os
import csv
from pathlib import Path
from typing import List, Dict, Any
from app.utils.logger import get_logger
from app.utils.exceptions import RAGProcessingError

logger = get_logger("voltix.rag_loader")

class DocumentLoader:
    """Loader for PDF, DOCX, TXT, MD, and CSV files extracting text and metadata."""

    @staticmethod
    def load_document(file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise RAGProcessingError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif ext == ".docx":
            return DocumentLoader._load_docx(path)
        elif ext in [".txt", ".md"]:
            return DocumentLoader._load_txt(path)
        elif ext == ".csv":
            return DocumentLoader._load_csv(path)
        else:
            raise RAGProcessingError(f"Unsupported file format: {ext}")

    @staticmethod
    def _load_pdf(path: Path) -> List[Dict[str, Any]]:
        pages = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "page_number": idx + 1,
                        "text": text.strip(),
                        "source": path.name
                    })
        except Exception as e:
            logger.warning(f"pypdf failed on {path.name}, attempting pdfplumber fallback: {e}")
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    for idx, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages.append({
                                "page_number": idx + 1,
                                "text": text.strip(),
                                "source": path.name
                            })
            except Exception as pdf_err:
                raise RAGProcessingError(f"Failed to parse PDF {path.name}: {pdf_err}") from pdf_err
        return pages

    @staticmethod
    def _load_docx(path: Path) -> List[Dict[str, Any]]:
        try:
            import docx
            doc = docx.Document(str(path))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return [{"page_number": 1, "text": text, "source": path.name}]
        except Exception as e:
            raise RAGProcessingError(f"Failed to parse DOCX {path.name}: {e}") from e

    @staticmethod
    def _load_txt(path: Path) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [{"page_number": 1, "text": text, "source": path.name}]
        except Exception as e:
            raise RAGProcessingError(f"Failed to read TXT/MD {path.name}: {e}") from e

    @staticmethod
    def _load_csv(path: Path) -> List[Dict[str, Any]]:
        try:
            rows_text = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    header_line = " | ".join(headers)
                    rows_text.append(f"CSV Headers: {header_line}")
                for idx, row in enumerate(reader, start=1):
                    row_str = " | ".join(row)
                    rows_text.append(f"Row {idx}: {row_str}")

            full_text = "\n".join(rows_text)
            return [{"page_number": 1, "text": full_text, "source": path.name}]
        except Exception as e:
            raise RAGProcessingError(f"Failed to read CSV {path.name}: {e}") from e
