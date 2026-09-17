import os
from pathlib import Path
from typing import Union

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".html", ".css",
    ".json", ".yaml", ".yml", ".csv", ".pdf", ".docx",
    ".c", ".cpp", ".h", ".hpp", ".java", ".sh", ".bat", ".ps1",
    ".sql", ".xml", ".ini", ".env.example", ".gitignore", ".toml"
}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit

def is_safe_path(base_dir: Union[str, Path], path: Union[str, Path]) -> bool:
    """Ensure path resolves inside the base directory to prevent path traversal."""
    base_resolved = Path(base_dir).resolve()
    target_resolved = Path(path).resolve()
    try:
        target_resolved.relative_to(base_resolved)
        return True
    except ValueError:
        return False

def validate_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent malicious path components."""
    base = os.path.basename(filename)
    cleaned = "".join(c for c in base if c.isalnum() or c in "._- ")
    return cleaned if cleaned else "uploaded_file.txt"

def validate_file_upload(filename: str, file_size: int):
    if not filename:
        return False, "Filename cannot be empty."
    if not validate_extension(filename):
        return False, f"Unsupported file extension: {Path(filename).suffix}"
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File size exceeds {MAX_FILE_SIZE_BYTES / (1024*1024)} MB limit."
    return True, "Valid"
