import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename: str) -> str:
    clean_name = secure_filename(filename)
    if not clean_name:
        clean_name = "uploaded_document.pdf"
    return clean_name
