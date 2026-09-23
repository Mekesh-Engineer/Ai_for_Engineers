import re

class TextParser:
    """Cleans text noise, preserves LaTeX math expressions, and formats tables."""

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove weird non-printable control chars except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        # Normalize multiple spaces
        text = re.sub(r" {2,}", " ", text)
        # Normalize multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
