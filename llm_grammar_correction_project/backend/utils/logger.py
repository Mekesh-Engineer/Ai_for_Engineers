import logging
import re
import sys
import time
from typing import Any, Dict, Optional

SECRET_PATTERNS = [
    (re.compile(r"(api[_-]?key|secret|token|password|auth)[\s:=]+['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?", re.IGNORECASE), r"\1=***REDACTED***"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE), "***REDACTED_API_KEY***"),
    (re.compile(r"hf_[a-zA-Z0-9]{20,}", re.IGNORECASE), "***REDACTED_HF_TOKEN***")
]

def sanitize_log_message(msg: str) -> str:
    """Sanitize sensitive keys/tokens from logs."""
    sanitized = str(msg)
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized

class StudioLogger:
    def __init__(self, name: str = "grammar_studio"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, msg: str, **kwargs):
        self.logger.info(sanitize_log_message(msg), **kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(sanitize_log_message(msg), **kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(sanitize_log_message(msg), **kwargs)

    def debug(self, msg: str, **kwargs):
        self.logger.debug(sanitize_log_message(msg), **kwargs)

    def log_inference(self, mode: str, model_id: str, request_type: str, duration_sec: float, token_stats: Optional[Dict[str, Any]] = None):
        stats_str = f" | Tokens: {token_stats}" if token_stats else ""
        self.info(f"[INFERENCE] Mode: {mode} | Model: {model_id} | Type: {request_type} | Latency: {duration_sec:.2f}s{stats_str}")

studio_logger = StudioLogger()
logger = studio_logger
