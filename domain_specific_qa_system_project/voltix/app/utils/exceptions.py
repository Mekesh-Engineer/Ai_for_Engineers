class VoltixBaseException(Exception):
    """Base exception for VOLTIX platform."""
    pass

class LLMConnectionError(VoltixBaseException):
    """Raised when Ollama or Cloud LLM daemon is unreachable."""
    pass

class RAGProcessingError(VoltixBaseException):
    """Raised when document extraction or FAISS indexing fails."""
    pass

class CalculationError(VoltixBaseException):
    """Raised when symbolic math or numerical solver fails."""
    pass
