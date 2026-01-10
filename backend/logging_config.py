"""
Structured logging configuration with request ID tracking
"""
import logging
import json
import sys
from datetime import datetime
from contextvars import ContextVar
from typing import Optional

# Context variable to store request ID across async calls
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class StructuredFormatter(logging.Formatter):
    """
    Formats log records as JSON with structured fields.
    Includes timestamp, level, request_id, message, and context.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request ID if available
        request_id = request_id_ctx.get()
        if request_id:
            log_data["request_id"] = request_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields passed to logger
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO", structured: bool = True):
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Use JSON structured logging if True, else standard format
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Set uvicorn loggers to use same format
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.addHandler(handler)
        logger.propagate = False


def set_request_id(request_id: str):
    """Set the request ID for the current context."""
    request_id_ctx.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return request_id_ctx.get()


def log_with_context(logger: logging.Logger, level: str, message: str, **extra):
    """
    Log with additional context fields.
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **extra: Additional fields to include in structured log
    """
    log_func = getattr(logger, level.lower())
    
    # Create a log record with extra fields
    extra_data = {"extra_fields": extra} if extra else {}
    log_func(message, extra=extra_data)
