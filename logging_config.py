import logging
import logging.handlers
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import config

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class MetricsLogger:
    """Dedicated logger for metrics and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger("metrics")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.handlers.RotatingFileHandler(
            config.METRICS_LOG_FILE,
            maxBytes=config.LOG_ROTATION_SIZE,
            backupCount=config.LOG_BACKUP_COUNT
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def log_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Log a metric value with optional tags."""
        data = {
            "metric": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(data))
    
    def log_event(self, event_name: str, data: Dict[str, Any]):
        """Log a custom event."""
        event_data = {
            "event": event_name,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(event_data))


class ErrorTracker:
    """Track and categorize errors for analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger("errors")
        self.logger.setLevel(logging.ERROR)
        
        handler = logging.handlers.RotatingFileHandler(
            config.ERROR_LOG_FILE,
            maxBytes=config.LOG_ROTATION_SIZE,
            backupCount=config.LOG_BACKUP_COUNT
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        
        self.error_counts = {}
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        category: str = "unknown",
        severity: str = "error"
    ):
        """Log an error with context and categorization."""
        error_data = {
            "category": category,
            "severity": severity,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Track error frequency
        error_key = f"{category}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        error_data["occurrence_count"] = self.error_counts[error_key]
        
        self.logger.error(json.dumps(error_data))
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()


def setup_logging():
    """Configure comprehensive logging system."""
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Console handler (simple format for readability)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Main file handler (detailed format)
    file_handler = logging.handlers.RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_ROTATION_SIZE,
        backupCount=config.LOG_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Error file handler (for ERROR and CRITICAL only)
    error_handler = logging.handlers.RotatingFileHandler(
        config.ERROR_LOG_FILE,
        maxBytes=config.LOG_ROTATION_SIZE,
        backupCount=config.LOG_BACKUP_COUNT
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.LOG_LEVEL)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("docker").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Logging system initialized")
    logger.info(f"Log file: {config.LOG_FILE}")
    logger.info(f"Error log: {config.ERROR_LOG_FILE}")
    logger.info(f"Metrics log: {config.METRICS_LOG_FILE}")
    logger.info("=" * 80)


class ContextLogger:
    """Context manager for logging with additional context."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting: {self.context.get('operation', 'unknown')}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                f"Completed: {self.context.get('operation', 'unknown')} "
                f"(duration: {duration:.2f}s)"
            )
        else:
            self.logger.error(
                f"Failed: {self.context.get('operation', 'unknown')} "
                f"(duration: {duration:.2f}s) - {exc_type.__name__}: {exc_val}"
            )
        
        return False  # Don't suppress exceptions


# Global instances
metrics_logger = MetricsLogger()
error_tracker = ErrorTracker()