import logging
import os
import sys
import json
from datetime import datetime, timezone
from config import LOG_FILE


class _JsonFileHandler(logging.FileHandler):
    """Writes one JSON object per log record to a .jsonl file."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                entry["exc"] = self.formatException(record.exc_info)
            self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger():
    """Sets up a logger with handlers for console, file, and structured JSON output."""
    logger = logging.getLogger("agent_system")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Formatters
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")

    # File Handler (human-readable)
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file {LOG_FILE}: {e}", file=sys.stderr)

    # Structured JSON log handler
    json_log_file = os.path.splitext(LOG_FILE)[0] + ".jsonl"
    try:
        json_handler = _JsonFileHandler(json_log_file, encoding="utf-8")
        json_handler.setLevel(logging.DEBUG)
        logger.addHandler(json_handler)
    except Exception as e:
        print(f"Warning: Could not create JSON log file {json_log_file}: {e}", file=sys.stderr)

    # Console Handler — force UTF-8 to avoid Windows-1252 charmap crashes
    try:
        utf8_stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
    except Exception:
        utf8_stdout = sys.stdout  # fallback if fileno() fails (e.g. redirected)
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# Single shared logger instance
logger = setup_logger()
