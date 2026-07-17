from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_dir: str | Path) -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_academic_router_handler", False) for handler in root.handlers):
        return
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = RotatingFileHandler(
        directory / "application.log",
        maxBytes=int(os.getenv("LOG_MAX_BYTES", "5000000")),
        backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler._academic_router_handler = True  # type: ignore[attr-defined]
    root.setLevel(level)
    root.addHandler(file_handler)
    if not any(type(handler) is logging.StreamHandler for handler in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._academic_router_handler = True  # type: ignore[attr-defined]
        root.addHandler(console_handler)
