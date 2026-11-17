# app/core/logging_setup.py
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

FORMAT = "%(asctime)s | [%(levelname)s] | %(threadName)s | %(message)s"


def setup_logging(log_dir: Optional[str] = "logs", level: str = "INFO", max_mb: float = 20.0,
                  backups: int = 10) -> logging.Logger:
    """Console + size-rotated file log (a 24/7 service must not fill the disk with one log file)."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | [%(levelname)s] | %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            Path(log_dir) / "factory_monitor.log", maxBytes=int(max_mb * 1024 * 1024), backupCount=backups,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(fh)
    return logging.getLogger("FactoryReID")
