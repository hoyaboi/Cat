"""
Logging utility for attack pipeline.
"""
import os
from typing import Optional
from datetime import datetime


_log_file_handle = None


def log(message: str, level: str = "INFO", log_file: Optional[str] = None):
    """Simple logging function that prints and optionally writes to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    
    if log_file:
        global _log_file_handle
        if _log_file_handle is None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            _log_file_handle = open(log_file, 'w', encoding='utf-8')
        _log_file_handle.write(log_message + '\n')
        _log_file_handle.flush()


def close_log_file():
    """Close log file handle."""
    global _log_file_handle
    if _log_file_handle:
        _log_file_handle.close()
        _log_file_handle = None
