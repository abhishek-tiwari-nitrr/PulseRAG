"""Small display helpers shared by the Streamlit views."""

from __future__ import annotations

from datetime import datetime

__all__ = ["format_score", "format_size", "format_timestamp", "status_icon"]
_BYTE_STEP = 1024.0


def format_score(value: float | None) -> str:
    """Render an optional 0-1 score to two decimals or "-" when absent."""
    return "-" if value is None else f"{value:.2f}"


def format_timestamp(value: datetime) -> str:
    """Render a UTC timestamp as `YYYY-MM-DD HH:MM:SS UTC`."""
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def status_icon(success: bool) -> str:
    """Return a pass/fail marker for table cells and expander titles."""
    return "✅ pass" if success else "❌ fail"


def format_size(size_bytes: int) -> str:
    """Render a byte count as `812 B`, `1.4 KB` or `12.0 MB`."""
    if size_bytes < _BYTE_STEP:
        return f"{size_bytes} B"
    if size_bytes < _BYTE_STEP**2:
        return f"{size_bytes / _BYTE_STEP:.1f} KB"
    return f"{size_bytes / _BYTE_STEP**2:.1f} MB"
