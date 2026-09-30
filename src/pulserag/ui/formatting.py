"""Small display helpers shared by the Streamlit views."""

from __future__ import annotations

from datetime import datetime

__all__ = ["format_score", "format_timestamp", "status_icon"]


def format_score(value: float | None) -> str:
    """Render an optional 0-1 score to two decimals or "-" when absent."""
    return "-" if value is None else f"{value:.2f}"


def format_timestamp(value: datetime) -> str:
    """Render a UTC timestamp as `YYYY-MM-DD HH:MM:SS UTC`."""
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def status_icon(success: bool) -> str:
    """Return a pass/fail marker for table cells and expander titles."""
    return "✅ pass" if success else "❌ fail"
