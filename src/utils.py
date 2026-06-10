"""General project utility functions."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal metric as a percentage string."""
    return f"{value * 100:.{decimals}f}%"
