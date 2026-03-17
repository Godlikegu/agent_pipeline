"""Utilities for generating task descriptions from papers."""

from .pdf_parser import OCRScriptConfig, PaperMarkdownParser
from .task_generator import TaskDescriptionGenerator, TaskGenerationResult

__all__ = [
    "OCRScriptConfig",
    "PaperMarkdownParser",
    "TaskDescriptionGenerator",
    "TaskGenerationResult",
]
