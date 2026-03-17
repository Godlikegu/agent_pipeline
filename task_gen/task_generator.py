"""LLM-driven task description generation from paper Markdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents.base import get_full_response

from .pdf_parser import PaperMarkdownParser


DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "task_generator_system.txt"


@dataclass
class TaskGenerationResult:
    """Artifacts produced during task description generation."""

    task_description: str
    paper_markdown: str
    user_prompt: str
    paper_markdown_path: Optional[Path] = None
    task_description_path: Optional[Path] = None

    def save_task_description(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.task_description, encoding="utf-8")
        self.task_description_path = output_path
        return output_path


class TaskDescriptionGenerator:
    """Generates a pipeline-ready task description from paper content."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        system_prompt: Optional[str] = None,
        system_prompt_path: Optional[str | Path] = None,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        max_loops: int = 3,
        paper_markdown_char_limit: int = 40000,
    ):
        self.client = client
        self.model_name = model_name
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else self._load_system_prompt(system_prompt_path)
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_loops = max_loops
        self.paper_markdown_char_limit = paper_markdown_char_limit

    @classmethod
    def from_config(
        cls,
        client: Any,
        model_name: str,
        config: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> "TaskDescriptionGenerator":
        config = config or {}
        generator_cfg = config.get("generator", {})
        return cls(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            system_prompt_path=generator_cfg.get("system_prompt_path"),
            temperature=generator_cfg.get("temperature", 0.2),
            max_tokens=generator_cfg.get("max_tokens", 16000),
            max_loops=generator_cfg.get("max_loops", 3),
            paper_markdown_char_limit=generator_cfg.get(
                "paper_markdown_char_limit", 40000
            ),
        )

    def clone_with_prompt(self, system_prompt: str) -> "TaskDescriptionGenerator":
        return TaskDescriptionGenerator(
            client=self.client,
            model_name=self.model_name,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_loops=self.max_loops,
            paper_markdown_char_limit=self.paper_markdown_char_limit,
        )

    def generate_from_markdown_path(
        self,
        markdown_path: str | Path,
        user_prompt: str,
        save_path: Optional[str | Path] = None,
    ) -> TaskGenerationResult:
        path = Path(markdown_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Paper Markdown not found: {path}")
        return self.generate_from_markdown(
            paper_markdown=path.read_text(encoding="utf-8"),
            user_prompt=user_prompt,
            save_path=save_path,
            paper_markdown_path=path,
        )

    def generate_from_markdown(
        self,
        paper_markdown: str,
        user_prompt: str,
        save_path: Optional[str | Path] = None,
        paper_markdown_path: Optional[str | Path] = None,
    ) -> TaskGenerationResult:
        model_input = self.build_model_input(
            paper_markdown=paper_markdown,
            user_prompt=user_prompt,
            paper_markdown_char_limit=self.paper_markdown_char_limit,
        )
        task_description = get_full_response(
            client=self.client,
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": model_input},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_loops=self.max_loops,
        ).strip()

        result = TaskGenerationResult(
            task_description=task_description,
            paper_markdown=paper_markdown,
            user_prompt=user_prompt,
            paper_markdown_path=(
                Path(paper_markdown_path).expanduser().resolve()
                if paper_markdown_path is not None
                else None
            ),
        )
        if save_path is not None:
            result.save_task_description(save_path)
        return result

    @staticmethod
    def build_model_input(
        paper_markdown: str,
        user_prompt: str,
        paper_markdown_char_limit: Optional[int] = None,
    ) -> str:
        cleaned_markdown = paper_markdown.strip()
        if paper_markdown_char_limit and len(cleaned_markdown) > paper_markdown_char_limit:
            cleaned_markdown = cleaned_markdown[:paper_markdown_char_limit].rstrip()
            cleaned_markdown += "\n\n[TRUNCATED FOR CONTEXT WINDOW]"

        return (
            "You are given a paper converted into Markdown and a user request.\n"
            "Generate the final task_description for this inverse-problem pipeline.\n\n"
            "### User Prompt\n"
            f"{user_prompt.strip()}\n\n"
            "### Paper Markdown\n"
            f"{cleaned_markdown}\n"
        )

    def _load_system_prompt(self, system_prompt_path: Optional[str | Path]) -> str:
        path = (
            Path(system_prompt_path).expanduser().resolve()
            if system_prompt_path is not None
            else DEFAULT_SYSTEM_PROMPT_PATH.resolve()
        )
        if not path.exists():
            raise FileNotFoundError(f"Task generator prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
