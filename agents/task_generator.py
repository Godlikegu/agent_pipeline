"""Task description generation agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseAgent


DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).parent / "prompts" / "task_generator_system.txt"
)

DEFAULT_USER_PROMPT = (
    "Generate a detailed task_description for the scientific coding pipeline. "
    "The final description must turn the user request and any optional paper, "
    "code, README, config, and test context into an implementation-ready "
    "specification with clear inputs, outputs, process steps, validation, and "
    "implementation cautions."
)


@dataclass
class TaskGenerationResult:
    """Artifacts produced by task description generation."""

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


class TaskGeneratorAgent(BaseAgent):
    """Generates pipeline-ready task descriptions from paper markdown."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        system_prompt_path: Optional[str | Path] = None,
        max_tokens: int = 16000,
        max_loops: int = 3,
        paper_markdown_char_limit: int = 40000,
        default_user_prompt: Optional[str] = None,
    ):
        self._custom_system_prompt = system_prompt
        self._system_prompt_path = (
            Path(system_prompt_path).expanduser().resolve()
            if system_prompt_path is not None
            else DEFAULT_SYSTEM_PROMPT_PATH.resolve()
        )
        self.max_tokens = max_tokens
        self.max_loops = max_loops
        self.paper_markdown_char_limit = paper_markdown_char_limit
        self.default_user_prompt = default_user_prompt or DEFAULT_USER_PROMPT
        super().__init__(client=client, model_name=model_name, temperature=temperature)

    @classmethod
    def from_config(
        cls,
        client: Any,
        model_name: str,
        config: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> "TaskGeneratorAgent":
        task_gen_cfg = config or {}
        if "task_gen" in task_gen_cfg:
            task_gen_cfg = task_gen_cfg.get("task_gen", {})
        generator_cfg = task_gen_cfg.get("generator", {})

        return cls(
            client=client,
            model_name=model_name,
            temperature=generator_cfg.get("temperature", 0.2),
            system_prompt=system_prompt,
            system_prompt_path=generator_cfg.get("system_prompt_path"),
            max_tokens=generator_cfg.get("max_tokens", 16000),
            max_loops=generator_cfg.get("max_loops", 3),
            paper_markdown_char_limit=generator_cfg.get(
                "paper_markdown_char_limit", 40000
            ),
            default_user_prompt=task_gen_cfg.get("default_user_prompt")
            or DEFAULT_USER_PROMPT,
        )

    def clone_with_prompt(self, system_prompt: str) -> "TaskGeneratorAgent":
        return TaskGeneratorAgent(
            client=self.client,
            model_name=self.model_name,
            temperature=self.temperature,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            max_loops=self.max_loops,
            paper_markdown_char_limit=self.paper_markdown_char_limit,
            default_user_prompt=self.default_user_prompt,
        )

    def _build_system_prompt(self) -> str:
        if self._custom_system_prompt is not None:
            return self._custom_system_prompt.strip()
        if not self._system_prompt_path.exists():
            raise FileNotFoundError(
                f"Task generator prompt file not found: {self._system_prompt_path}"
            )
        return self._system_prompt_path.read_text(encoding="utf-8").strip()

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        if context.get("sources_bundle_text"):
            user_prompt = (context.get("user_prompt") or self.default_user_prompt).strip()
            return self.build_model_input_from_sources(
                user_prompt=user_prompt,
                sources_bundle_text=context["sources_bundle_text"],
            )

        paper_markdown = context.get("paper_markdown", "").strip()
        if not paper_markdown:
            raise ValueError("TaskGeneratorAgent requires paper_markdown in context.")

        user_prompt = (context.get("user_prompt") or self.default_user_prompt).strip()
        return self.build_model_input(
            paper_markdown=paper_markdown,
            user_prompt=user_prompt,
            paper_markdown_char_limit=self.paper_markdown_char_limit,
        )

    def generate(self, context: Dict[str, Any]) -> str:
        return self.call_llm(
            self._build_user_prompt(context),
            max_tokens=self.max_tokens,
            max_loops=self.max_loops,
        )

    def generate_from_markdown(
        self,
        paper_markdown: str,
        user_prompt: Optional[str] = None,
        save_path: Optional[str | Path] = None,
        paper_markdown_path: Optional[str | Path] = None,
    ) -> TaskGenerationResult:
        resolved_user_prompt = (user_prompt or self.default_user_prompt).strip()
        task_description = self.generate(
            {
                "paper_markdown": paper_markdown,
                "user_prompt": resolved_user_prompt,
            }
        ).strip()

        result = TaskGenerationResult(
            task_description=task_description,
            paper_markdown=paper_markdown,
            user_prompt=resolved_user_prompt,
            paper_markdown_path=(
                Path(paper_markdown_path).expanduser().resolve()
                if paper_markdown_path is not None
                else None
            ),
        )
        if save_path is not None:
            result.save_task_description(save_path)
        return result

    def generate_from_markdown_path(
        self,
        markdown_path: str | Path,
        user_prompt: Optional[str] = None,
        save_path: Optional[str | Path] = None,
    ) -> TaskGenerationResult:
        path = Path(markdown_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Paper markdown not found: {path}")
        return self.generate_from_markdown(
            paper_markdown=path.read_text(encoding="utf-8"),
            user_prompt=user_prompt,
            save_path=save_path,
            paper_markdown_path=path,
        )

    def generate_from_sources(
        self,
        sources: Any,
        save_path: Optional[str | Path] = None,
    ) -> TaskGenerationResult:
        resolved_user_prompt = (getattr(sources, "user_prompt", None) or self.default_user_prompt).strip()
        source_bundle_text = self.build_sources_bundle_text(sources)
        task_description = self.generate(
            {
                "sources_bundle_text": source_bundle_text,
                "user_prompt": resolved_user_prompt,
            }
        ).strip()

        result = TaskGenerationResult(
            task_description=task_description,
            paper_markdown=getattr(getattr(sources, "paper_markdown", None), "content", ""),
            user_prompt=resolved_user_prompt,
            paper_markdown_path=getattr(getattr(sources, "paper_markdown", None), "path", None),
        )
        output_path = save_path or getattr(sources, "output_path", None)
        if output_path is not None:
            result.save_task_description(output_path)
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
            "You are given a paper converted into markdown and a user request.\n"
            "Generate the final task_description for this inverse-problem pipeline.\n\n"
            "### User Prompt\n"
            f"{user_prompt.strip()}\n\n"
            "### Paper Markdown\n"
            f"{cleaned_markdown}\n"
        )

    @staticmethod
    def build_model_input_from_sources(
        user_prompt: str,
        sources_bundle_text: str,
    ) -> str:
        return (
            "You are given a scientific coding request and optional supporting materials.\n"
            "Generate the final task_description for the agentic pipeline.\n\n"
            "### Required User Prompt\n"
            f"{user_prompt.strip()}\n\n"
            "### Optional Source Materials\n"
            f"{sources_bundle_text.strip()}\n"
        )

    @staticmethod
    def build_sources_bundle_text(sources: Any) -> str:
        sections = []

        def add_section(title: str, content: str, path: Optional[Path] = None) -> None:
            clean = (content or "").strip()
            if not clean:
                return
            section = f"## {title}\n"
            if path is not None:
                section += f"Source Path: {path}\n\n"
            section += clean
            sections.append(section)

        paper = getattr(sources, "paper_markdown", None)
        if paper is not None:
            add_section("paper_markdown", getattr(paper, "content", ""), getattr(paper, "path", None))

        cleaned_code = getattr(sources, "cleaned_code", None)
        if cleaned_code is not None:
            add_section("cleaned_code", getattr(cleaned_code, "content", ""), getattr(cleaned_code, "path", None))

        readme = getattr(sources, "readme", None)
        if readme is not None:
            add_section("readme", getattr(readme, "content", ""), getattr(readme, "path", None))

        config_snippets = getattr(sources, "config_snippets", []) or []
        if config_snippets:
            config_parts = []
            for snippet in config_snippets:
                header = f"### {getattr(snippet, 'path', None) or 'config'}"
                config_parts.append(f"{header}\n{getattr(snippet, 'content', '')}".strip())
            add_section("config_snippets", "\n\n".join(config_parts))

        test_snippets = getattr(sources, "test_snippets", []) or []
        if test_snippets:
            test_parts = []
            for snippet in test_snippets:
                header = f"### {getattr(snippet, 'path', None) or 'test'}"
                test_parts.append(f"{header}\n{getattr(snippet, 'content', '')}".strip())
            add_section("test_snippets", "\n\n".join(test_parts))

        if not sections:
            sections.append("## no_optional_sources\nOnly the required user prompt is available.")

        return "\n\n".join(sections)
