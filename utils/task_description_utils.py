from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agents.task_generator import DEFAULT_USER_PROMPT, TaskGenerationResult, TaskGeneratorAgent


DEFAULT_CODE_CHAR_LIMIT = 12000
DEFAULT_TEXT_CHAR_LIMIT = 8000


@dataclass
class SourceText:
    label: str
    content: str
    path: Optional[Path] = None


@dataclass
class TaskDescriptionSources:
    user_prompt: str
    paper_markdown: Optional[SourceText] = None
    cleaned_code: Optional[SourceText] = None
    readme: Optional[SourceText] = None
    config_snippets: List[SourceText] = field(default_factory=list)
    test_snippets: List[SourceText] = field(default_factory=list)
    task_description_path: Optional[Path] = None
    output_path: Optional[Path] = None


def read_text_file(path: Path, char_limit: Optional[int] = None) -> str:
    text = path.read_text(encoding="utf-8")
    if char_limit and len(text) > char_limit:
        text = text[:char_limit].rstrip() + "\n\n[TRUNCATED FOR CONTEXT WINDOW]"
    return text


def resolve_reference_code_path(task_info: Dict[str, object]) -> Path:
    preferred = task_info.get("cleaned_code_path") or task_info.get("gt_code_path")
    if not preferred:
        raise ValueError("Either cleaned_code_path or gt_code_path must be provided.")
    path = Path(str(preferred)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference code path not found: {path}")
    return path


def load_or_generate_task_description(
    *,
    task_info: Dict[str, object],
    config: Dict[str, object],
    client: Any,
    model_name: str,
    meta_data: Dict[str, object] = None,
) -> str:
    explicit_path = task_info.get("task_description_path")
    if explicit_path:
        desc_path = Path(str(explicit_path)).expanduser().resolve()
        if desc_path.exists():
            return read_text_file(desc_path)

    sources = build_task_description_sources(task_info, config, meta_data=meta_data)
    generator = TaskGeneratorAgent.from_config(client=client, model_name=model_name, config=config)
    result = generator.generate_from_sources(
        sources=sources,
        save_path=sources.output_path,
    )
    return result.task_description


def build_task_description_sources(
    task_info: Dict[str, object],
    config: Dict[str, object],
    meta_data: Dict[str, object] = None,
) -> TaskDescriptionSources:
    task_gen_cfg = config.get("task_gen", {})
    default_desc_dir = config.get("paths", {}).get(
        "task_descriptions_dir",
        "./data/task_descriptions",
    )
    task_name = str(task_info.get("name") or "task")
    user_prompt = str(
        task_info.get("user_prompt")
        or task_gen_cfg.get("default_user_prompt")
        or DEFAULT_USER_PROMPT
    ).strip()
    output_path = Path(
        str(
            task_info.get("task_description_path")
            or task_info.get("task_description_output_path")
            or (Path(default_desc_dir) / f"{task_name}_description.md")
        )
    ).expanduser().resolve()

    sources = TaskDescriptionSources(
        user_prompt=user_prompt,
        task_description_path=_optional_path(task_info.get("task_description_path")),
        output_path=output_path,
    )

    paper_path = _resolve_paper_markdown_path(task_info, config)
    if paper_path and paper_path.exists():
        sources.paper_markdown = SourceText(
            label="paper_markdown",
            content=read_text_file(paper_path, task_gen_cfg.get("paper_markdown_char_limit", 40000)),
            path=paper_path,
        )

    cleaned_code_path = _optional_path(task_info.get("cleaned_code_path"))
    if cleaned_code_path and cleaned_code_path.is_dir():
        candidate = cleaned_code_path / "code_cleaned.py"
        cleaned_code_path = candidate if candidate.exists() else None
    if cleaned_code_path and cleaned_code_path.exists():
        sources.cleaned_code = SourceText(
            label="cleaned_code",
            content=read_text_file(cleaned_code_path, DEFAULT_CODE_CHAR_LIMIT),
            path=cleaned_code_path,
        )

    readme_path = _optional_path(task_info.get("readme_path"))
    if readme_path is None:
        reference_root = _optional_path(task_info.get("cleaned_code_path")) or _optional_path(task_info.get("gt_code_path"))
        readme_path = auto_detect_readme(reference_root)
    if readme_path and readme_path.exists():
        sources.readme = SourceText(
            label="readme",
            content=read_text_file(readme_path, DEFAULT_TEXT_CHAR_LIMIT),
            path=readme_path,
        )

    # Inject meta_data as a config snippet so TaskGeneratorAgent naturally incorporates it
    if meta_data:
        import json as _json
        meta_content = _json.dumps(meta_data, indent=2, default=str)
        sources.config_snippets.append(
            SourceText(
                label="meta_data",
                content=f"Physical parameters and data format specification:\n{meta_content}",
            )
        )

    config_paths = _coerce_path_list(task_info.get("config_paths"))
    if not config_paths:
        reference_root = _optional_path(task_info.get("cleaned_code_path")) or _optional_path(task_info.get("gt_code_path"))
        config_paths = auto_detect_config_paths(reference_root)
    for path in config_paths:
        if path.exists():
            sources.config_snippets.append(
                SourceText(label="config", content=read_text_file(path, DEFAULT_TEXT_CHAR_LIMIT), path=path)
            )

    test_paths = _coerce_path_list(task_info.get("test_paths"))
    if not test_paths:
        reference_root = _optional_path(task_info.get("cleaned_code_path")) or _optional_path(task_info.get("gt_code_path"))
        test_paths = auto_detect_test_paths(reference_root)
    for path in test_paths:
        if path.exists():
            sources.test_snippets.append(
                SourceText(label="test", content=read_text_file(path, DEFAULT_TEXT_CHAR_LIMIT), path=path)
            )

    return sources


def auto_detect_readme(reference_path: Optional[Path]) -> Optional[Path]:
    if reference_path is None:
        return None
    root = reference_path if reference_path.is_dir() else reference_path.parent
    candidates = [
        root / "README.md",
        root / "README.rst",
        root / "README.txt",
        root / "readme.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def auto_detect_config_paths(reference_path: Optional[Path]) -> List[Path]:
    if reference_path is None:
        return []
    root = reference_path if reference_path.is_dir() else reference_path.parent
    candidates = [
        root / "requirements.txt",
        root / "pyproject.toml",
        root / "setup.py",
        root / "setup.cfg",
        root / "environment.yml",
        root / "environment.yaml",
    ]
    discovered = [candidate.resolve() for candidate in candidates if candidate.exists()]
    config_dir = root / "config"
    if config_dir.exists():
        discovered.extend(sorted(config_dir.glob("*.y*ml"))[:3])
    return _dedupe_paths(discovered)


def auto_detect_test_paths(reference_path: Optional[Path]) -> List[Path]:
    if reference_path is None:
        return []
    root = reference_path if reference_path.is_dir() else reference_path.parent
    candidates: List[Path] = []
    for folder_name in ("tests", "test"):
        folder = root / folder_name
        if not folder.exists():
            continue
        candidates.extend(sorted(folder.rglob("test*.py"))[:3])
        candidates.extend(sorted(folder.rglob("*_test.py"))[:3])
    return _dedupe_paths(candidates[:4])


def _resolve_paper_markdown_path(task_info: Dict[str, object], config: Dict[str, object]) -> Optional[Path]:
    explicit = _optional_path(task_info.get("paper_markdown_path"))
    if explicit:
        return explicit
    paper_dir = (
        config.get("paths", {}).get("paper_markdown_dir")
        or config.get("task_gen", {}).get("paper_markdown_dir")
    )
    task_name = task_info.get("name")
    if paper_dir and task_name:
        candidate = Path(str(paper_dir)).expanduser().resolve() / f"{task_name}.md"
        if candidate.exists():
            return candidate
    return None


def _optional_path(value: object) -> Optional[Path]:
    if not value:
        return None
    return Path(str(value)).expanduser().resolve()


def _coerce_path_list(value: object) -> List[Path]:
    if not value:
        return []
    if isinstance(value, (str, Path)):
        return [Path(str(value)).expanduser().resolve()]
    if isinstance(value, Sequence):
        return [Path(str(item)).expanduser().resolve() for item in value if item]
    return []


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        unique.append(resolved)
        seen.add(resolved)
    return unique
