from __future__ import annotations

import ast
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List

from agents.base import get_full_response

from .models import RepoDiscovery

SCRIPT_HINTS = ("demo", "example", "examples", "main", "run", "infer", "predict")
LOW_PRIORITY_HINTS = ("test", "tests", "benchmark")


class LLMCleanerSynthesizer:
    def __init__(
        self,
        *,
        client,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 12000,
        max_loops: int = 3,
        snippet_char_limit: int = 4000,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_loops = max_loops
        self.snippet_char_limit = snippet_char_limit

    def generate_variant(self, discovery: RepoDiscovery) -> Dict:
        context_files = self._select_context_files(discovery)
        usage_files = self._select_usage_files(discovery)
        seed_files = self._seed_files(
            discovery,
            [path for path in discovery.python_files if not self._is_low_priority(path, discovery.repo_root)],
        )
        prompt = self._build_prompt(
            discovery,
            context_files=context_files,
            usage_files=usage_files,
        )
        response = get_full_response(
            client=self.client,
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python research-code cleaner. "
                        "Generate a single-file adapter that safely wraps an existing repository. "
                        "Return only executable Python code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            max_loops=self.max_loops,
            temperature=self.temperature,
        )
        cleaned_code = self._extract_code(response)
        self._validate_contract(cleaned_code)
        return {
            "code": cleaned_code,
            "source": "llm",
            "metadata": {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_loops": self.max_loops,
                "prompt": prompt,
                "response_preview": response[:2000],
                "selected_files": self._relative_paths(context_files, discovery),
                "usage_files": self._relative_paths(usage_files, discovery),
                "seed_files": self._relative_paths(seed_files, discovery),
                "entry_module_guess": discovery.entry_module.relative_to(discovery.repo_root).as_posix(),
                "static_candidates": {
                    "data_candidates": list(discovery.data_candidates),
                    "main_candidates": list(discovery.main_candidates),
                    "eval_candidates": list(discovery.eval_candidates),
                    "class_candidates": list(discovery.class_candidates),
                },
                "context_strategy": "static-hints-plus-import-closure",
                "selection_notes": [
                    "Static analysis is used as a hint source only.",
                    "Prompt includes usage/demo files when available.",
                    "Prompt expands local import dependencies from seed files.",
                ],
            },
        }

    def _build_prompt(self, discovery: RepoDiscovery, *, context_files: List[Path], usage_files: List[Path]) -> str:
        entry_module = discovery.entry_module.relative_to(discovery.repo_root).as_posix()
        readmes = self._format_file_group("README files", discovery.readme_files[:2])
        configs = self._format_file_group("Config files", discovery.config_files[:4])
        repo_files = self._format_file_group("Dependency-expanded Python files", context_files)
        usage_examples = self._format_file_group("Usage/demo Python files", usage_files)
        discovery_notes = "\n".join(f"- {note}" for note in discovery.notes[:8]) or "- none"
        data_candidates = ", ".join(discovery.data_candidates) or "none"
        main_candidates = ", ".join(discovery.main_candidates) or "none"
        eval_candidates = ", ".join(discovery.eval_candidates) or "none"
        class_candidates = ", ".join(discovery.class_candidates) or "none"
        inventory = self._inventory_summary(discovery.python_files, discovery.repo_root)
        return f"""You are generating `code_cleaned.py` for a research repository.

Runtime layout:
- The generated file will live in a run directory beside a `source_repo/` folder.
- The adapter must import modules from `source_repo`.
- The file will be executed as `python code_cleaned.py --stage run-all --input ... --output ... --metrics ... --context ...`.

Repository summary:
- Repo root: {discovery.repo_root.name}
- Entry module guess from static analysis: {entry_module}
- Data candidates from static analysis: {data_candidates}
- Main candidates from static analysis: {main_candidates}
- Eval candidates from static analysis: {eval_candidates}
- Class candidates from static analysis: {class_candidates}

Discovery notes:
{discovery_notes}

Requirements:
1. Output only Python code. Do not wrap it in Markdown fences.
2. The code must define exactly these user-facing functions:
   - `data_process(raw_inputs=None, context=None) -> dict`
   - `main_process(processed_inputs, context=None)`
   - `eval(processed_inputs, prediction, context=None) -> dict`
3. Add a CLI `_main()` that supports `--stage`, `--input`, `--output`, `--metrics`, `--context`.
4. Prefer conservative wrappers around the original repository rather than reimplementing algorithms.
5. The adapter must be self-contained in one file, but it may import from `source_repo`.
6. Use numpy for normalization and output saving where needed.
7. Be robust to missing optional functions. Fall back gracefully when the repo structure is incomplete.
8. If the original repository exposes only a main inference function, adapt around it instead of inventing extra behavior.
9. Keep behavior deterministic and validation-friendly.
10. Treat static-analysis candidates as hints only, not ground truth. Prefer direct evidence from source snippets, usage scripts, and import relationships.

Suggested implementation shape:
- resolve repo root from `Path(__file__).resolve().parent / "source_repo"`
- infer the best callable/module from the provided source, even if the entry-module guess is imperfect
- normalize input arrays
- call discovered candidates conservatively
- compute fallback metrics when repository eval hooks are unavailable

Repository inventory:
{inventory}

Repository context:

{readmes}

{configs}

{usage_examples}

{repo_files}
"""

    def _select_context_files(self, discovery: RepoDiscovery) -> List[Path]:
        python_files = [path for path in discovery.python_files if not self._is_low_priority(path, discovery.repo_root)]
        if not python_files:
            return [discovery.entry_module]

        import_graph = self._build_local_import_graph(python_files, discovery.repo_root)
        selected: List[Path] = []
        seen = set()
        queue = deque(self._seed_files(discovery, python_files))

        while queue and len(selected) < 8:
            path = queue.popleft()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            selected.append(path)
            for child in import_graph.get(path, []):
                if child not in seen:
                    queue.append(child)

        for path in python_files:
            if len(selected) >= 8:
                break
            if path in seen:
                continue
            selected.append(path)
            seen.add(path)

        return selected

    def _select_usage_files(self, discovery: RepoDiscovery) -> List[Path]:
        usage_files: List[Path] = []
        for path in discovery.python_files:
            relative = path.relative_to(discovery.repo_root).as_posix().lower()
            if any(token in relative for token in SCRIPT_HINTS):
                usage_files.append(path)
            if len(usage_files) >= 3:
                break
        return usage_files

    def _seed_files(self, discovery: RepoDiscovery, python_files: List[Path]) -> List[Path]:
        seeds: List[Path] = [discovery.entry_module]
        for path in python_files:
            if path in seeds:
                continue
            relative = path.relative_to(discovery.repo_root).as_posix().lower()
            if any(token in relative for token in SCRIPT_HINTS):
                seeds.append(path)
            elif path.parent == discovery.repo_root:
                seeds.append(path)
            if len(seeds) >= 5:
                break
        return seeds

    def _build_local_import_graph(self, python_files: Iterable[Path], repo_root: Path) -> Dict[Path, List[Path]]:
        module_index = {self._module_name_for(path, repo_root): path for path in python_files}
        graph: Dict[Path, List[Path]] = {}
        for path in python_files:
            graph[path] = self._resolve_local_imports(path, repo_root, module_index)
        return graph

    def _resolve_local_imports(self, path: Path, repo_root: Path, module_index: Dict[str, Path]) -> List[Path]:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except SyntaxError:
            return []

        module_name = self._module_name_for(path, repo_root)
        imported_paths: List[Path] = []
        for node in ast.walk(tree):
            target_modules: List[str] = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    match = self._best_local_match(alias.name, module_index)
                    if match:
                        target_modules.append(match)
            elif isinstance(node, ast.ImportFrom):
                resolved = self._resolve_import_from(module_name, node)
                if resolved:
                    match = self._best_local_match(resolved, module_index)
                    if match:
                        target_modules.append(match)
            for target in target_modules:
                imported_path = module_index.get(target)
                if imported_path and imported_path not in imported_paths:
                    imported_paths.append(imported_path)
        return imported_paths

    def _format_file_group(self, title: str, paths: List[Path]) -> str:
        if not paths:
            return f"{title}:\n- none"
        chunks = [f"{title}:"]
        for path in paths:
            try:
                snippet = self._read_snippet(path)
            except OSError as exc:
                snippet = f"<failed to read: {exc}>"
            chunks.append(f"FILE: {path.as_posix()}\n```python\n{snippet}\n```")
        return "\n\n".join(chunks)

    def _read_snippet(self, path: Path) -> str:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(content) <= self.snippet_char_limit:
            return content
        return content[: self.snippet_char_limit].rstrip() + "\n# ... truncated ..."

    def _extract_code(self, response: str) -> str:
        fenced = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        code = fenced.group(1) if fenced else response
        return code.strip() + "\n"

    def _validate_contract(self, cleaned_code: str) -> None:
        required_markers = ("def data_process", "def main_process", "def eval", "def _main")
        missing = [marker for marker in required_markers if marker not in cleaned_code]
        if missing:
            raise RuntimeError(f"LLM cleaner output is missing required adapter markers: {missing}")

    def _relative_paths(self, paths: List[Path], discovery: RepoDiscovery) -> List[str]:
        labels = []
        for path in paths:
            labels.append(path.relative_to(discovery.repo_root).as_posix())
        return labels

    def _inventory_summary(self, python_files: List[Path], repo_root: Path) -> str:
        items = []
        for path in python_files[:20]:
            items.append(f"- {path.relative_to(repo_root).as_posix()}")
        return "\n".join(items) or "- none"

    def _is_low_priority(self, path: Path, repo_root: Path) -> bool:
        relative = path.relative_to(repo_root).as_posix().lower()
        return any(token in relative for token in LOW_PRIORITY_HINTS)

    def _module_name_for(self, path: Path, repo_root: Path) -> str:
        relative = path.relative_to(repo_root).with_suffix("")
        parts = list(relative.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _best_local_match(self, module_name: str, module_index: Dict[str, Path]) -> str | None:
        if module_name in module_index:
            return module_name
        prefix = module_name + "."
        for candidate in module_index:
            if candidate.startswith(prefix):
                return candidate
        return None

    def _resolve_import_from(self, module_name: str, node: ast.ImportFrom) -> str | None:
        if node.level:
            base_parts = module_name.split(".")
            if module_name.endswith(".__init__"):
                base_parts = base_parts[:-1]
            climb = max(node.level - 1, 0)
            anchor = base_parts[:-climb] if climb else base_parts
            if not module_name.endswith(".__init__") and anchor:
                anchor = anchor[:-1]
            if node.module:
                anchor = anchor + node.module.split(".")
            return ".".join(part for part in anchor if part)
        if node.module:
            return node.module
        return None
