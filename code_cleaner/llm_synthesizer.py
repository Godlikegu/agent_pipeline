"""
LLM-driven code cleaning synthesizer.

Reads ALL Python source files from the repository, sends the full codebase
to the LLM, and generates a single standalone cleaned file that preserves
all functionality with improved structure and readability.

The cleaned code must be self-contained (no imports from source_repo) and
expose a unified interface:
  - load_and_preprocess_data(...)
  - main_process(...)
  - evaluate_results(...)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from agents.base import get_full_response

from .models import RepoDiscovery

TOTAL_SOURCE_CHAR_BUDGET = 300_000
PER_FILE_CHAR_LIMIT = 60_000
REQUIRED_FUNCTIONS = ("def load_and_preprocess_data", "def main_process", "def evaluate_results")


class LLMCleanerSynthesizer:
    def __init__(
        self,
        *,
        client,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 16000,
        max_loops: int = 3,
        paper_md: Optional[str] = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_loops = max_loops
        self.paper_md = paper_md

    def generate_variant(self, discovery: RepoDiscovery) -> Dict:
        all_sources = self._collect_all_sources(discovery)
        readme_text = self._collect_readmes(discovery)
        config_text = self._collect_configs(discovery)
        prompt = self._build_prompt(discovery, all_sources, readme_text, config_text)

        response = get_full_response(
            client=self.client,
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system_prompt()},
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
                "files_included": len(all_sources),
                "total_chars_sent": sum(len(c) for _, c in all_sources),
                "context_strategy": "full-repo-read",
            },
        }

    def _system_prompt(self) -> str:
        return (
            "You are an expert research-code engineer. Your task is to clean and "
            "restructure an entire research repository into a single, standalone, "
            "well-structured Python file.\n\n"
            "CRITICAL RULES:\n"
            "1. The output file must be SELF-CONTAINED. Do NOT import from the "
            "original repository or its package names. Include ALL necessary logic "
            "directly in the file. No `from source_repo import ...` or "
            "`from <original_package> import ...`.\n"
            "2. Preserve ALL algorithms, helper functions, and mathematical operations "
            "from the original code. Do not omit any functionality.\n"
            "3. Improve code quality: consistent naming, clear docstrings, remove "
            "dead code, but NEVER change the mathematical semantics.\n"
            "4. Output ONLY executable Python code. No markdown fences.\n"
            "5. The file must include a `_main()` CLI function using argparse with "
            "--stage, --input, --output, --metrics, --context flags. The --output "
            "flag MUST be used to save the result via np.save().\n"
            "6. There should be exactly ONE `if __name__ == '__main__': _main()` block.\n"
        )

    def _build_prompt(
        self,
        discovery: RepoDiscovery,
        all_sources: List[tuple[str, str]],
        readme_text: str,
        config_text: str,
    ) -> str:
        source_block = self._format_sources(all_sources)
        inventory = "\n".join(f"  - {rel}" for rel, _ in all_sources)

        paper_section = ""
        if self.paper_md:
            paper_text = self.paper_md[:40000]
            paper_section = f"\n## Paper (for context)\n{paper_text}\n"

        return f"""## Task
Clean the following research repository into a single standalone Python file
(`code_cleaned.py`) that preserves ALL functionality.

## Repository: {discovery.repo_root.name}

### File inventory ({len(all_sources)} Python files):
{inventory}

### Required interface

The cleaned file MUST define exactly these three top-level functions:

1. `load_and_preprocess_data(...)` — Load raw data from file/array, preprocess
   (background subtraction, normalization, upsampling, etc.), return processed
   data and any metadata needed by later stages.

2. `main_process(...)` — The core computational pipeline. This should contain
   the main algorithm(s) from the repository. For example, if the repo
   implements a sparse deconvolution, this function should contain the full
   Hessian iteration, deconvolution loops, forward operators, etc.
   Sub-functions like `forward_operator()`, `run_inversion()` are encouraged
   as helpers called within or alongside `main_process`.

3. `evaluate_results(...)` — Evaluate the output (compute metrics like PSNR,
   SSIM, MSE, save results). Compare against expected output if available.

### CLI entry point (MANDATORY)

The file MUST include a `_main()` function and `if __name__ == '__main__': _main()`
that uses argparse with these EXACT arguments:

```python
def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="run-all",
                        choices=["load", "main", "eval", "run-all"])
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument("--metrics", dest="metrics_path")
    parser.add_argument("--context", dest="context_path")
    args = parser.parse_args()
    # ... load context from args.context_path (JSON) ...
    # ... load input from args.input_path (npy or image file) ...
    processed = load_and_preprocess_data(...)
    prediction = main_process(...)
    metrics = evaluate_results(...)
    # CRITICAL: save output with np.save(args.output_path, result)
    if args.output_path:
        np.save(args.output_path, np.asarray(prediction, dtype=np.float64))
    if args.metrics_path:
        Path(args.metrics_path).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))
```

CRITICAL: The `--output` path MUST be used to save the prediction result via
`np.save()`. The `--input` path can be a `.npy` file (load with np.load) or
an image file. If `--input` is not provided, generate synthetic test data.

### Additional requirements
- Include ALL helper functions (math operations, kernels, utilities, etc.)
  from the original repo as module-level functions in the cleaned file.
- Do NOT add a separate `if __name__ == '__main__'` block besides the one
  that calls `_main()`. There should be exactly ONE main entry point.
- Use numpy, scipy, and standard library only unless the original repo
  requires specific packages (preserve those imports).
- The code must be immediately runnable without source_repo. Do NOT import
  anything from source_repo or the original package.
- Preserve numerical precision and algorithmic correctness.
- Clean up code style but do NOT simplify or skip any algorithm steps.
- Group code into clear sections with comments: HELPERS, LOAD, MAIN, EVAL.

{readme_text}

{config_text}

{paper_section}

## Full source code of the repository

{source_block}
"""

    def _collect_all_sources(self, discovery: RepoDiscovery) -> List[tuple[str, str]]:
        """Read ALL Python files, prioritizing core modules over tests."""
        sources: List[tuple[str, str]] = []
        budget_remaining = TOTAL_SOURCE_CHAR_BUDGET

        priority_files = self._prioritize_files(discovery)

        for path in priority_files:
            if budget_remaining <= 0:
                break
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not content.strip():
                continue
            rel = path.relative_to(discovery.repo_root).as_posix()
            truncated = content[:min(len(content), PER_FILE_CHAR_LIMIT, budget_remaining)]
            if len(truncated) < len(content):
                truncated += "\n# ... [truncated] ..."
            sources.append((rel, truncated))
            budget_remaining -= len(truncated)

        return sources

    def _prioritize_files(self, discovery: RepoDiscovery) -> List[Path]:
        """Sort files: core modules first, tests/benchmarks last."""
        low_priority = {"test", "tests", "benchmark", "benchmarks", "example", "examples", "doc", "docs"}
        core: List[Path] = []
        secondary: List[Path] = []
        low: List[Path] = []

        for path in discovery.python_files:
            parts = set(p.lower() for p in path.relative_to(discovery.repo_root).parts)
            if parts & low_priority:
                low.append(path)
            elif path == discovery.entry_module or path.parent == discovery.repo_root:
                core.insert(0, path)
            else:
                secondary.append(path)

        return core + secondary + low

    def _collect_readmes(self, discovery: RepoDiscovery) -> str:
        if not discovery.readme_files:
            return "## README\nNo README found."
        parts = ["## README files"]
        for path in discovery.readme_files[:2]:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")[:8000]
                parts.append(f"### {path.name}\n{content}")
            except OSError:
                pass
        return "\n\n".join(parts)

    def _collect_configs(self, discovery: RepoDiscovery) -> str:
        if not discovery.config_files:
            return ""
        parts = ["## Config files"]
        for path in discovery.config_files[:4]:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")[:4000]
                rel = path.relative_to(discovery.repo_root).as_posix()
                parts.append(f"### {rel}\n```\n{content}\n```")
            except OSError:
                pass
        return "\n\n".join(parts)

    def _format_sources(self, sources: List[tuple[str, str]]) -> str:
        blocks: List[str] = []
        for rel, content in sources:
            blocks.append(f"### FILE: {rel}\n```python\n{content}\n```")
        return "\n\n".join(blocks)

    def _extract_code(self, response: str) -> str:
        fenced = re.search(r"```(?:python)?\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip() + "\n"

        text = response.strip()
        if text.startswith("```"):
            first_nl = text.index("\n") if "\n" in text else len(text)
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip() + "\n"

    def _validate_contract(self, cleaned_code: str) -> None:
        missing = [fn for fn in REQUIRED_FUNCTIONS if fn not in cleaned_code]
        if missing:
            raise RuntimeError(
                f"LLM cleaner output is missing required functions: {missing}\n"
                "The cleaned code must define: load_and_preprocess_data, main_process, evaluate_results"
            )
