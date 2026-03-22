"""
Static fallback facade generator.

When the LLM synthesizer fails, this module concatenates all discovered
Python source files into a single file and wraps them with the required
interface stubs (load_and_preprocess_data, main_process, evaluate_results).

The generated code is standalone — it does NOT import from source_repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .models import RepoDiscovery


class FacadeSynthesizer:
    """Concatenate the repo source into one file with interface wrappers."""

    def generate_variants(self, discovery: RepoDiscovery) -> list[str]:
        all_code = self._concat_sources(discovery)
        variant = self._wrap_with_interface(all_code, discovery)
        return [variant]

    def _concat_sources(self, discovery: RepoDiscovery) -> str:
        """Read and concatenate all Python files from the repository."""
        local_packages = self._detect_local_packages(discovery)
        blocks: List[str] = []
        for path in self._prioritized_files(discovery):
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if not content:
                continue
            rel = path.relative_to(discovery.repo_root).as_posix()
            cleaned = self._strip_main_guard(content)
            cleaned = self._strip_local_imports(cleaned, local_packages)
            blocks.append(f"# --- Source: {rel} ---\n{cleaned}")
        return "\n\n".join(blocks)

    def _detect_local_packages(self, discovery: RepoDiscovery) -> set[str]:
        """Detect the top-level package names within the repo."""
        packages: set[str] = set()
        for path in discovery.python_files:
            rel = path.relative_to(discovery.repo_root)
            top = rel.parts[0] if len(rel.parts) > 1 else None
            if top and (discovery.repo_root / top / "__init__.py").exists():
                packages.add(top)
        packages.add(discovery.repo_root.name.replace("-", "_"))
        return packages

    def _prioritized_files(self, discovery: RepoDiscovery) -> List[Path]:
        low_priority = {"test", "tests", "benchmark", "setup", "conf"}
        core: List[Path] = []
        rest: List[Path] = []
        for path in discovery.python_files:
            parts = {p.lower() for p in path.relative_to(discovery.repo_root).parts}
            if parts & low_priority:
                rest.append(path)
            else:
                core.append(path)
        return core + rest

    def _strip_main_guard(self, code: str) -> str:
        lines = code.split("\n")
        result: List[str] = []
        skip = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("if __name__") and "__main__" in stripped:
                skip = True
                continue
            if skip:
                if line and not line[0].isspace() and stripped:
                    skip = False
                    result.append(line)
                continue
            result.append(line)
        return "\n".join(result)

    def _strip_local_imports(self, code: str, local_packages: set[str]) -> str:
        """Remove import lines that reference the repo's own packages."""
        import re
        lines = code.split("\n")
        result: List[str] = []
        for line in lines:
            stripped = line.strip()
            skip = False
            if stripped.startswith("from ") or stripped.startswith("import "):
                for pkg in local_packages:
                    if re.match(rf"^(from|import)\s+{re.escape(pkg)}\b", stripped):
                        skip = True
                        break
            if not skip:
                result.append(line)
        return "\n".join(result)

    def _wrap_with_interface(self, source_code: str, discovery: RepoDiscovery) -> str:
        return f'''"""
Auto-generated fallback facade for {discovery.repo_root.name}.
All source files have been concatenated into this single standalone file.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ============================================================================
# CONCATENATED REPOSITORY SOURCE
# ============================================================================

{source_code}


# ============================================================================
# INTERFACE: load_and_preprocess_data
# ============================================================================

def load_and_preprocess_data(*args, **kwargs):
    """
    Load and preprocess data. Stub — please implement using the functions above.
    """
    raise NotImplementedError(
        "Static facade: load_and_preprocess_data must be manually implemented "
        "or use the LLM synthesizer to generate a proper cleaned file."
    )


# ============================================================================
# INTERFACE: main_process
# ============================================================================

def main_process(*args, **kwargs):
    """
    Main processing pipeline. Stub — please implement using the functions above.
    """
    raise NotImplementedError(
        "Static facade: main_process must be manually implemented "
        "or use the LLM synthesizer to generate a proper cleaned file."
    )


# ============================================================================
# INTERFACE: evaluate_results
# ============================================================================

def evaluate_results(*args, **kwargs):
    """
    Evaluate results and compute metrics. Stub — please implement.
    """
    raise NotImplementedError(
        "Static facade: evaluate_results must be manually implemented "
        "or use the LLM synthesizer to generate a proper cleaned file."
    )


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def _main():
    parser = argparse.ArgumentParser(description="Cleaned code facade.")
    parser.add_argument("--stage", default="run-all",
                        choices=["load", "main", "eval", "run-all"])
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument("--metrics", dest="metrics_path")
    parser.add_argument("--context", dest="context_path")
    args = parser.parse_args()

    context = {{}}
    if args.context_path and Path(args.context_path).exists():
        context = json.loads(Path(args.context_path).read_text(encoding="utf-8"))
    if args.input_path:
        context["input_path"] = args.input_path

    processed = load_and_preprocess_data(context=context)
    if args.stage == "load":
        print(json.dumps({{"status": "loaded"}}, ensure_ascii=False))
        return

    prediction = main_process(processed, context=context)
    if args.stage == "main":
        if args.output_path:
            np.save(args.output_path, np.asarray(prediction, dtype=np.float64))
        return

    metrics = evaluate_results(prediction, context=context)
    if args.output_path:
        np.save(args.output_path, np.asarray(prediction, dtype=np.float64))
    if args.metrics_path:
        Path(args.metrics_path).write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    _main()
'''
