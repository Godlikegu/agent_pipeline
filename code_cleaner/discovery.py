from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .models import RepoDiscovery


DATA_NAMES = ("load", "prepare", "preprocess", "dataset", "input", "read_data", "load_data", "load_and_preprocess")
MAIN_NAMES = ("solve", "run", "main", "infer", "predict", "reconstruct", "forward", "deconv", "restore",
              "inverse", "optimize", "iterate", "process", "compute", "execute", "pipeline")
EVAL_NAMES = ("eval", "evaluate", "metric", "score", "benchmark", "assess", "compare", "validate_result")
CLASS_NAMES = ("solver", "model", "runner", "pipeline", "engine")
SCRIPTISH_NAMES = ("demo", "example", "examples", "tutorial", "test", "tests", "benchmark")
UTILITY_NAMES = ("operation", "ops", "kernel", "helper", "helpers", "util", "utils", "common")
ROOT_DATA_NAMES = {
    "data",
    "dataset",
    "input",
    "inputs",
    "measurement",
    "measurements",
    "observation",
    "observations",
    "image",
    "images",
    "img",
    "stack",
    "volume",
    "x",
    "y",
}


@dataclass
class FileAnalysis:
    path: Path
    module_name: str
    has_main_guard: bool
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    main_candidates: List[str] = field(default_factory=list)
    data_candidates: List[str] = field(default_factory=list)
    eval_candidates: List[str] = field(default_factory=list)
    class_candidates: List[str] = field(default_factory=list)
    repo_imports: List[str] = field(default_factory=list)
    external_imports: List[str] = field(default_factory=list)
    score: int = 0
    reasons: List[str] = field(default_factory=list)
    readme_hits: List[str] = field(default_factory=list)
    script_hits: List[str] = field(default_factory=list)


def discover_repository(repo_root: Path) -> RepoDiscovery:
    python_files = sorted(
        path
        for path in repo_root.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts and ".git" not in path.parts
    )
    if not python_files:
        raise FileNotFoundError(f"No python files found under {repo_root}")

    module_index = {_module_name_for(path, repo_root): path for path in python_files}
    readme_files = [path for path in repo_root.iterdir() if path.is_file() and path.name.lower().startswith("readme")]
    readme_signals = _collect_readme_signals(readme_files)
    script_signals = _collect_script_signals(python_files, repo_root, module_index)

    analyses = [_analyze_file(path, repo_root, module_index, readme_signals, script_signals) for path in python_files]
    analyses.sort(key=_analysis_sort_key, reverse=True)
    entry = analyses[0]

    config_files = [
        path
        for path in repo_root.rglob("*")
        if path.is_file()
        and path.suffix in {".yml", ".yaml", ".toml", ".ini", ".cfg", ".json"}
        and ".git" not in path.parts
    ][:8]
    test_files = [path for path in repo_root.rglob("test*.py") if path.is_file()][:8]

    top_candidates = [
        f"{analysis.module_name} (score={analysis.score}; reasons={'; '.join(analysis.reasons[:4]) or 'none'})"
        for analysis in analyses[:5]
    ]

    return RepoDiscovery(
        repo_root=repo_root,
        entry_module=entry.path,
        python_files=python_files,
        readme_files=readme_files,
        config_files=config_files,
        test_files=test_files,
        data_candidates=entry.data_candidates,
        main_candidates=entry.main_candidates or [entry.path.stem],
        eval_candidates=entry.eval_candidates,
        class_candidates=entry.class_candidates,
        notes=[
            f"Selected entry module: {entry.module_name}",
            f"Selection reasons: {'; '.join(entry.reasons) or 'fallback to highest score'}",
            *[f"Candidate {index + 1}: {item}" for index, item in enumerate(top_candidates)],
        ],
    )


def _analyze_file(
    path: Path,
    repo_root: Path,
    module_index: Dict[str, Path],
    readme_signals: dict,
    script_signals: dict,
) -> FileAnalysis:
    module_name = _module_name_for(path, repo_root)
    source = path.read_text(encoding="utf-8", errors="ignore")
    analysis = FileAnalysis(
        path=path,
        module_name=module_name,
        has_main_guard='if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source,
    )

    stem_lower = path.stem.lower()
    repo_tokens = [token for token in repo_root.name.lower().replace("_", "-").split("-") if token]
    top_level_package = module_name.split(".", 1)[0]

    if path.parent == repo_root:
        analysis.score += 2
        analysis.reasons.append("top-level module")
    elif (path.parent / "__init__.py").exists():
        analysis.score += 3
        analysis.reasons.append("inside importable package")

    if any(token in stem_lower for token in SCRIPTISH_NAMES):
        analysis.score -= 8
        analysis.reasons.append("script-like filename penalty")

    if any(token == stem_lower or token in stem_lower for token in UTILITY_NAMES):
        analysis.score -= 4
        analysis.reasons.append("utility/helper filename penalty")

    if analysis.has_main_guard:
        analysis.score += 3
        analysis.reasons.append("has __main__ guard")

    if any(token in stem_lower for token in repo_tokens):
        analysis.score += 2
        analysis.reasons.append("matches repo name token")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        analysis.reasons.append("syntax parse failed")
        return analysis

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            lowered = node.name.lower()
            analysis.functions.append(node.name)
            if any(token in lowered for token in DATA_NAMES):
                analysis.data_candidates.append(node.name)
                analysis.score += 1
                analysis.reasons.append(f"data candidate `{node.name}`")
            if any(token in lowered for token in MAIN_NAMES):
                analysis.main_candidates.append(node.name)
                analysis.score += 2
                analysis.reasons.append(f"main candidate `{node.name}`")
            if any(token in lowered for token in EVAL_NAMES):
                analysis.eval_candidates.append(node.name)
                analysis.score += 1
                analysis.reasons.append(f"eval candidate `{node.name}`")
            if lowered == stem_lower or stem_lower in lowered or lowered in stem_lower:
                analysis.score += 3
                analysis.reasons.append(f"callable `{node.name}` matches filename")
            if _looks_like_entry_signature(node):
                analysis.score += 2
                analysis.reasons.append(f"`{node.name}` has entry-like signature")
        elif isinstance(node, ast.ClassDef):
            lowered = node.name.lower()
            analysis.classes.append(node.name)
            if any(token in lowered for token in CLASS_NAMES):
                analysis.class_candidates.append(node.name)
                analysis.score += 2
                analysis.reasons.append(f"class candidate `{node.name}`")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            repo_imports, external_imports = _classify_import_node(node, module_name, module_index)
            analysis.repo_imports.extend(repo_imports)
            analysis.external_imports.extend(external_imports)

    if module_name in readme_signals["modules"]:
        analysis.score += 14
        analysis.readme_hits.append(module_name)
        analysis.reasons.append("direct README import match")
    elif any(module_name.endswith(signal) for signal in readme_signals["modules"]):
        analysis.score += 8
        analysis.readme_hits.append("suffix-match")
        analysis.reasons.append("README suffix import match")

    matching_readme_callables = [name for name in analysis.functions if name in readme_signals["callables"]]
    if matching_readme_callables:
        analysis.score += 7 + len(matching_readme_callables)
        analysis.readme_hits.extend(matching_readme_callables)
        analysis.reasons.append(f"README callable match: {', '.join(matching_readme_callables[:3])}")

    referenced_by_scripts = script_signals["modules"].get(module_name, 0)
    if referenced_by_scripts:
        analysis.score += 10 + referenced_by_scripts
        analysis.script_hits.append(f"referenced-by-scripts:{referenced_by_scripts}")
        analysis.reasons.append(f"imported by demo/example scripts ({referenced_by_scripts})")

    callable_hits = sum(1 for name in analysis.functions if name in script_signals["callables"])
    if callable_hits:
        analysis.score += 4 + callable_hits
        analysis.reasons.append(f"script callable match count={callable_hits}")

    imported_by_repo = script_signals["reverse_imports"].get(module_name, 0)
    if imported_by_repo:
        analysis.score += min(imported_by_repo, 4)
        analysis.reasons.append(f"used by repo modules ({imported_by_repo})")

    if analysis.repo_imports and analysis.main_candidates:
        analysis.score += 2
        analysis.reasons.append("coordinates internal submodules")

    if top_level_package and top_level_package in module_name and path.stem == module_name.split(".")[-1]:
        analysis.score += 1

    analysis.data_candidates = _dedupe(analysis.data_candidates)
    analysis.main_candidates = _dedupe(analysis.main_candidates)
    analysis.eval_candidates = _dedupe(analysis.eval_candidates)
    analysis.class_candidates = _dedupe(analysis.class_candidates)
    analysis.reasons = _dedupe(analysis.reasons)
    return analysis


def _collect_readme_signals(readme_files: Sequence[Path]) -> dict:
    modules = set()
    callables = set()
    for path in readme_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        modules.update(re.findall(r"from\s+([A-Za-z_][\w\.]*)\s+import\s+[A-Za-z_][\w]*", text))
        modules.update(re.findall(r"import\s+([A-Za-z_][\w\.]*)", text))
        callables.update(re.findall(r"from\s+[A-Za-z_][\w\.]*\s+import\s+([A-Za-z_][\w]*)", text))
        callables.update(re.findall(r"\b([A-Za-z_][\w]*)\s*\(", text))
    return {"modules": modules, "callables": callables}


def _collect_script_signals(python_files: Sequence[Path], repo_root: Path, module_index: Dict[str, Path]) -> dict:
    module_counts: Dict[str, int] = {}
    reverse_imports: Dict[str, int] = {}
    callable_names = set()

    for path in python_files:
        if not _is_scriptish(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        imported_modules, imported_callables = _collect_ast_import_signals(tree, _module_name_for(path, repo_root), module_index)
        callable_names.update(imported_callables)
        for module_name in imported_modules:
            module_counts[module_name] = module_counts.get(module_name, 0) + 1

    for path in python_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        module_name = _module_name_for(path, repo_root)
        imported_modules, _ = _collect_ast_import_signals(tree, module_name, module_index)
        for imported_module in imported_modules:
            reverse_imports[imported_module] = reverse_imports.get(imported_module, 0) + 1

    return {
        "modules": module_counts,
        "callables": callable_names,
        "reverse_imports": reverse_imports,
    }


def _collect_ast_import_signals(tree: ast.AST, module_name: str, module_index: Dict[str, Path]) -> tuple[List[str], List[str]]:
    modules: List[str] = []
    callables: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _top_level_modules(module_index):
                    target = _best_matching_module(alias.name, module_index)
                    if target:
                        modules.append(target)
        elif isinstance(node, ast.ImportFrom):
            repo_imports, _ = _classify_import_node(node, module_name, module_index)
            modules.extend(repo_imports)
            for alias in node.names:
                if alias.name != "*":
                    callables.append(alias.name)
    return _dedupe(modules), _dedupe(callables)


def _classify_import_node(
    node: ast.Import | ast.ImportFrom,
    module_name: str,
    module_index: Dict[str, Path],
) -> tuple[List[str], List[str]]:
    repo_imports: List[str] = []
    external_imports: List[str] = []
    top_level_modules = _top_level_modules(module_index)

    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in top_level_modules:
                target = _best_matching_module(alias.name, module_index)
                if target:
                    repo_imports.append(target)
            else:
                external_imports.append(root)
        return _dedupe(repo_imports), _dedupe(external_imports)

    target_module = _resolve_import_from(module_name, node)
    if target_module:
        if target_module in module_index:
            repo_imports.append(target_module)
        elif target_module.split(".", 1)[0] in top_level_modules:
            best = _best_matching_module(target_module, module_index)
            if best:
                repo_imports.append(best)
        else:
            external_imports.append(target_module.split(".", 1)[0])
    return _dedupe(repo_imports), _dedupe(external_imports)


def _resolve_import_from(module_name: str, node: ast.ImportFrom) -> str | None:
    if node.level:
        base_parts = module_name.split(".")
        if module_name.endswith(".__init__"):
            base_parts = base_parts[:-1]
        climb = max(node.level - 1, 0)
        anchor = base_parts[:-climb] if climb else base_parts
        if module_name.endswith(".__init__"):
            anchor = anchor
        elif anchor:
            anchor = anchor[:-1]
        if node.module:
            extra = node.module.split(".")
            anchor = anchor + extra
        return ".".join(part for part in anchor if part)
    if node.module:
        return node.module
    return None


def _best_matching_module(module_name: str, module_index: Dict[str, Path]) -> str | None:
    if module_name in module_index:
        return module_name
    prefix = module_name + "."
    for candidate in module_index:
        if candidate.startswith(prefix):
            return candidate
    return None


def _module_name_for(path: Path, repo_root: Path) -> str:
    relative = path.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _analysis_sort_key(analysis: FileAnalysis) -> tuple:
    module_depth = analysis.module_name.count(".")
    direct_readme_match = bool(analysis.readme_hits)
    script_match = bool(analysis.script_hits)
    return (analysis.score, direct_readme_match, script_match, -module_depth, analysis.module_name)


def _top_level_modules(module_index: Dict[str, Path]) -> set[str]:
    return {module_name.split(".", 1)[0] for module_name in module_index}


def _looks_like_entry_signature(node: ast.FunctionDef) -> bool:
    positional = [arg.arg.lower() for arg in node.args.args]
    if not positional:
        return False
    if positional[0] not in ROOT_DATA_NAMES:
        return False
    return len(positional) <= 5


def _is_scriptish(path: Path) -> bool:
    lowered = path.stem.lower()
    return any(token in lowered for token in SCRIPTISH_NAMES)


def _dedupe(values: Iterable[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique
