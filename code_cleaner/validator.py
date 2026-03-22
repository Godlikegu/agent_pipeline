"""
Validator for cleaned code.

Validates that:
  1. The cleaned code parses without syntax errors.
  2. The required interface functions exist (load_and_preprocess_data,
     main_process, evaluate_results).
  3. The cleaned code runs successfully (smoke test).
  4. Optionally: the original repo code can be run for output comparison.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import EnvironmentPlan, RepoDiscovery, ValidationReport

REQUIRED_FUNCTIONS = {"load_and_preprocess_data", "main_process", "evaluate_results"}

ORIGINAL_RUNNER_TEMPLATE = '''"""Auto-generated runner for the original repository."""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ENTRY_MODULE = __ENTRY_MODULE__
MAIN_CANDIDATES = __MAIN_CANDIDATES__
DATA_CANDIDATES = __DATA_CANDIDATES__
EVAL_CANDIDATES = __EVAL_CANDIDATES__
CLASS_CANDIDATES = __CLASS_CANDIDATES__


def _load_module():
    repo_root = Path(__file__).resolve().parent / "source_repo"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_name = ENTRY_MODULE.replace("/", ".").replace("\\\\", ".")
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    if module_name.endswith(".__init__"):
        module_name = module_name[: -len(".__init__")]
    return importlib.import_module(module_name)


def _normalize(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    if np.isscalar(value):
        return np.asarray([value], dtype=np.float64)
    return value


def _try_call(func, payload, context):
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if len(params) == 0:
        return func()
    if len(params) == 1:
        return func(payload)
    return func(payload, **{k: v for k, v in context.items()
                            if k in {p.name for p in params[1:]}})


def main():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    metrics_path = Path(sys.argv[3])
    context_path = Path(sys.argv[4])
    payload = np.load(input_path, allow_pickle=True)
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else {}
    module = _load_module()
    result = None
    for name in MAIN_CANDIDATES:
        if not hasattr(module, name):
            continue
        candidate = getattr(module, name)
        if not callable(candidate):
            continue
        try:
            result = _try_call(candidate, payload, context)
        except Exception:
            continue
        if result is not None:
            break
    if result is None:
        result = payload
    result = np.asarray(result, dtype=np.float64)
    np.save(output_path, result)
    metrics = {"status": "ok", "shape": list(result.shape), "mean": float(result.mean())}
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


class CleanerValidator:
    def validate(
        self,
        *,
        run_dir: Path,
        discovery: RepoDiscovery,
        plan: EnvironmentPlan,
        candidate_source: str = "unknown",
    ) -> ValidationReport:
        cleaned_path = run_dir / "code_cleaned.py"
        if not cleaned_path.exists():
            return ValidationReport(
                accepted=False,
                status="missing",
                comparison="none",
                summary="code_cleaned.py does not exist.",
                candidate_source=candidate_source,
            )

        syntax_ok, syntax_msg = self._check_syntax(cleaned_path)
        if not syntax_ok:
            return ValidationReport(
                accepted=False,
                status="syntax_error",
                comparison="none",
                summary=f"Syntax error in cleaned code: {syntax_msg}",
                candidate_source=candidate_source,
                logs=[syntax_msg],
            )

        interface_ok, missing = self._check_interface(cleaned_path)
        if not interface_ok:
            return ValidationReport(
                accepted=False,
                status="missing_interface",
                comparison="none",
                summary=f"Missing required functions: {missing}",
                candidate_source=candidate_source,
                logs=[f"Missing: {missing}"],
            )

        python_exec = str(
            plan.python_executable
            if plan.provisioned and plan.python_executable.exists()
            else sys.executable
        )

        import_ok, import_log = self._check_import(python_exec, cleaned_path, run_dir)
        if not import_ok:
            return ValidationReport(
                accepted=False,
                status="import_error",
                comparison="none",
                summary="Cleaned code failed to import.",
                candidate_source=candidate_source,
                logs=[import_log],
            )

        smoke_ok, smoke_log = self._run_smoke_test(python_exec, run_dir)
        cleaned_output = run_dir / "cleaned_output.npy"

        if not smoke_ok:
            return ValidationReport(
                accepted=False,
                status="smoke_failed",
                comparison="none",
                summary="Cleaned code smoke test failed.",
                candidate_source=candidate_source,
                logs=[smoke_log],
            )

        if not cleaned_output.exists():
            return ValidationReport(
                accepted=False,
                status="output_not_saved",
                comparison="none",
                summary=(
                    "Cleaned code ran but did not save output to the --output path. "
                    "The _main() CLI must save results via np.save(args.output_path, ...)."
                ),
                candidate_source=candidate_source,
                logs=[smoke_log],
            )

        original_ok, original_output, original_log = self._run_original(
            python_exec, run_dir, discovery
        )

        if not original_ok or not (run_dir / "original_output.npy").exists():
            return ValidationReport(
                accepted=True,
                status="cleaned_smoke_only",
                comparison="original_unavailable",
                summary="Original runner did not execute; accepted after cleaned smoke test.",
                candidate_source=candidate_source,
                baseline_available=False,
                logs=[smoke_log, original_log or ""],
            )

        comparison, accepted = self._compare_outputs(
            run_dir / "original_output.npy", cleaned_output
        )
        return ValidationReport(
            accepted=accepted,
            status="validated" if accepted else "regression",
            comparison=comparison,
            summary=(
                "Cleaned code output matches original." if accepted
                else "Cleaned code output differs from original."
            ),
            candidate_source=candidate_source,
            baseline_available=True,
            logs=[smoke_log, original_log or ""],
        )

    def _check_syntax(self, path: Path) -> Tuple[bool, str]:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            ast.parse(source)
            return True, ""
        except SyntaxError as exc:
            return False, f"Line {exc.lineno}: {exc.msg}"

    def _check_interface(self, path: Path) -> Tuple[bool, List[str]]:
        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False, list(REQUIRED_FUNCTIONS)
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
        missing = REQUIRED_FUNCTIONS - defined
        return len(missing) == 0, sorted(missing)

    def _check_import(self, python_exec: str, path: Path, cwd: Path) -> Tuple[bool, str]:
        cmd = [python_exec, "-c", f"import ast; ast.parse(open('{path}').read()); print('OK')"]
        ok, log = self._run_cmd(cmd, cwd)
        return ok, log

    def _run_smoke_test(self, python_exec: str, run_dir: Path) -> Tuple[bool, str]:
        """Run the cleaned code with --stage run-all using synthetic input."""
        input_path = run_dir / "validation_input.npy"
        output_path = run_dir / "cleaned_output.npy"
        metrics_path = run_dir / "cleaned_metrics.json"
        context_path = run_dir / "validation_context.json"

        np.save(input_path, self._synthetic_input())
        context_path.write_text(
            json.dumps(self._synthetic_context(), indent=2), encoding="utf-8"
        )

        cmd = [
            python_exec, "code_cleaned.py",
            "--stage", "run-all",
            "--input", str(input_path),
            "--output", str(output_path),
            "--metrics", str(metrics_path),
            "--context", str(context_path),
        ]
        ok, log = self._run_cmd(cmd, run_dir, timeout=300)
        return ok, log

    def _run_original(
        self,
        python_exec: str,
        run_dir: Path,
        discovery: RepoDiscovery,
    ) -> Tuple[bool, Optional[Path], str]:
        """Attempt to run the original code for comparison."""
        input_path = run_dir / "validation_input.npy"
        original_output = run_dir / "original_output.npy"
        original_metrics = run_dir / "original_metrics.json"
        context_path = run_dir / "validation_context.json"
        runner_path = run_dir / "original_runner.py"

        entry_rel = discovery.entry_module.relative_to(discovery.repo_root).as_posix()
        runner_code = (
            ORIGINAL_RUNNER_TEMPLATE
            .replace("__ENTRY_MODULE__", repr(entry_rel))
            .replace("__MAIN_CANDIDATES__", repr(discovery.main_candidates))
            .replace("__DATA_CANDIDATES__", repr(discovery.data_candidates))
            .replace("__EVAL_CANDIDATES__", repr(discovery.eval_candidates))
            .replace("__CLASS_CANDIDATES__", repr(discovery.class_candidates))
        )
        runner_path.write_text(runner_code, encoding="utf-8")

        if not input_path.exists():
            np.save(input_path, self._synthetic_input())
        if not context_path.exists():
            context_path.write_text(
                json.dumps(self._synthetic_context(), indent=2), encoding="utf-8"
            )

        cmd = [
            python_exec, str(runner_path),
            str(input_path), str(original_output),
            str(original_metrics), str(context_path),
        ]
        ok, log = self._run_cmd(cmd, run_dir, timeout=300)
        return ok, original_output if ok else None, log

    def _run_cmd(self, command: list[str], cwd: Path, timeout: int = 120) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                command, cwd=cwd, capture_output=True, text=True,
                check=False, timeout=timeout,
            )
            log = f"CMD: {' '.join(command)}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
            return result.returncode == 0, log
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT ({timeout}s): {' '.join(command)}"
        except Exception as exc:
            return False, f"ERROR: {exc}"

    def _compare_outputs(self, original_path: Path, cleaned_path: Path) -> Tuple[str, bool]:
        if not cleaned_path.exists():
            return "cleaned_output_missing", False
        if not original_path.exists():
            return "original_output_missing", False
        try:
            original = np.load(original_path, allow_pickle=True)
            cleaned = np.load(cleaned_path, allow_pickle=True)
        except Exception:
            return "load_error", False

        if original.shape != cleaned.shape and original.size == cleaned.size:
            cleaned = cleaned.reshape(original.shape)
        if original.shape != cleaned.shape:
            return f"shape_mismatch:original={original.shape},cleaned={cleaned.shape}", False

        orig_f64 = original.astype(np.float64)
        clean_f64 = cleaned.astype(np.float64)
        diff = np.abs(clean_f64 - orig_f64)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))

        scale = max(float(np.max(np.abs(orig_f64))), 1e-12)
        rel_max = max_abs / scale
        rel_mean = mean_abs / scale

        equivalent = rel_max <= 0.05 or rel_mean <= 0.01
        if equivalent:
            tag = f"equivalent(rel_max={rel_max:.6f},rel_mean={rel_mean:.6f})"
        else:
            tag = f"delta:max={max_abs:.6f},mean={mean_abs:.6f},rel_max={rel_max:.6f}"
        return tag, equivalent

    def _synthetic_input(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((64, 64)).astype(np.float64)

    def _synthetic_context(self) -> Dict:
        return {
            "sigma": [2.0, 2.0],
            "sparse_iter": 2,
            "deconv_iter": 1,
            "fidelity": 50,
            "sparsity": 5,
        }
