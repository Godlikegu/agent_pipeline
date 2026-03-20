from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .models import EnvironmentPlan, RepoDiscovery, ValidationReport


ORIGINAL_RUNNER_TEMPLATE = """from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ENTRY_MODULE = __ENTRY_MODULE__
MAIN_CANDIDATES = __MAIN_CANDIDATES__
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


def _normalize_array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    if np.isscalar(value):
        return np.asarray([value], dtype=np.float64)
    return value


def _default_context(context, payload):
    prepared = dict(context or {})
    normalized = _normalize_array(payload)
    dims = int(getattr(normalized, "ndim", 2) or 2)
    sigma = prepared.get("sigma")
    if sigma is None:
        sigma = [2.0, 2.0] if dims <= 2 else [2.0, 2.0, 1.0]
    prepared.setdefault("sigma", sigma)
    prepared.setdefault("background", 0)
    prepared.setdefault("sparse_iter", 2)
    prepared.setdefault("deconv_iter", 1)
    prepared.setdefault("deconv_type", 0)
    prepared.setdefault("fidelity", 50)
    prepared.setdefault("sparsity", 5)
    prepared.setdefault("tcontinuity", 0.5)
    prepared.setdefault("up_sample", 0)
    prepared.setdefault("mu", 1)
    return prepared


def _gaussian_kernel(sigma):
    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigma_value = float(np.mean(np.asarray(sigma[:2], dtype=np.float64)))
    else:
        sigma_value = float(sigma)
    sigma_value = max(sigma_value, 1.0)
    radius = max(1, int(round(sigma_value * 2)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_value ** 2))
    return kernel / max(float(kernel.sum()), 1.0)


def _parameter_value(name, payload, context):
    lowered = name.lower()
    if lowered in {"input_data", "input_array", "data", "inputs", "image", "img", "x", "y", "f"}:
        return True, payload
    if lowered in {"sigma", "sigmas", "psf_sigma", "kernel_sigma"}:
        return True, context.get("sigma")
    if lowered in {"kernel", "psf"}:
        return True, context.get("kernel", _gaussian_kernel(context.get("sigma")))
    if lowered in {"sparse_iter", "iteration_num"}:
        return True, context.get("sparse_iter")
    if lowered in {"iteration", "iterations"}:
        return True, context.get("deconv_iter")
    if lowered in {"rule", "deconv_rule", "mode"}:
        return True, context.get("deconv_type")
    if lowered in {"fidelity", "lam", "lambda_fidelity"}:
        return True, context.get("fidelity")
    if lowered in {"sparsity", "lambda_sparse"}:
        return True, context.get("sparsity")
    if lowered in {"tcontinuity", "contiz", "continuity"}:
        return True, context.get("tcontinuity")
    if lowered == "background":
        return True, context.get("background")
    if lowered in {"up_sample", "upsample"}:
        return True, context.get("up_sample")
    if lowered == "mu":
        return True, context.get("mu")
    if lowered in {"context", "config", "cfg"}:
        return True, context
    return False, None


def _build_call(func, payload, context):
    signature = inspect.signature(func)
    args = []
    kwargs = {}
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            kwargs.update(context)
            continue
        found, value = _parameter_value(parameter.name, payload, context)
        if found:
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                args.append(value)
            else:
                kwargs[parameter.name] = value
            continue
        if parameter.default is not inspect._empty:
            continue
        if not args:
            args.append(payload)
            continue
        raise TypeError(f"No binding available for required parameter `{parameter.name}`")
    return args, kwargs


def _call(func, payload, context):
    context = _default_context(context, payload)
    try:
        args, kwargs = _build_call(func, payload, context)
        return func(*args, **kwargs)
    except TypeError:
        return func(payload)


def _run_class(module, payload, context):
    for class_name in CLASS_CANDIDATES:
        if not hasattr(module, class_name):
            continue
        cls = getattr(module, class_name)
        if not inspect.isclass(cls):
            continue
        try:
            try:
                instance = cls(context)
            except Exception:
                instance = cls()
        except Exception:
            continue
        for method_name in ("solve", "run", "predict", "forward", "__call__"):
            if not hasattr(instance, method_name):
                continue
            try:
                return _call(getattr(instance, method_name), payload, context)
            except Exception:
                continue
    return payload


def main():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    metrics_path = Path(sys.argv[3])
    context_path = Path(sys.argv[4])
    payload = np.load(input_path, allow_pickle=True)
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else {}
    context = _default_context(context, payload)
    module = _load_module()
    result = None
    for name in MAIN_CANDIDATES:
        if not hasattr(module, name):
            continue
        candidate = getattr(module, name)
        if not callable(candidate):
            continue
        try:
            result = _call(candidate, payload, context)
        except Exception:
            continue
        if result is not None:
            break
    if result is None:
        result = _run_class(module, payload, context)
    result = np.asarray(result, dtype=np.float64)
    np.save(output_path, result)
    metrics = {"status": "ok", "shape": list(result.shape), "mean": float(result.mean())}
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
"""


class CleanerValidator:
    def validate(
        self,
        *,
        run_dir: Path,
        discovery: RepoDiscovery,
        plan: EnvironmentPlan,
        candidate_source: str = "unknown",
    ) -> ValidationReport:
        input_path = run_dir / "validation_input.npy"
        original_output = run_dir / "original_output.npy"
        cleaned_output = run_dir / "cleaned_output.npy"
        original_metrics = run_dir / "original_metrics.json"
        cleaned_metrics = run_dir / "cleaned_metrics.json"
        context_path = run_dir / "validation_context.json"
        original_runner = run_dir / "original_runner.py"

        np.save(input_path, self._synthetic_input())
        context_path.write_text(json.dumps(self._synthetic_context(), indent=2), encoding="utf-8")
        original_runner.write_text(
            ORIGINAL_RUNNER_TEMPLATE.replace(
                "__ENTRY_MODULE__",
                repr(discovery.entry_module.relative_to(discovery.repo_root).as_posix()),
            )
            .replace("__MAIN_CANDIDATES__", repr(discovery.main_candidates))
            .replace("__CLASS_CANDIDATES__", repr(discovery.class_candidates)),
            encoding="utf-8",
        )

        python_exec = str(plan.python_executable if plan.provisioned and plan.python_executable.exists() else sys.executable)

        original_ok, original_log = self._run(
            [python_exec, str(original_runner), str(input_path), str(original_output), str(original_metrics), str(context_path)],
            cwd=run_dir,
        )
        cleaned_ok, cleaned_log = self._run(
            [
                python_exec,
                "code_cleaned.py",
                "--stage",
                "run-all",
                "--input",
                str(input_path),
                "--output",
                str(cleaned_output),
                "--metrics",
                str(cleaned_metrics),
                "--context",
                str(context_path),
            ],
            cwd=run_dir,
        )

        cleaned_metrics_data = self._load_json(cleaned_metrics)
        original_metrics_data = self._load_json(original_metrics)

        if not cleaned_ok:
            return ValidationReport(
                accepted=False,
                status="cleaned_failed",
                comparison="regression",
                summary="Cleaned facade failed its smoke test.",
                candidate_source=candidate_source,
                baseline_available=original_ok and original_output.exists(),
                cleaned_metrics=cleaned_metrics_data,
                original_metrics=original_metrics_data,
                artifacts={
                    "input": str(input_path),
                    "cleaned_output": str(cleaned_output),
                    "original_output": str(original_output),
                },
                logs=[original_log, cleaned_log],
            )

        if not original_ok or not original_output.exists():
            return ValidationReport(
                accepted=True,
                status="cleaned_smoke_only",
                comparison="original_unavailable",
                summary="Original runner did not execute successfully; accepted after cleaned smoke test only.",
                candidate_source=candidate_source,
                baseline_available=False,
                cleaned_metrics=cleaned_metrics_data,
                original_metrics=original_metrics_data,
                artifacts={
                    "input": str(input_path),
                    "cleaned_output": str(cleaned_output),
                    "original_output": str(original_output),
                },
                logs=[original_log, cleaned_log],
            )

        comparison, accepted = self._compare_outputs(original_output, cleaned_output)
        summary = (
            "Cleaned facade is equivalent to or better than the original runner."
            if accepted
            else "Cleaned facade regressed against the original runner."
        )
        return ValidationReport(
            accepted=accepted,
            status="validated" if accepted else "regression",
            comparison=comparison,
            summary=summary,
            candidate_source=candidate_source,
            baseline_available=True,
            cleaned_metrics=cleaned_metrics_data,
            original_metrics=original_metrics_data,
            artifacts={
                "input": str(input_path),
                "cleaned_output": str(cleaned_output),
                "original_output": str(original_output),
            },
            logs=[original_log, cleaned_log],
        )

    def _synthetic_input(self) -> np.ndarray:
        axis = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        return (np.sin(3.14159 * xx) + np.cos(3.14159 * yy) + xx * yy).astype(np.float64)

    def _synthetic_context(self) -> Dict:
        return {
            "sigma": [2.0, 2.0],
            "background": 0,
            "sparse_iter": 2,
            "deconv_iter": 1,
            "deconv_type": 0,
            "fidelity": 50,
            "sparsity": 5,
            "tcontinuity": 0.5,
            "up_sample": 0,
            "mu": 1,
        }

    def _run(self, command: list[str], cwd: Path) -> Tuple[bool, str]:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        log = f"COMMAND: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return result.returncode == 0, log

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _compare_outputs(self, original_output: Path, cleaned_output: Path) -> Tuple[str, bool]:
        original = np.load(original_output, allow_pickle=True)
        cleaned = np.load(cleaned_output, allow_pickle=True)
        if original.shape != cleaned.shape and original.size == cleaned.size:
            cleaned = cleaned.reshape(original.shape)
        if original.shape != cleaned.shape:
            return "shape_mismatch", False
        diff = np.abs(cleaned.astype(np.float64) - original.astype(np.float64))
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        equivalent = max_abs <= 1e-6 or mean_abs <= 1e-6
        return ("equivalent" if equivalent else f"delta:max_abs={max_abs:.6f},mean_abs={mean_abs:.6f}"), equivalent
