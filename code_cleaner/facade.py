from __future__ import annotations

from .models import RepoDiscovery


class FacadeSynthesizer:
    def generate_variants(self, discovery: RepoDiscovery) -> list[str]:
        variants = []
        main_candidates = discovery.main_candidates or ["main"]
        if main_candidates:
            variants.append(
                self.render(
                    discovery,
                    data_candidates=discovery.data_candidates,
                    main_candidates=main_candidates,
                    eval_candidates=discovery.eval_candidates,
                    class_candidates=discovery.class_candidates,
                )
            )
            if len(main_candidates) > 1:
                variants.append(
                    self.render(
                        discovery,
                        data_candidates=discovery.data_candidates,
                        main_candidates=[main_candidates[0]],
                        eval_candidates=discovery.eval_candidates,
                        class_candidates=discovery.class_candidates,
                    )
                )
                variants.append(
                    self.render(
                        discovery,
                        data_candidates=discovery.data_candidates,
                        main_candidates=main_candidates[1:],
                        eval_candidates=discovery.eval_candidates,
                        class_candidates=discovery.class_candidates,
                    )
                )
        return variants or [self.render(discovery, [], ["main"], [], [])]

    def render(
        self,
        discovery: RepoDiscovery,
        *,
        data_candidates: list[str],
        main_candidates: list[str],
        eval_candidates: list[str],
        class_candidates: list[str],
    ) -> str:
        entry_rel = discovery.entry_module.relative_to(discovery.repo_root).as_posix()
        return f'''from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np


ENTRY_MODULE = {entry_rel!r}
DATA_CANDIDATES = {data_candidates!r}
MAIN_CANDIDATES = {main_candidates!r}
EVAL_CANDIDATES = {eval_candidates!r}
CLASS_CANDIDATES = {class_candidates!r}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent / "source_repo"


def _load_original_module():
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_name = ENTRY_MODULE.replace("/", ".").replace("\\\\", ".")
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    if module_name.endswith(".__init__"):
        module_name = module_name[: -len(".__init__")]
    return importlib.import_module(module_name)


def _generate_synthetic_input():
    axis = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return (np.sin(3.14159 * xx) + np.cos(3.14159 * yy) + xx * yy).astype(np.float64)


def _load_artifact(path: str | None):
    if not path:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    if artifact_path.suffix == ".npy":
        return np.load(artifact_path, allow_pickle=True)
    if artifact_path.suffix == ".json":
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    return artifact_path.read_text(encoding="utf-8")


def _normalize_array(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, dict):
        if "array" in value:
            return _normalize_array(value["array"])
        if "prediction" in value:
            return _normalize_array(value["prediction"])
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    if np.isscalar(value):
        return np.asarray([value], dtype=np.float64)
    return value


def _default_context(context, payload):
    prepared = dict(context or {{}})
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
    prepared.setdefault("resolution", 1.0)
    prepared.setdefault("pixelsize", 1.0)
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
    kernel_sum = float(kernel.sum()) or 1.0
    return kernel / kernel_sum


def _parameter_value(name, payload, context):
    lowered = name.lower()
    if lowered in {{"raw_inputs", "raw_input", "input_data", "input_array", "measurement", "measurements", "data", "inputs", "image", "images", "img", "stack", "volume", "x", "y", "f"}}:
        return True, payload
    if lowered in {{"processed_inputs", "processed", "batch"}}:
        return True, payload
    if lowered in {{"context", "config", "cfg", "metadata", "kwargs"}}:
        return True, context
    if lowered in {{"sigma", "sigmas", "psf_sigma", "kernel_sigma"}}:
        return True, context.get("sigma")
    if lowered in {{"kernel", "psf"}}:
        return True, context.get("kernel", _gaussian_kernel(context.get("sigma")))
    if lowered in {{"sparse_iter", "iteration_num"}}:
        return True, context.get("sparse_iter")
    if lowered in {{"iteration", "iterations"}}:
        return True, context.get("deconv_iter")
    if lowered in {{"rule", "deconv_rule", "mode"}}:
        return True, context.get("deconv_type")
    if lowered in {{"fidelity", "lam", "lambda_fidelity"}}:
        return True, context.get("fidelity")
    if lowered in {{"sparsity", "lambda_sparse"}}:
        return True, context.get("sparsity")
    if lowered in {{"tcontinuity", "contiz", "continuity"}}:
        return True, context.get("tcontinuity")
    if lowered == "background":
        return True, context.get("background")
    if lowered in {{"up_sample", "upsample"}}:
        return True, context.get("up_sample")
    if lowered == "mu":
        return True, context.get("mu")
    if lowered == "resolution":
        return True, context.get("resolution")
    if lowered in {{"pixelsize", "pixel_size"}}:
        return True, context.get("pixelsize")
    return False, None


def _build_call(callable_obj, payload, context):
    signature = inspect.signature(callable_obj)
    args = []
    kwargs = {{}}
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
        raise TypeError(f"No binding available for required parameter `{{parameter.name}}`")
    return args, kwargs


def _call_with_optional_context(callable_obj, payload, context):
    context = _default_context(context, payload)
    try:
        args, kwargs = _build_call(callable_obj, payload, context)
        return callable_obj(*args, **kwargs)
    except TypeError:
        pass
    positional_attempts = [
        (payload, context),
        (payload,),
        tuple(),
    ]
    for args in positional_attempts:
        try:
            return callable_obj(*args)
        except TypeError:
            continue
    raise


def data_process(raw_inputs=None, context=None) -> dict:
    base_context = dict(context or {{}})
    if raw_inputs is None:
        raw_inputs = _load_artifact(base_context.get("input_path"))
        if raw_inputs is None:
            raw_inputs = _generate_synthetic_input()
    context = _default_context(base_context, raw_inputs)
    module = _load_original_module()
    processed = {{
        "raw_inputs": raw_inputs,
        "input": _normalize_array(raw_inputs),
        "context": context,
        "reference": _load_artifact(context.get("reference_path")),
        "baseline": _load_artifact(context.get("baseline_path")),
    }}
    for name in DATA_CANDIDATES:
        if not hasattr(module, name):
            continue
        candidate = getattr(module, name)
        if not callable(candidate):
            continue
        try:
            result = _call_with_optional_context(candidate, processed["input"], context)
        except Exception:
            continue
        if result is None:
            continue
        if isinstance(result, dict):
            processed.update(result)
            if "input" in result:
                processed["input"] = _normalize_array(result["input"])
        else:
            processed["prepared"] = result
            if processed["input"] is None:
                processed["input"] = _normalize_array(result)
        break
    processed.setdefault("prepared", processed.get("input"))
    return processed


def _run_class_candidate(module, payload, context):
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
                return _call_with_optional_context(getattr(instance, method_name), payload, context)
            except Exception:
                continue
    return None


def main_process(processed_inputs, context=None):
    context = _default_context(context, processed_inputs)
    module = _load_original_module()
    payload = processed_inputs
    if isinstance(processed_inputs, dict):
        payload = processed_inputs.get("prepared", processed_inputs.get("input", processed_inputs))
        context = _default_context(processed_inputs.get("context", context), payload)

    for name in MAIN_CANDIDATES:
        if not hasattr(module, name):
            continue
        candidate = getattr(module, name)
        if not callable(candidate):
            continue
        try:
            result = _call_with_optional_context(candidate, payload, context)
        except Exception:
            continue
        if result is not None:
            return _normalize_array(result)

    class_result = _run_class_candidate(module, payload, context)
    if class_result is not None:
        return _normalize_array(class_result)
    return _normalize_array(payload)


def _compute_metrics(prediction, reference):
    pred = _normalize_array(prediction)
    ref = _normalize_array(reference)
    if pred is None or ref is None:
        return {{"status": "missing_reference"}}
    if isinstance(pred, np.ndarray) and isinstance(ref, np.ndarray):
        if pred.shape != ref.shape and pred.size == ref.size:
            pred = pred.reshape(ref.shape)
        if pred.shape != ref.shape:
            return {{
                "status": "shape_mismatch",
                "prediction_shape": list(pred.shape),
                "reference_shape": list(ref.shape),
            }}
        diff = pred.astype(np.float64) - ref.astype(np.float64)
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))
        max_val = max(float(np.max(np.abs(ref))), 1e-8)
        psnr = float(10.0 * np.log10((max_val ** 2) / max(mse, 1e-12)))
        return {{
            "status": "ok",
            "mse": mse,
            "mae": mae,
            "max_abs": max_abs,
            "psnr": psnr,
        }}
    return {{"status": "non_numeric"}}


def eval(processed_inputs, prediction, context=None) -> dict:
    context = _default_context(context, prediction)
    module = _load_original_module()
    for name in EVAL_CANDIDATES:
        if not hasattr(module, name):
            continue
        candidate = getattr(module, name)
        if not callable(candidate):
            continue
        try:
            result = _call_with_optional_context(candidate, prediction, context)
        except Exception:
            continue
        if isinstance(result, dict):
            return result
    reference = None
    if isinstance(processed_inputs, dict):
        reference = processed_inputs.get("reference")
        if reference is None:
            reference = processed_inputs.get("baseline")
        if reference is None:
            reference = processed_inputs.get("input")
    return _compute_metrics(prediction, reference)


def _save_outputs(prediction, output_path: str | None, metrics: dict, metrics_path: str | None):
    if output_path:
        np.save(output_path, _normalize_array(prediction))
    if metrics_path:
        Path(metrics_path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_context(path: str | None):
    if not path:
        return {{}}
    payload = _load_artifact(path)
    return payload if isinstance(payload, dict) else {{}}


def _main():
    parser = argparse.ArgumentParser(description="Unified cleaned facade for the source repository.")
    parser.add_argument("--stage", default="run-all", choices=["data-process", "main", "eval", "run-all"])
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument("--metrics", dest="metrics_path")
    parser.add_argument("--context", dest="context_path")
    args = parser.parse_args()

    context = _load_context(args.context_path)
    if args.input_path:
        context["input_path"] = args.input_path

    processed = data_process(context=context)
    if args.stage == "data-process":
        print(json.dumps({{"keys": sorted(processed.keys())}}, ensure_ascii=False))
        return

    prediction = main_process(processed, context=context)
    if args.stage == "main":
        _save_outputs(prediction, args.output_path, {{"status": "main_only"}}, args.metrics_path)
        return

    metrics = eval(processed, prediction, context=context)
    _save_outputs(prediction, args.output_path, metrics, args.metrics_path)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    _main()
'''
