"""
Test harness generation for code_cleaner.

Generates the test data (input.npy, gt_output.npy, baseline.npy),
evaluation script (eval_script.py), and optionally a data generation script (data_gen.py).

These artifacts serve two purposes:
  1. Verifying that cleaned code produces correct results
  2. Providing the sandbox environment for the downstream pipeline
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def setup_sandbox(
    sandbox_dir: str,
    reference_code_path: str,
    log_fn: Callable = print,
) -> None:
    """
    Initialise a sandbox directory with reference code and dataset folder.
    Handles both single-file and directory reference code.
    """
    import shutil

    log_fn(f"[Sandbox] Initializing at {sandbox_dir} ...")
    os.makedirs(sandbox_dir, exist_ok=True)

    dest = os.path.join(sandbox_dir, "gt_code")
    if os.path.exists(dest):
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        else:
            os.remove(dest)

    ref = Path(reference_code_path).expanduser().resolve()
    if ref.is_dir():
        shutil.copytree(str(ref), dest)
    elif ref.is_file():
        os.makedirs(dest, exist_ok=True)
        shutil.copy(str(ref), dest)
    else:
        raise FileNotFoundError(f"Reference code not found: {ref}")

    os.makedirs(os.path.join(sandbox_dir, "dataset"), exist_ok=True)
    log_fn("[Sandbox] Done.")


def run_cmd(
    python_path: str,
    sandbox_dir: str,
    script_name: str,
    args: List[str] | None = None,
    timeout: int = 600,
    check_syntax_only: bool = False,
    syntax_check_timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Execute a Python script inside the sandbox."""
    if check_syntax_only:
        cmd = [python_path, "-m", "py_compile", script_name]
        timeout = syntax_check_timeout
    else:
        cmd = [python_path, script_name] + (args or [])
    try:
        result = subprocess.run(cmd, cwd=sandbox_dir, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT EXPIRED"


def load_data_shapes(
    sandbox_dir: str,
    log_fn: Callable = print,
) -> Tuple[Optional[tuple], Optional[tuple]]:
    """Load input and output shapes from the sandbox dataset."""
    import numpy as np

    input_shape = output_shape = None
    input_path = os.path.join(sandbox_dir, "dataset", "input.npy")
    gt_path = os.path.join(sandbox_dir, "dataset", "gt_output.npy")
    try:
        if os.path.exists(input_path):
            input_shape = tuple(np.load(input_path).shape)
            log_fn(f"  Input shape: {input_shape}")
        if os.path.exists(gt_path):
            output_shape = tuple(np.load(gt_path).shape)
            log_fn(f"  Output shape: {output_shape}")
    except Exception as e:
        log_fn(f"  Shape loading failed: {e}")
    return input_shape, output_shape


def generate_test_harness(
    *,
    sandbox_dir: str,
    python_path: str,
    task_desc: str,
    data_gen_agent: Any,
    eval_gen_agent: Any,
    write_file_fn: Callable,
    log_fn: Callable = print,
    data_gen_timeout: int = 180,
    gt_code_snippet_limit: int = 4000,
    syntax_check_timeout: int = 30,
    max_retries: int = 5,
    package_list: str = "",
) -> Tuple[Optional[tuple], Optional[tuple], Dict]:
    """
    Generate test data, eval script and baseline metrics.

    Returns (input_shape, output_shape, baseline_metrics).
    """
    log_fn("[TestHarness] Generating test data and evaluation scripts ...")
    input_path = os.path.join(sandbox_dir, "dataset", "input.npy")
    gt_path = os.path.join(sandbox_dir, "dataset", "gt_output.npy")
    baseline_path = os.path.join(sandbox_dir, "dataset", "baseline.npy")

    gt_code_content = _load_gt_snippet(sandbox_dir, gt_code_snippet_limit, log_fn)

    if os.path.exists(input_path) and os.path.exists(gt_path) and os.path.exists(baseline_path):
        log_fn("  [Step 1] Data files already exist — skipping generation.")
    else:
        log_fn("  [Step 1] Generating data via agent ...")
        gen_ctx: dict = {
            "task_desc": task_desc,
            "gt_code_snippet": (
                f"Reference code snippet for data structure:\n```python\n{gt_code_content}\n```"
            ),
            "package_list": package_list,
        }
        for attempt in range(max_retries):
            code = data_gen_agent.generate(gen_ctx)
            write_file_fn("data_gen.py", code)
            ok, _, err = run_cmd(python_path, sandbox_dir, "data_gen.py", timeout=data_gen_timeout)
            if ok and all(os.path.exists(os.path.join(sandbox_dir, "dataset", f)) for f in ["input.npy", "gt_output.npy", "baseline.npy"]):
                log_fn(f"    Data generation succeeded (attempt {attempt + 1})")
                break
            log_fn(f"    Data generation failed (attempt {attempt + 1}): {err}")
            gen_ctx["feedback"] = (
                f"Missing files or error: {err}\n"
                f"Generate synthetic data using numpy/scipy only. Packages: {package_list[:500]}"
            )
        else:
            raise RuntimeError("Failed to generate test data after max retries.")

    input_shape, output_shape = load_data_shapes(sandbox_dir, log_fn)

    eval_script_path = os.path.join(sandbox_dir, "eval_script.py")
    baseline_metrics: Dict = {}

    if not os.path.exists(eval_script_path):
        log_fn("  [Step 2] Generating evaluation script ...")
        shape_info = f"Input: {input_shape}, Output: {output_shape}" if input_shape else "Unknown"
        eval_ctx: dict = {
            "task_desc": task_desc,
            "data_shape_hint": (
                f"Data Shapes: {shape_info}. "
                "Eval script must load 'output.npy' and compare against "
                "'dataset/baseline.npy' or 'dataset/gt_output.npy'."
            ),
        }
        for attempt in range(max_retries):
            code = eval_gen_agent.generate(eval_ctx)
            write_file_fn("eval_script.py", code)
            ok, out, err = run_cmd(python_path, sandbox_dir, "eval_script.py", args=["dataset/baseline.npy"])
            if ok:
                try:
                    baseline_metrics = json.loads(out)
                    log_fn(f"    Baseline metrics: {baseline_metrics}")
                    break
                except json.JSONDecodeError:
                    eval_ctx["feedback"] = (
                        f"Output invalid JSON: {out}\n"
                        "Print ONLY a single JSON line to stdout."
                    )
            else:
                eval_ctx["feedback"] = f"Runtime error: {err}\nPackages: {package_list[:500]}"
                log_fn(f"    Eval script failed (attempt {attempt + 1}): {err}")
        else:
            raise RuntimeError("Eval script validation failed after max retries.")
    else:
        log_fn("  [Step 2] Existing eval_script.py found.")
        ok, out, err = run_cmd(python_path, sandbox_dir, "eval_script.py", args=["dataset/baseline.npy"])
        if ok:
            try:
                baseline_metrics = json.loads(out)
                log_fn(f"    Baseline metrics: {baseline_metrics}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Eval script output not valid JSON: {out}")
        else:
            raise RuntimeError(f"Eval script failed: {err}")

    log_fn("[TestHarness] Complete.")
    return input_shape, output_shape, baseline_metrics


def reset_sandbox(sandbox_dir: str, log_fn: Callable = print) -> None:
    """Reset sandbox to post-test-harness state (keep dataset + eval + data_gen)."""
    import shutil

    preserve_dirs = {"dataset", "gt_code"}
    preserve_files = {"eval_script.py", "data_gen.py"}
    log_fn("[Sandbox] Resetting to test-harness state ...")
    for item in os.listdir(sandbox_dir):
        item_path = os.path.join(sandbox_dir, item)
        if (os.path.isdir(item_path) and item in preserve_dirs) or \
           (os.path.isfile(item_path) and item in preserve_files):
            continue
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            log_fn(f"  Warning: failed to remove {item}: {e}")
    log_fn("  Sandbox reset complete.")


def _load_gt_snippet(sandbox_dir: str, limit: int, log_fn: Callable) -> str:
    try:
        gt_folder = os.path.join(sandbox_dir, "gt_code")
        py_files = [f for f in os.listdir(gt_folder) if f.endswith(".py")]
        if py_files:
            with open(os.path.join(gt_folder, py_files[0]), "r") as f:
                return f.read()[:limit]
    except Exception as e:
        log_fn(f"  Warning: failed to load GT code snippet: {e}")
    return "N/A"
