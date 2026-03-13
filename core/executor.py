import os
import json
from typing import Any, Callable, Tuple, Optional

from .sandbox import run_cmd


def load_data_shapes(sandbox_dir: str, log_fn: Callable) -> Tuple[Optional[tuple], Optional[tuple]]:
    input_shape = None
    output_shape = None
    try:
        import numpy as np
        input_path = os.path.join(sandbox_dir, "dataset", "input.npy")
        gt_path = os.path.join(sandbox_dir, "dataset", "gt_output.npy")
        if os.path.exists(input_path):
            input_shape = np.load(input_path).shape
            log_fn(f"  ✅ Loaded input shape: {input_shape}")
        else:
            log_fn("  ⚠️ input.npy not found - shape unknown")
        if os.path.exists(gt_path):
            output_shape = np.load(gt_path).shape
            log_fn(f"  ✅ Loaded output shape: {output_shape}")
        else:
            log_fn("  ⚠️ gt_output.npy not found - shape unknown")
    except Exception as e:
        log_fn(f"  ⚠️ Shape loading failed: {e}")
    return input_shape, output_shape


def phase_0_preparation(
    sandbox_dir: str,
    python_path: str,
    task_desc: str,
    package_list: str,
    data_gen_agent: Any,
    eval_gen_agent: Any,
    write_file_fn: Callable,
    log_fn: Callable,
    data_gen_timeout: int = 180,
    gt_code_snippet_limit: int = 4000,
    syntax_check_timeout: int = 30,
) -> Tuple[Optional[tuple], Optional[tuple], dict]:
    """
    Runs Phase 0: data generation + evaluation script preparation.

    Returns:
        (input_shape, output_shape, baseline_metrics)
    """
    log_fn("\n>>> [Phase 0] Initializing Data & Evaluation Scripts...")
    input_path = os.path.join(sandbox_dir, "dataset/input.npy")
    gt_path = os.path.join(sandbox_dir, "dataset/gt_output.npy")
    baseline_path = os.path.join(sandbox_dir, "dataset/baseline.npy")

    # Load GT code snippet ONLY for data generation (Phase 0)
    # NOT stored for Planner/Coder — that would be answer leaking
    gt_code_content = "N/A"
    try:
        gt_folder = os.path.join(sandbox_dir, "gt_code")
        py_files = [f for f in os.listdir(gt_folder) if f.endswith('.py')]
        if py_files:
            with open(os.path.join(gt_folder, py_files[0]), 'r') as f:
                gt_code_content = f.read()[:gt_code_snippet_limit]
    except Exception as e:
        log_fn(f"  ⚠️ Failed to load GT code content: {e}")

    if os.path.exists(input_path) and os.path.exists(gt_path) and os.path.exists(baseline_path):
        log_fn("  [Step 1] Data files found. Skipping Data Generation.")
    else:
        log_fn("  [Step 1] Generating Data Generation Script...")

        # Inject actual GT code content instead of just path
        gen_ctx = {
            'task_desc': task_desc,
            'gt_code_snippet': f"Here is a snippet of the Ground Truth code for reference on data structure:\n```python\n{gt_code_content}\n```",
            'package_list': package_list
        }

        for attempt in range(5):
            code = data_gen_agent.generate(gen_ctx)
            write_file_fn("data_gen.py", code)
            success, out, err = run_cmd(python_path, sandbox_dir, "data_gen.py", timeout=data_gen_timeout)
            input_exists = os.path.exists(os.path.join(sandbox_dir, "dataset/input.npy"))
            gt_exists = os.path.exists(os.path.join(sandbox_dir, "dataset/gt_output.npy"))
            baseline_exists = os.path.exists(os.path.join(sandbox_dir, "dataset/baseline.npy"))
            if success and input_exists and gt_exists and baseline_exists:
                log_fn(f"    ✅ Data generation success (Attempt {attempt+1})")
                break
            else:
                log_fn(f"    ⚠️ Data generation failed (Attempt {attempt+1}). Error: {err}")
                gen_ctx['feedback'] = (f"Missing files. Error: {err}\nIMPORTANT: Do NOT use external files or APIs. Generate synthetic data using numpy/scipy only. Available packages: {package_list[:500]}")
        else:
            raise RuntimeError("❌ Failed to generate valid data_gen.py after 5 retries.")

    input_shape, output_shape = load_data_shapes(sandbox_dir, log_fn)
    log_fn("[Phase 0] Complete.\n")

    log_fn("  [Step 2] Preparing Evaluation Script...")
    eval_script_path = os.path.join(sandbox_dir, "eval_script.py")
    baseline_metrics = {}

    if not os.path.exists(eval_script_path):
        log_fn("  [Step 2] Generating Evaluation Script...")

        # Inject actual shape info instead of file path
        shape_info = f"Input Shape: {input_shape}, Output Shape: {output_shape}" if input_shape else "Unknown Shape"

        eval_ctx = {
            'task_desc': task_desc,
            'data_shape_hint': f"Data Shapes: {shape_info}. The evaluation script must load 'output.npy' and compare it against 'dataset/baseline.npy' or 'dataset/gt_output.npy'."
        }

        for attempt in range(5):
            code = eval_gen_agent.generate(eval_ctx)
            write_file_fn("eval_script.py", code)
            success, out, err = run_cmd(python_path, sandbox_dir, "eval_script.py", args=["dataset/baseline.npy"])
            if success:
                try:
                    metrics = json.loads(out)
                    baseline_metrics = metrics
                    log_fn(f"    ✅ Baseline Metrics: {metrics}")
                    break
                except json.JSONDecodeError:
                    log_fn(f"    ⚠️ Eval script output invalid JSON (Attempt {attempt+1}): {out}")
                    eval_ctx['feedback'] = f"Output invalid JSON. Output: {out}\nError: JSONDecodeError. CRITICAL: Print ONLY a single line of JSON to stdout, nothing else."
            else:
                log_fn(f"    ⚠️ Eval script validation failed (Attempt {attempt+1}). Error: {err}")
                eval_ctx['feedback'] = f"Runtime Error: {err}\nAvailable packages: {package_list[:500]}"
        else:
            raise RuntimeError(f"❌ Eval script validation failed after 5 attempts. Last Error: {err}")
    else:
        log_fn("  [Step 2] Found existing eval_script.py. Skipping generation.")
        success, out, err = run_cmd(python_path, sandbox_dir, "eval_script.py", args=["dataset/baseline.npy"])
        if success:
            try:
                metrics = json.loads(out)
                baseline_metrics = metrics
                log_fn(f"    ✅ Baseline Metrics: {metrics}")
            except json.JSONDecodeError:
                raise RuntimeError(f"❌ Eval script output is not valid JSON. Output: {out}")
        else:
            raise RuntimeError(f"❌ Eval script validation failed. Error: {err}")

    log_fn("[Phase 0] Complete.\n")

    return input_shape, output_shape, baseline_metrics
