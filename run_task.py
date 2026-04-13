"""
run_task.py — Main pipeline entry point.

Runs one or more tasks from a YAML task list.
Supports structured tasks under data/tasks/ with sandbox construction.

Usage:
    python -m run_task --task-config config/tasks/auto_tasks.yaml --model cds/Claude-4.6-opus
    python -m run_task --task-filter sim,bpm
"""
import os
import sys
import yaml
import json
import time
import shutil
import traceback
import argparse
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.llm_client import create_client
from utils.task_description_utils import (
    load_or_generate_task_description,
    resolve_reference_code_path,
)
from core.workflow import PipelineWorkflow
from skills import create_skill_manager
from utils.reporter import ExecutionReporter


def _parse_eval_agent_response(response: str) -> tuple:
    """Parse EvalGenAgent response into (output_keys_dict, eval_code_str).

    The EvalGenAgent outputs:
      ```json:output_keys
      { "keys": {...}, "save_instruction": "..." }
      ```
      ```python
      # eval_script.py code
      ```

    Returns (output_keys_dict_or_None, eval_python_code_str).
    """
    import re as _re
    from utils.text_utils import extract_python

    # Extract ```json:output_keys ... ``` block
    output_keys = None
    m = _re.search(r'```json:output_keys\s*\n(.*?)\n\s*```', response, _re.DOTALL)
    if m:
        try:
            output_keys = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # If no tagged block, try to find any JSON with "keys" and "save_instruction"
    if output_keys is None:
        m = _re.search(r'```json\s*\n(\{.*?"keys".*?\})\s*\n```', response, _re.DOTALL)
        if m:
            try:
                candidate = json.loads(m.group(1))
                if "keys" in candidate:
                    output_keys = candidate
            except json.JSONDecodeError:
                pass

    # Extract eval_script code
    eval_code = extract_python(response)

    return output_keys, eval_code


def _extract_notebook_eval_cells(task_dir: str, max_chars: int = 4000) -> str:
    """Extract evaluation/metric-related cells from the task's notebook."""
    notebooks_dir = os.path.join(task_dir, "notebooks")
    if not os.path.isdir(notebooks_dir):
        return ""

    METRIC_KEYWORDS = (
        "ncc", "nrmse", "metric", "evaluate", "eval_", "boundary",
        "ncc_fibers", "ncc_cysts", "velocity_nrmse", "snr", "cnr",
        "fwhm", "sharpness", "lateral", "axial", "position_angle",
        "noise_reduction", "mae",
    )

    collected = []
    total_len = 0

    for fname in sorted(os.listdir(notebooks_dir)):
        if not fname.endswith(".ipynb"):
            continue
        nb_path = os.path.join(notebooks_dir, fname)
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                source_lower = source.lower()
                if any(kw in source_lower for kw in METRIC_KEYWORDS):
                    chunk = f"# --- from {fname} ---\n{source.strip()}\n"
                    if total_len + len(chunk) > max_chars:
                        break
                    collected.append(chunk)
                    total_len += len(chunk)
        except Exception:
            pass

    return "\n".join(collected)


def _extract_src_metric_functions(task_dir: str, max_chars: int = 3000) -> str:
    """Extract metric-related functions from src/visualization.py and src/solvers.py."""
    METRIC_KEYWORDS = (
        "metric", "ncc", "nrmse", "evaluate", "measure", "compute_",
        "fwhm", "cnr", "snr", "sharpness", "lateral", "axial",
        "noise_reduction", "mae", "position_angle",
    )

    collected = []
    total_len = 0

    for src_file in ["src/visualization.py", "src/solvers.py"]:
        fpath = os.path.join(task_dir, src_file)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract individual functions that contain metric keywords
            import re
            func_pattern = re.compile(
                r'^(def\s+\w+\s*\([^)]*\).*?(?=\ndef\s|\Z))',
                re.MULTILINE | re.DOTALL
            )
            for match in func_pattern.finditer(content):
                func_text = match.group(1)
                func_lower = func_text.lower()
                if any(kw in func_lower for kw in METRIC_KEYWORDS):
                    chunk = f"# --- from {src_file} ---\n{func_text.strip()}\n"
                    if total_len + len(chunk) > max_chars:
                        break
                    collected.append(chunk)
                    total_len += len(chunk)
        except Exception:
            pass

    return "\n".join(collected)


def _extract_eval_context(task_dir: str, sandbox_dir: str) -> dict:
    """Extract evaluation-relevant context from the task's existing codebase.

    Returns a dict with keys: metrics_json, gt_structure, notebook_eval_code,
    src_metric_functions, required_output_keys, boundary_definitions.
    """
    context = {}

    # 1. Full metrics.json
    metrics_path = os.path.join(task_dir, "evaluation", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            context["metrics_json"] = json.load(f)

    # 2. GT structure from gt_keys.json (generated by setup_task_sandbox)
    gt_keys_path = os.path.join(sandbox_dir, "data", "gt_keys.json")
    if os.path.exists(gt_keys_path):
        with open(gt_keys_path, "r", encoding="utf-8") as f:
            context["gt_structure"] = json.load(f)

    # 3. Extract metric computation code from notebook
    notebook_code = _extract_notebook_eval_cells(task_dir)
    if notebook_code:
        context["notebook_eval_code"] = notebook_code

    # 4. Extract metric functions from src/
    src_metrics = _extract_src_metric_functions(task_dir)
    if src_metrics:
        context["src_metric_functions"] = src_metrics

    # 5. Required output keys and boundary definitions (derived from metrics.json)
    if "metrics_json" in context:
        boundaries = {}
        for k, v in context["metrics_json"].items():
            if k.endswith("_boundary") or k.endswith("_boundary_deg") or k.endswith("_boundary_nm"):
                boundaries[k] = v
        # Derive metric key names from boundary keys
        required_keys = []
        boundary_to_metric = {}  # explicit mapping for EvalGenAgent
        for bk in boundaries:
            mk = bk.replace("_boundary_deg", "_deg").replace("_boundary_nm", "_nm").replace("_boundary", "")
            required_keys.append(mk)
            boundary_to_metric[bk] = mk
        context["required_output_keys"] = required_keys
        context["boundary_definitions"] = boundaries
        context["boundary_to_metric_map"] = boundary_to_metric

    return context


def _extract_viz_context(task_dir: str, max_chars: int = 5000) -> dict:
    """Extract visualization-relevant context from the task's notebook and src."""
    context = {}

    VIZ_KEYWORDS = (
        "plt.", "imshow", "plot", "figure", "subplot", "colorbar",
        "savefig", "visualization", "display", "comparison",
    )

    # 1. Extract visualization cells from notebook
    notebooks_dir = os.path.join(task_dir, "notebooks")
    if os.path.isdir(notebooks_dir):
        collected = []
        total_len = 0
        for fname in sorted(os.listdir(notebooks_dir)):
            if not fname.endswith(".ipynb"):
                continue
            nb_path = os.path.join(notebooks_dir, fname)
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    nb = json.load(f)
                for cell in nb.get("cells", []):
                    if cell.get("cell_type") != "code":
                        continue
                    source = "".join(cell.get("source", []))
                    source_lower = source.lower()
                    if any(kw in source_lower for kw in VIZ_KEYWORDS):
                        chunk = f"# --- from {fname} ---\n{source.strip()}\n"
                        if total_len + len(chunk) > max_chars:
                            break
                        collected.append(chunk)
                        total_len += len(chunk)
            except Exception:
                pass
        if collected:
            context["notebook_viz_code"] = "\n".join(collected)

    # 2. Read src/visualization.py (key functions)
    viz_path = os.path.join(task_dir, "src", "visualization.py")
    if os.path.isfile(viz_path):
        try:
            with open(viz_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Truncate if too long
            if len(content) > max_chars:
                content = content[:max_chars] + "\n# ... (truncated)"
            context["src_visualization"] = content
        except Exception:
            pass

    return context


def setup_task_sandbox(task_dir: str, sandbox_dir: str) -> dict:
    """Build sandbox from a structured task directory.

    Creates sandbox_dir/data/ that mirrors task_dir/data/ (excluding GT files),
    plus gt_keys.json (key metadata). GT files are placed at sandbox root
    to prevent solver from accessing them (solver can only read data/).

    Returns loaded meta_data dict (or empty dict if not found).
    """
    import numpy as np

    task_dir = os.path.abspath(task_dir)
    sandbox_dir = os.path.abspath(sandbox_dir)
    data_dst = os.path.join(sandbox_dir, "data")
    os.makedirs(data_dst, exist_ok=True)

    data_src = os.path.join(task_dir, "data")
    meta_data = {}
    data_info = {}

    # GT file names that must NOT be in data/ (solver isolation)
    GT_FILENAMES = {"ground_truth.npz", "baseline_reference.npz", "ground_truth.npy"}

    # --- 1. Copy files from task data/ to sandbox data/ ---
    #     GT files go to sandbox ROOT instead of data/
    if os.path.isdir(data_src):
        for item in sorted(os.listdir(data_src)):
            src = os.path.join(data_src, item)
            if item in GT_FILENAMES:
                # GT files → sandbox root (not data/)
                dst = os.path.join(sandbox_dir, item)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  Copied {item} -> sandbox root (solver-isolated)")
            elif os.path.isdir(src):
                dst = os.path.join(data_dst, item)
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    print(f"  Copied dir {item}/ -> data/{item}/")
            else:
                dst = os.path.join(data_dst, item)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  Copied {item} -> data/{item}")

    # --- 1b. Remove any leftover GT files from data/ (from previous runs) ---
    for gt_name in GT_FILENAMES:
        gt_in_data = os.path.join(data_dst, gt_name)
        if os.path.exists(gt_in_data):
            os.remove(gt_in_data)
            print(f"  Removed leftover {gt_name} from data/ (solver isolation)")
    # Also remove legacy gt_output.npy from data/
    legacy_gt = os.path.join(data_dst, "gt_output.npy")
    if os.path.exists(legacy_gt):
        os.remove(legacy_gt)
        print(f"  Removed leftover gt_output.npy from data/ (solver isolation)")

    # --- 2. Generate gt_keys.json from ground_truth/baseline (in sandbox root) ---
    gt_keys_path = os.path.join(data_dst, "gt_keys.json")
    if not os.path.exists(gt_keys_path):
        for npz_name in ["ground_truth.npz", "baseline_reference.npz"]:
            npz_path = os.path.join(sandbox_dir, npz_name)
            if os.path.exists(npz_path):
                npz = np.load(npz_path)
                keys = list(npz.keys())
                if keys:
                    gt_keys_info = {
                        "source": npz_name,
                        "keys": {k: {"shape": list(npz[k].shape), "dtype": str(npz[k].dtype)} for k in keys}
                    }
                    with open(gt_keys_path, "w", encoding="utf-8") as f:
                        json.dump(gt_keys_info, f, indent=2)
                    print(f"  Generated gt_keys.json ({len(keys)} keys: {keys})")
                break

    # --- 2b. Copy evaluation/metrics.json to sandbox data/metrics.json ---
    metrics_src = os.path.join(task_dir, "evaluation", "metrics.json")
    metrics_dst = os.path.join(data_dst, "metrics.json")
    if os.path.isfile(metrics_src) and not os.path.exists(metrics_dst):
        shutil.copy2(metrics_src, metrics_dst)
        print(f"  Copied evaluation/metrics.json -> data/metrics.json")

    # --- 3. Load meta_data.json ---
    meta_path = os.path.join(data_dst, "meta_data.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

    # --- 4. Generate data_info.json: npz keys/shapes/dtypes ---
    for fname in sorted(os.listdir(data_dst)):
        fpath = os.path.join(data_dst, fname)
        if fname.endswith(".npz") and os.path.isfile(fpath):
            try:
                npz = np.load(fpath)
                file_info = {}
                for k in npz.keys():
                    arr = npz[k]
                    file_info[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                data_info[fname] = file_info
            except Exception:
                pass

    if data_info:
        info_path = os.path.join(data_dst, "data_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=2)
        print(f"  Generated data/data_info.json")

    return meta_data


def run_single_task(
    task_info: dict,
    client: OpenAI,
    model_name: str,
    config: dict,
    skill_manager=None,
    reporter: Optional[ExecutionReporter] = None,
) -> Dict[str, any]:
    """Execute a single task and return structured result."""
    task_name = task_info["name"]
    start_time = time.time()
    paths_cfg = config.get("paths", {})

    sandbox_dir = task_info.get("sandbox_dir")
    if not sandbox_dir:
        sandbox_root = paths_cfg.get("sandbox_root", "./test_sandbox")
        sandbox_dir = os.path.join(sandbox_root, f"{task_name}")
    sandbox_dir = os.path.abspath(sandbox_dir)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    print(f"  Sandbox      : {sandbox_dir}")
    print(f"  Python Path  : {task_info.get('python_path', 'default')}")

    try:
        # --- Phase 1: Setup sandbox from task directory ---
        meta_data = {}
        task_dir = task_info.get("task_dir")
        if task_dir:
            task_dir = os.path.abspath(task_dir)
            print(f"  Task Dir     : {task_dir}")
            meta_data = setup_task_sandbox(task_dir, sandbox_dir)
        else:
            # Legacy path: sandbox already configured externally
            reference_code_path = None
            try:
                reference_code_path = str(resolve_reference_code_path(task_info))
            except (ValueError, FileNotFoundError):
                pass
            if reference_code_path:
                print(f"  Reference Code: {reference_code_path}")

        # --- Phase 2: Load or generate task description ---
        task_description = load_or_generate_task_description(
            task_info=task_info,
            config=config,
            client=client,
            model_name=model_name,
            meta_data=meta_data if meta_data else None,
        )

        # --- Phase 2.5: Load per-task evaluation thresholds ---
        eval_thresholds = {}
        eval_boundaries = {}  # All *_boundary keys for flexible threshold checking
        if task_dir:
            metrics_json_path = os.path.join(task_dir, "evaluation", "metrics.json")
            if os.path.exists(metrics_json_path):
                with open(metrics_json_path, "r", encoding="utf-8") as f:
                    task_metrics = json.load(f)
                # Legacy: extract min_ncc/max_nrmse for backward compat
                if "ncc_boundary" in task_metrics:
                    eval_thresholds["min_ncc"] = task_metrics["ncc_boundary"]
                if "nrmse_boundary" in task_metrics:
                    eval_thresholds["max_nrmse"] = task_metrics["nrmse_boundary"]
                # Collect ALL boundary keys for flexible evaluation
                eval_boundaries = {k: v for k, v in task_metrics.items()
                                   if k.endswith("_boundary") or k.endswith("_boundary_deg")
                                   or k.endswith("_boundary_nm")}
                eval_thresholds["eval_boundaries"] = eval_boundaries
                print(f"  Eval boundaries: {eval_boundaries}")

        # Append sandbox data layout to task description so all agents know actual paths
        data_dir = os.path.join(sandbox_dir, "data")
        layout_lines = []
        # List .npz files with key/shape/dtype info
        data_info_path = os.path.join(data_dir, "data_info.json")
        if os.path.exists(data_info_path):
            with open(data_info_path, "r", encoding="utf-8") as f:
                data_info = json.load(f)
            for fname, info in data_info.items():
                if fname == "gt_output.npy":
                    continue  # Don't expose GT to solver agents
                if isinstance(info, dict) and "shape" in info:
                    layout_lines.append(f"  - data/{fname} shape={info['shape']} dtype={info['dtype']}")
                else:
                    for key, kinfo in info.items():
                        layout_lines.append(
                            f"  - data/{fname} key='{key}' shape={kinfo['shape']} dtype={kinfo['dtype']}"
                        )
        elif os.path.isdir(data_dir):
            # Fallback: scan data directory
            for fname in sorted(os.listdir(data_dir)):
                fpath = os.path.join(data_dir, fname)
                if os.path.isfile(fpath) and fname not in (
                    "gt_output.npy", "gt_key.txt", "gt_keys.json", "data_info.json",
                    "ground_truth.npz", "baseline_reference.npz", "ground_truth.npy",
                ):
                    layout_lines.append(f"  - data/{fname}")
        if os.path.exists(os.path.join(data_dir, "meta_data.json")):
            layout_lines.append("  - data/meta_data.json (physical parameters)")
        if layout_lines:
            layout_section = (
                "\n\n### SANDBOX DATA LAYOUT (CRITICAL - use ONLY these paths)\n"
                "Available files in the working directory:\n"
                + "\n".join(layout_lines)
                + "\nDo NOT use any other input paths — only the files listed above exist."
                "\nFor .npz files: use `data = np.load('data/<file>.npz'); arr = data['<key>']` to load."
            )
            task_description += layout_section

        # --- Phase 3: Generate eval_script.py and visualize_output.py if missing ---
        eval_script_path = os.path.join(sandbox_dir, "eval_script.py")
        if not os.path.exists(eval_script_path):
            from agents.sandbox_agents import EvalGenAgent, VizGenAgent, get_installed_libraries
            from utils.text_utils import extract_python

            python_path = task_info.get("python_path", sys.executable)
            eval_gen_agent = EvalGenAgent(client, model_name)
            package_list = get_installed_libraries(python_path)

            # Detect data shape for hint
            import numpy as np
            data_shape_hint = "N/A"
            gt_path = os.path.join(sandbox_dir, "data", "gt_output.npy")
            if os.path.exists(gt_path):
                gt = np.load(gt_path, allow_pickle=True)
                data_shape_hint = f"shape={gt.shape}, dtype={gt.dtype}"

            # Extract rich evaluation context from task codebase
            eval_extra = {}
            if task_dir:
                eval_extra = _extract_eval_context(task_dir, sandbox_dir)

            eval_ctx = {
                "task_desc": task_description,
                "data_shape_hint": data_shape_hint,
                "package_list": package_list,
                "meta_data": meta_data,
                "eval_thresholds": eval_thresholds,
                **eval_extra,
            }
            print("  Generating eval_script.py + output_keys.json...")
            eval_response = eval_gen_agent.generate(eval_ctx)

            # Parse dual output: output_keys.json + eval_script.py
            output_keys, eval_code = _parse_eval_agent_response(eval_response)

            if output_keys and "keys" in output_keys:
                output_keys_path = os.path.join(sandbox_dir, "data", "output_keys.json")
                with open(output_keys_path, "w", encoding="utf-8") as f:
                    json.dump(output_keys, f, indent=2)
                n_keys = len(output_keys["keys"])
                print(f"  output_keys.json generated ({n_keys} keys: {list(output_keys['keys'].keys())})")
            else:
                # Fallback: derive from gt_keys.json if available
                gt_keys_path = os.path.join(sandbox_dir, "data", "gt_keys.json")
                if os.path.exists(gt_keys_path):
                    with open(gt_keys_path, "r", encoding="utf-8") as f:
                        gt_keys = json.load(f)
                    gt_key_info = gt_keys.get("keys", {})
                    fallback_keys = {
                        k: {"shape": v["shape"], "dtype": v["dtype"]}
                        for k, v in gt_key_info.items()
                    }
                    if len(fallback_keys) == 1:
                        key_name = list(fallback_keys.keys())[0]
                        save_instr = f"np.savez('output.npz', {key_name}=result)"
                    else:
                        args = ", ".join(f"{k}={k}_arr" for k in fallback_keys)
                        save_instr = f"np.savez('output.npz', {args})"
                    output_keys = {"keys": fallback_keys, "save_instruction": save_instr}
                    output_keys_path = os.path.join(sandbox_dir, "data", "output_keys.json")
                    with open(output_keys_path, "w", encoding="utf-8") as f:
                        json.dump(output_keys, f, indent=2)
                    print(f"  output_keys.json generated (fallback from gt_keys: {list(fallback_keys.keys())})")
                else:
                    # No GT keys available — use generic single-key format
                    output_keys = {"keys": {"output": {}}, "save_instruction": "np.savez('output.npz', output=result)"}
                    output_keys_path = os.path.join(sandbox_dir, "data", "output_keys.json")
                    with open(output_keys_path, "w", encoding="utf-8") as f:
                        json.dump(output_keys, f, indent=2)
                    print("  output_keys.json generated (generic fallback)")

            with open(eval_script_path, "w", encoding="utf-8") as f:
                f.write(eval_code)
            print("  eval_script.py generated.")

            # Generate task-specific visualization script
            viz_script_path = os.path.join(sandbox_dir, "visualize_output.py")
            if not os.path.exists(viz_script_path) and task_dir:
                viz_extra = _extract_viz_context(task_dir)
                gt_keys_path = os.path.join(sandbox_dir, "data", "gt_keys.json")
                gt_structure = None
                if os.path.exists(gt_keys_path):
                    with open(gt_keys_path, "r", encoding="utf-8") as f:
                        gt_structure = json.load(f)
                viz_ctx = {
                    "task_desc": task_description,
                    "data_shape_hint": data_shape_hint,
                    "package_list": package_list,
                    "gt_structure": gt_structure,
                    "output_keys": output_keys,
                    **viz_extra,
                }
                print("  Generating visualize_output.py...")
                viz_gen_agent = VizGenAgent(client, model_name)
                viz_code = viz_gen_agent.generate(viz_ctx)
                viz_code = extract_python(viz_code)
                with open(viz_script_path, "w", encoding="utf-8") as f:
                    f.write(viz_code)
                print("  visualize_output.py generated.")

        # --- Phase 4: Run workflow ---
        workflow = PipelineWorkflow(
            task_name=task_name,
            task_desc=task_description,
            sandbox_dir=sandbox_dir,
            python_path=task_info.get("python_path", sys.executable),
            client=client,
            model_name=model_name,
            config=config,
            skill_manager=skill_manager,
            eval_thresholds=eval_thresholds,
            task_dir=task_dir,
        )

        success = workflow.run()
        elapsed = time.time() - start_time

        if reporter:
            reporter.add_result(task_name, workflow, success, elapsed)

        status = "SUCCESS" if success else "FAILED"
        print(f"\n  [{status}] Task '{task_name}' completed in {elapsed:.2f}s")
        return {"task_name": task_name, "success": success, "elapsed_sec": round(elapsed, 2), "error": None}

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n  EXCEPTION in task '{task_name}': {error_msg}")
        print(f"  Traceback:\n{traceback.format_exc()}")

        if reporter:
            class _Dummy:
                retry_count = 0
                used_knowledge_ids = []
                failure_history = [{"ticket_assigned_to": "System", "analysis": str(e)}]
                distillation_stats = {}
            reporter.add_result(task_name, _Dummy(), False, elapsed)

        return {"task_name": task_name, "success": False, "elapsed_sec": round(elapsed, 2), "error": error_msg}


def main():
    parser = argparse.ArgumentParser(description="Run Agentic Pipeline Task")
    parser.add_argument("--config", default=None)
    parser.add_argument("--task-config",
                        default=str(Path(__file__).parent / "config" / "tasks" / "debug_tasks.yaml"))
    parser.add_argument("--llm-config",
                        default=str(Path(__file__).parent / "config" / "llm.yaml"))
    parser.add_argument("--model", default="Vendor2/Claude-4.6-opus")
    parser.add_argument("--task-filter", help="Comma-separated task names to run")
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.llm_config, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f)
    with open(args.task_config, "r", encoding="utf-8") as f:
        task_config = yaml.safe_load(f)

    model_key = args.model
    if model_key not in llm_config["models"]:
        raise ValueError(f"Model '{model_key}' not found in {args.llm_config}")

    print(f"LLM Model: {model_key}")
    client, model_name = create_client(llm_config, model_key)

    skill_manager = create_skill_manager(config, client, model_name)
    skills_enabled = config.get("skills", {}).get("enabled", False)
    print(f"Skill Manager: {type(skill_manager).__name__} (enabled={skills_enabled})")

    reports_dir = config.get("paths", {}).get("reports_dir", str(Path(__file__).parent / "reports"))
    reporter = ExecutionReporter(root_output_dir=reports_dir)

    all_tasks = task_config.get("tasks", [])
    if not all_tasks:
        raise ValueError(f"No tasks in {args.task_config}")

    task_filter = args.task_filter or os.environ.get("TASK_NAMES", "").strip()
    if task_filter:
        names = [n.strip() for n in task_filter.split(",") if n.strip()]
        tasks_to_run = [t for t in all_tasks if t["name"] in names]
        if not tasks_to_run:
            raise ValueError(f"No matching tasks for filter: {task_filter}")
        print(f"Filtered: {len(tasks_to_run)}/{len(all_tasks)} tasks")
    else:
        tasks_to_run = all_tasks
        print(f"Running all {len(tasks_to_run)} tasks")

    results: List[Dict] = []
    total_start = time.time()

    for idx, task_info in enumerate(tasks_to_run, 1):
        print(f"\n[Task {idx}/{len(tasks_to_run)}]")
        result = run_single_task(task_info, client, model_name, config,
                                 skill_manager=skill_manager, reporter=reporter)
        results.append(result)
        if idx < len(tasks_to_run):
            time.sleep(1.0)

    total_elapsed = time.time() - total_start
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(results)} | Success: {len(successful)} | Failed: {len(failed)} | Time: {total_elapsed:.2f}s")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  - {r['task_name']}: {r['error'] or 'Workflow returned False'}")

    reporter.generate_report()
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
