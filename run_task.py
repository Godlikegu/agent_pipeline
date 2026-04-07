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


def setup_task_sandbox(task_dir: str, sandbox_dir: str) -> dict:
    """Build sandbox from a structured task directory.

    Creates:
      sandbox_dir/
        dataset/
          gt_output.npy      <- extracted from ground_truth.npz / baseline_reference.npz (or copied from ground_truth.npy)
          raw_data.npz        <- from task_dir/data/raw_data.npz (+ raw_data1.npz, etc.)
          meta_data.json      <- from task_dir/data/meta_data.json
          data_info.json      <- auto-generated: npz keys, shapes, dtypes

    Returns loaded meta_data dict (or empty dict if not found).
    """
    import numpy as np

    task_dir = os.path.abspath(task_dir)
    sandbox_dir = os.path.abspath(sandbox_dir)
    dataset_dir = os.path.join(sandbox_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    data_dir = os.path.join(task_dir, "data")
    meta_data = {}
    data_info = {}

    # --- 1. Ground truth / baseline reference -> gt_output.npy ---
    gt_dst = os.path.join(dataset_dir, "gt_output.npy")
    if not os.path.exists(gt_dst):
        gt_npy = os.path.join(data_dir, "ground_truth.npy")
        gt_npz = os.path.join(data_dir, "ground_truth.npz")
        bl_npz = os.path.join(data_dir, "baseline_reference.npz")
        if os.path.exists(gt_npy):
            shutil.copy2(gt_npy, gt_dst)
            print(f"  Copied ground_truth.npy -> dataset/gt_output.npy")
        elif os.path.exists(gt_npz):
            npz = np.load(gt_npz)
            keys = list(npz.keys())
            if keys:
                gt_key = keys[0]
                arr = npz[gt_key]
                np.save(gt_dst, arr)
                print(f"  Extracted ground_truth.npz['{gt_key}'] -> dataset/gt_output.npy "
                      f"(shape={arr.shape}, dtype={arr.dtype})")
                with open(os.path.join(dataset_dir, "gt_key.txt"), "w") as f:
                    f.write(gt_key)
        elif os.path.exists(bl_npz):
            npz = np.load(bl_npz)
            keys = list(npz.keys())
            if keys:
                bl_key = keys[0]
                arr = npz[bl_key]
                np.save(gt_dst, arr)
                print(f"  Extracted baseline_reference.npz['{bl_key}'] -> dataset/gt_output.npy "
                      f"(shape={arr.shape}, dtype={arr.dtype})")
                with open(os.path.join(dataset_dir, "gt_key.txt"), "w") as f:
                    f.write(bl_key)

    # Record gt_output.npy info
    if os.path.exists(gt_dst):
        gt_arr = np.load(gt_dst, allow_pickle=True)
        data_info["gt_output.npy"] = {"shape": list(gt_arr.shape), "dtype": str(gt_arr.dtype)}

    # --- 2. Copy meta_data.json ---
    meta_src = os.path.join(data_dir, "meta_data.json")
    meta_dst = os.path.join(dataset_dir, "meta_data.json")
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, meta_dst)
        with open(meta_src, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        print(f"  Copied meta_data.json -> dataset/meta_data.json")

    # --- 3. Copy raw_data*.npz files directly into dataset/ ---
    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if fname.startswith("raw_data") and fname.endswith(".npz"):
                src = os.path.join(data_dir, fname)
                dst = os.path.join(dataset_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  Copied {fname} -> dataset/{fname}")

    # --- 4. Generate data_info.json: npz keys/shapes/dtypes ---
    for fname in sorted(os.listdir(dataset_dir)):
        fpath = os.path.join(dataset_dir, fname)
        if fname.endswith(".npz") and os.path.isfile(fpath):
            npz = np.load(fpath)
            file_info = {}
            for k in npz.keys():
                arr = npz[k]
                file_info[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            data_info[fname] = file_info

    if data_info:
        info_path = os.path.join(dataset_dir, "data_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=2)
        print(f"  Generated dataset/data_info.json")

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
        if task_dir:
            metrics_json_path = os.path.join(task_dir, "evaluation", "metrics.json")
            if os.path.exists(metrics_json_path):
                with open(metrics_json_path, "r", encoding="utf-8") as f:
                    task_metrics = json.load(f)
                if "ncc_boundary" in task_metrics:
                    eval_thresholds["min_ncc"] = task_metrics["ncc_boundary"]
                if "nrmse_boundary" in task_metrics:
                    eval_thresholds["max_nrmse"] = task_metrics["nrmse_boundary"]
                print(f"  Eval thresholds: ncc>={eval_thresholds.get('min_ncc')}, "
                      f"nrmse<={eval_thresholds.get('max_nrmse')}")

        # Append sandbox data layout to task description so all agents know actual paths
        dataset_dir = os.path.join(sandbox_dir, "dataset")
        layout_lines = []
        # List .npz files with key/shape/dtype info
        data_info_path = os.path.join(dataset_dir, "data_info.json")
        if os.path.exists(data_info_path):
            with open(data_info_path, "r", encoding="utf-8") as f:
                data_info = json.load(f)
            for fname, info in data_info.items():
                if fname == "gt_output.npy":
                    continue  # Don't expose GT to solver agents
                if isinstance(info, dict) and "shape" in info:
                    # Plain .npy entry
                    layout_lines.append(f"  - dataset/{fname} shape={info['shape']} dtype={info['dtype']}")
                else:
                    # .npz entry with keys
                    for key, kinfo in info.items():
                        layout_lines.append(
                            f"  - dataset/{fname} key='{key}' shape={kinfo['shape']} dtype={kinfo['dtype']}"
                        )
        else:
            # Fallback: scan dataset directory
            for fname in sorted(os.listdir(dataset_dir)):
                fpath = os.path.join(dataset_dir, fname)
                if os.path.isfile(fpath) and fname not in ("gt_output.npy", "gt_key.txt", "data_info.json"):
                    layout_lines.append(f"  - dataset/{fname}")
        if os.path.exists(os.path.join(dataset_dir, "meta_data.json")):
            layout_lines.append("  - dataset/meta_data.json (physical parameters)")
        if layout_lines:
            layout_section = (
                "\n\n### SANDBOX DATA LAYOUT (CRITICAL - use ONLY these paths)\n"
                "Available files in the working directory:\n"
                + "\n".join(layout_lines)
                + "\n\nYou MUST save the final reconstruction result to `output.npy` in the working directory."
                "\nDo NOT use any other input paths — only the files listed above exist."
                "\nFor .npz files: use `data = np.load('dataset/<file>.npz'); arr = data['<key>']` to load."
            )
            task_description += layout_section

        # --- Phase 3: Generate eval_script.py if missing ---
        eval_script_path = os.path.join(sandbox_dir, "eval_script.py")
        if not os.path.exists(eval_script_path):
            from agents.sandbox_agents import EvalGenAgent, get_installed_libraries
            from utils.text_utils import extract_python

            python_path = task_info.get("python_path", sys.executable)
            eval_gen_agent = EvalGenAgent(client, model_name)
            package_list = get_installed_libraries(python_path)

            # Detect data shape for hint
            import numpy as np
            data_shape_hint = "N/A"
            gt_path = os.path.join(sandbox_dir, "dataset", "gt_output.npy")
            if os.path.exists(gt_path):
                gt = np.load(gt_path, allow_pickle=True)
                data_shape_hint = f"shape={gt.shape}, dtype={gt.dtype}"

            eval_ctx = {
                "task_desc": task_description,
                "data_shape_hint": data_shape_hint,
                "package_list": package_list,
                "meta_data": meta_data,
                "eval_thresholds": eval_thresholds,
            }
            print("  Generating eval_script.py...")
            eval_code = eval_gen_agent.generate(eval_ctx)
            eval_code = extract_python(eval_code)
            with open(eval_script_path, "w", encoding="utf-8") as f:
                f.write(eval_code)
            print("  eval_script.py generated.")

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
    parser.add_argument("--model", default="cds/Claude-4.6-opus")
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
