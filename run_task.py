"""
run_task.py — Main pipeline entry point.

Runs one or more tasks from a YAML task list.
Assumes sandbox has been pre-configured by code_cleaner (clean_code.sh).

Usage:
    python -m run_task --task-config config/tasks/debug_tasks.yaml --model cds/Claude-4.6-opus
    python -m run_task --task-filter sim,bpm
"""
import os
import sys
import yaml
import time
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
        sandbox_root = paths_cfg.get("sandbox_root", "/data/guyuxuan/agent/end_sandbox")
        sandbox_dir = os.path.join(sandbox_root, f"{task_name}_sandbox")

    reference_code_path = None
    try:
        reference_code_path = str(resolve_reference_code_path(task_info))
    except (ValueError, FileNotFoundError):
        pass

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    print(f"  Sandbox      : {sandbox_dir}")
    if reference_code_path:
        print(f"  Reference Code: {reference_code_path}")
    print(f"  Python Path  : {task_info.get('python_path', 'default')}")

    try:
        if reference_code_path and not os.path.exists(os.path.join(sandbox_dir, "dataset")):
            from code_cleaner.test_harness import setup_sandbox
            setup_sandbox(sandbox_dir, reference_code_path)

        task_description = load_or_generate_task_description(
            task_info=task_info,
            config=config,
            client=client,
            model_name=model_name,
        )

        if not os.path.exists(os.path.join(sandbox_dir, "eval_script.py")) and reference_code_path:
            from code_cleaner.test_harness import generate_test_harness
            from agents.sandbox_agents import DataGenAgent, EvalGenAgent, get_installed_libraries
            from utils.text_utils import extract_python

            python_path = task_info.get("python_path", sys.executable)
            data_gen_agent = DataGenAgent(client, model_name)
            eval_gen_agent = EvalGenAgent(client, model_name)
            package_list = get_installed_libraries(python_path)

            def write_fn(fname, content):
                content = extract_python(content)
                with open(os.path.join(sandbox_dir, fname), "w", encoding="utf-8") as f:
                    f.write(content)

            cc_cfg = config.get("code_cleaner", {})
            generate_test_harness(
                sandbox_dir=sandbox_dir,
                python_path=python_path,
                task_desc=task_description,
                data_gen_agent=data_gen_agent,
                eval_gen_agent=eval_gen_agent,
                write_file_fn=write_fn,
                data_gen_timeout=cc_cfg.get("data_gen_timeout", 180),
                gt_code_snippet_limit=cc_cfg.get("gt_code_snippet_limit", 4000),
                max_retries=cc_cfg.get("data_gen_max_retries", 5),
                package_list=package_list,
            )

        workflow = PipelineWorkflow(
            task_name=task_name,
            task_desc=task_description,
            sandbox_dir=sandbox_dir,
            python_path=task_info.get("python_path", sys.executable),
            client=client,
            model_name=model_name,
            config=config,
            skill_manager=skill_manager,
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
