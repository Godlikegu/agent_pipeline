"""
run_task.py — 主运行入口

使用方法：
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

# Ensure project root is in path when running as script
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.llm_client import create_client
from core.workflow import InverseProblemWorkflow
from skills import create_skill_manager
from utils.reporter import ExecutionReporter


def load_task_description(task_name: str, task_descriptions_dir: str) -> str:
    """加载任务描述，支持降级策略。"""
    desc_path = Path(task_descriptions_dir) / f"{task_name}_description.md"
    if desc_path.exists():
        print(f"  ✓ Loading task description from: {desc_path}")
        with open(desc_path, "r", encoding='utf-8') as f:
            return f.read()
    else:
        print(f"  ⚠ Warning: Task description not found at {desc_path}. Using default.")
        return f"Recover the signal from noisy measurements using a physics-based inverse solver. Task: {task_name}"


def run_single_task(
    task_info: dict,
    client: OpenAI,
    model_name: str,
    config: dict,
    skill_manager=None,
    reporter: Optional[ExecutionReporter] = None,
) -> Dict[str, any]:
    """执行单个任务并返回结构化结果。"""
    task_name = task_info['name']
    task_descriptions_dir = config['paths']['task_descriptions_dir']
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"▶ Starting Task: {task_name}")
    print(f"{'='*60}")
    print(f"  GT Code Path : {task_info['gt_code_path']}")
    print(f"  Python Path  : {task_info.get('python_path', 'default')}")

    try:
        task_description = load_task_description(task_name, task_descriptions_dir)

        workflow = InverseProblemWorkflow(
            task_name=task_name,
            task_desc=task_description,
            gt_code_path=task_info['gt_code_path'],
            python_path=task_info.get('python_path', sys.executable),
            client=client,
            model_name=model_name,
            config=config,
            skill_manager=skill_manager,
        )

        success = workflow.run()
        elapsed = time.time() - start_time

        if reporter:
            reporter.add_result(task_name, workflow, success, elapsed)

        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"\n  [{status}] Task '{task_name}' completed in {elapsed:.2f}s")
        return {"task_name": task_name, "success": success, "elapsed_sec": round(elapsed, 2), "error": None}

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n  ✗ EXCEPTION in task '{task_name}': {error_msg}")
        print(f"  Traceback:\n{traceback.format_exc()}")

        if reporter:
            class DummyWorkflow:
                retry_count = 0
                used_knowledge_ids = []
                failure_history = [{'ticket_assigned_to': 'System', 'analysis': str(e)}]
                distillation_stats = {}
            reporter.add_result(task_name, DummyWorkflow(), False, elapsed)

        return {"task_name": task_name, "success": False, "elapsed_sec": round(elapsed, 2), "error": error_msg}


def main():
    parser = argparse.ArgumentParser(description="Run Agentic Pipeline Task")
    parser.add_argument(
        "--config", default=None,
        help="Override config path (overrides values in config/default.yaml)"
    )
    parser.add_argument(
        "--task-config",
        default=str(Path(__file__).parent / "config" / "tasks" / "debug_tasks.yaml"),
        help="Path to task list YAML file"
    )
    parser.add_argument(
        "--llm-config",
        default=str(Path(__file__).parent / "config" / "llm.yaml"),
        help="Path to LLM config YAML file"
    )
    parser.add_argument(
        "--model", default="cds/Claude-4.6-opus",
        help="LLM model key (from llm.yaml)"
    )
    parser.add_argument(
        "--task-filter", help="Filter tasks by name (comma separated)"
    )
    args = parser.parse_args()

    # 加载统一配置（default.yaml + optional override）
    config = load_config(args.config)

    # 加载 LLM 配置
    with open(args.llm_config, 'r', encoding='utf-8') as f:
        llm_config = yaml.safe_load(f)

    # 加载任务列表
    with open(args.task_config, 'r', encoding='utf-8') as f:
        task_config = yaml.safe_load(f)

    model_key = args.model
    if model_key not in llm_config['models']:
        raise ValueError(f"Model '{model_key}' not found in {args.llm_config}")

    print(f"Using LLM Model: {model_key}")
    client, model_name = create_client(llm_config, model_key)

    # 初始化 SkillManager（通过工厂函数，自动处理 enabled/mode）
    skill_manager = create_skill_manager(config, client, model_name)
    print(f"✓ Skill Manager: {type(skill_manager).__name__} (mode={config['skills']['mode']}, enabled={config['skills']['enabled']})")

    # Initialize Reporter
    reporter = ExecutionReporter(root_output_dir=str(Path(__file__).parent / "reports"))

    # 确定待执行任务列表
    all_tasks = task_config.get('tasks', [])
    if not all_tasks:
        raise ValueError(f"No tasks found in {args.task_config}")

    task_filter = args.task_filter or os.environ.get("TASK_NAMES", "").strip()
    if task_filter:
        selected_names = [name.strip() for name in task_filter.split(",") if name.strip()]
        tasks_to_run = [t for t in all_tasks if t['name'] in selected_names]
        if not tasks_to_run:
            raise ValueError(f"No matching tasks found for filter: {task_filter}")
        print(f"Filtered tasks to run ({len(tasks_to_run)}/{len(all_tasks)}): {selected_names}")
    else:
        tasks_to_run = all_tasks
        print(f"Running all tasks ({len(tasks_to_run)} total)")

    # 执行所有任务
    results: List[Dict] = []
    total_start = time.time()

    for idx, task_info in enumerate(tasks_to_run, 1):
        print(f"\n[Task {idx}/{len(tasks_to_run)}]")
        result = run_single_task(task_info, client, model_name, config, skill_manager=skill_manager, reporter=reporter)
        results.append(result)

        if idx < len(tasks_to_run):
            time.sleep(1.0)

    total_elapsed = time.time() - total_start

    # 生成汇总报告
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total Tasks   : {len(results)}")
    print(f"Successful    : {len(successful)} ✓")
    print(f"Failed        : {len(failed)} ✗")
    print(f"Total Time    : {total_elapsed:.2f}s")

    if failed:
        print("\nFailed Tasks Details:")
        for r in failed:
            print(f"  - {r['task_name']}: {r['error'] or 'Workflow returned False'}")

    reporter.generate_report()
    print("="*60)

    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
