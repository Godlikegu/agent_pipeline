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
from agents.task_generator import DEFAULT_USER_PROMPT, TaskGeneratorAgent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_paper_markdown_path(task_info: dict, config: dict) -> Optional[Path]:
    explicit_path = task_info.get("paper_markdown_path")
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    default_dir = config.get("task_gen", {}).get("paper_markdown_dir")
    if default_dir:
        candidate = Path(default_dir).expanduser().resolve() / f"{task_info['name']}.md"
        if candidate.exists():
            return candidate
    return None


def load_task_description(
    task_info: dict,
    config: dict,
    client: OpenAI,
    model_name: str,
) -> str:
    """Load an explicit task description or generate one from paper markdown.
    When generating, saves to task_description_path if set (cache: next run will load from same path).
    """
    explicit_desc_path = task_info.get("task_description_path")
    if explicit_desc_path:
        desc_path = Path(explicit_desc_path).expanduser().resolve()
        if desc_path.exists():
            print(f"  ✓ Loading task description from: {desc_path}")
            return _read_text(desc_path)
        # 文件不存在则继续走“从论文生成”，并保存到 task_description_path，相当于缓存

    paper_markdown_path = _resolve_paper_markdown_path(task_info, config)
    if paper_markdown_path is None:
        raise FileNotFoundError(
            "No task_description_path (with existing file) and no paper markdown could be found. "
            "Provide task_description_path, or paper_markdown_path / paper_markdown_dir."
        )
    if not paper_markdown_path.exists():
        raise FileNotFoundError(f"Paper markdown not found: {paper_markdown_path}")

    task_gen_cfg = config.get("task_gen", {})
    user_prompt = (
        task_info.get("user_prompt")
        or task_gen_cfg.get("default_user_prompt")
        or DEFAULT_USER_PROMPT
    )
    default_desc_dir = config.get("paths", {}).get(
        "task_descriptions_dir", "./data/task_descriptions"
    )
    # 优先写到 task_description_path，实现“同路径读写”的缓存
    output_path = Path(
        task_info.get("task_description_path")
        or task_info.get("task_description_output_path")
        or (Path(default_desc_dir) / f"{task_info['name']}_description.md")
    ).expanduser().resolve()

    generator = TaskGeneratorAgent.from_config(
        client=client,
        model_name=model_name,
        config=config,
    )

    print(f"  ✓ Generating task description from markdown: {paper_markdown_path}")
    result = generator.generate_from_markdown_path(
        markdown_path=paper_markdown_path,
        user_prompt=user_prompt,
        save_path=output_path,
    )
    print(f"  ✓ Generated task description saved to: {result.task_description_path}")
    return result.task_description


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
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"▶ Starting Task: {task_name}")
    print(f"{'='*60}")
    print(f"  GT Code Path : {task_info['gt_code_path']}")
    print(f"  Python Path  : {task_info.get('python_path', 'default')}")

    try:
        task_description = load_task_description(
            task_info=task_info,
            config=config,
            client=client,
            model_name=model_name,
        )

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
