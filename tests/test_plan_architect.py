"""
tests/test_plan_architect.py — Multi-model benchmark for Planner and Architect agents.

For each of 7 LLM models x 51 tasks:
  1. PlannerAgent generates a plan from the task's README.md
  2. ArchitectAgent generates architecture using approach.md as plan + README.md as task_desc

Models run in parallel (one thread per model). Results saved to plan_test/.

Usage:
    python -m tests.test_plan_architect
    python -m tests.test_plan_architect --task-filter ct_fan_beam,ct_sparse_view
    python -m tests.test_plan_architect --model-filter gemini,claude
    python -m tests.test_plan_architect --output-dir my_output
"""

import os
import sys
import yaml
import time
import argparse
import json
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from utils.llm_client import create_client
from agents.planner import PlannerAgent, CriticAgent
from agents.architect import ArchitectAgent


def load_configs(llm_config_path: str, task_config_path: str):
    with open(llm_config_path, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f)
    with open(task_config_path, "r", encoding="utf-8") as f:
        task_config = yaml.safe_load(f)

    # Normalize llm_config: if models are at root level (not nested under 'models'),
    # wrap them so create_client() can find them at llm_config['models'][key].
    if llm_config.get("models") is None or not isinstance(llm_config.get("models"), dict):
        model_entries = {k: v for k, v in llm_config.items() if isinstance(v, dict)}
        llm_config = {"models": model_entries}

    return llm_config, task_config


def get_first_n_models(llm_config: dict, n: int = 7):
    models = llm_config.get("models", {})
    return list(models.keys())[:n]


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _is_valid_output(path: str) -> bool:
    """Check if output file exists with valid (non-error) content."""
    if not os.path.exists(path):
        return False
    try:
        content = read_file(path)
        return len(content.strip()) > 0 and not content.startswith("ERROR:")
    except Exception:
        return False


def run_model(model_key: str, tasks: list, llm_config: dict, output_root: str,
              temperature: float = 0.7, use_critic: bool = False, max_critic_rounds: int = 3):
    """Worker function: run all tasks for a single model."""
    safe_name = model_key.split("/")[-1]
    plan_folder = "plan_critic" if use_critic else "plan"
    plan_dir = os.path.join(output_root, plan_folder, safe_name)
    arch_dir = os.path.join(output_root, "architect", safe_name)
    os.makedirs(plan_dir, exist_ok=True)
    os.makedirs(arch_dir, exist_ok=True)

    client, model_name = create_client(llm_config, model_key)

    results = {"plan_ok": 0, "plan_fail": 0, "arch_ok": 0, "arch_fail": 0,
               "arch_skip": 0, "plan_cached": 0, "arch_cached": 0}
    total = len(tasks)

    for idx, task in enumerate(tasks, 1):
        task_name = task["name"]
        readme_path = task.get("task_description_path", "")
        task_dir = task.get("task_dir", "")
        task_desc = None

        # --- Planner ---
        plan_file = os.path.join(plan_dir, f"{task_name}_plan.md")
        if _is_valid_output(plan_file):
            results["plan_cached"] += 1
            print(f"  [{safe_name}] [{idx}/{total}] Planner {task_name} ... CACHED")
        else:
            try:
                t0 = time.time()
                task_desc = read_file(readme_path)

                planner = PlannerAgent(client, model_name, temperature=temperature)
                plan_output = planner.generate({"task_desc": task_desc})

                if use_critic:
                    critic = CriticAgent(client, model_name, temperature=temperature)
                    for critic_round in range(max_critic_rounds):
                        critic_result = critic.generate({"task_desc": task_desc, "plan": plan_output})
                        parsed = json.loads(critic_result)
                        if parsed.get("decision") == "PASS":
                            print(f"  [{safe_name}] [{idx}/{total}] Critic {task_name} PASS (round {critic_round+1})")
                            break
                        suggestion = parsed.get("suggestion", "")
                        reason = parsed.get("reason", "")
                        print(f"  [{safe_name}] [{idx}/{total}] Critic {task_name} REJECT (round {critic_round+1}): {reason}")
                        feedback = f"Critic REJECTED your plan. Reason: {reason}. Suggestion: {suggestion}"
                        plan_output = planner.generate({"task_desc": task_desc, "feedback": feedback})

                with open(plan_file, "w", encoding="utf-8") as f:
                    f.write(plan_output)

                elapsed = time.time() - t0
                results["plan_ok"] += 1
                print(f"  [{safe_name}] [{idx}/{total}] Planner {task_name} ... OK ({elapsed:.1f}s)")
            except Exception as e:
                results["plan_fail"] += 1
                err_msg = f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                with open(plan_file, "w", encoding="utf-8") as f:
                    f.write(err_msg)
                print(f"  [{safe_name}] [{idx}/{total}] Planner {task_name} ... FAIL ({e})")

        # --- Architect ---
        arch_file = os.path.join(arch_dir, f"{task_name}_architect.md")
        approach_path = os.path.join(task_dir, "plan", "approach.md") if task_dir else ""

        if not approach_path or not os.path.exists(approach_path):
            results["arch_skip"] += 1
            if not os.path.exists(arch_file):
                with open(arch_file, "w", encoding="utf-8") as f:
                    f.write(f"SKIPPED: approach.md not found at {approach_path}")
            print(f"  [{safe_name}] [{idx}/{total}] Architect {task_name} ... SKIP (no approach.md)")
            continue

        if _is_valid_output(arch_file):
            results["arch_cached"] += 1
            print(f"  [{safe_name}] [{idx}/{total}] Architect {task_name} ... CACHED")
            continue

        try:
            t0 = time.time()
            if task_desc is None:
                task_desc = read_file(readme_path)
            approach = read_file(approach_path)

            architect = ArchitectAgent(client, model_name, temperature=temperature)
            arch_output = architect.generate({"task_desc": task_desc, "plan": approach})

            with open(arch_file, "w", encoding="utf-8") as f:
                f.write(arch_output)

            elapsed = time.time() - t0
            results["arch_ok"] += 1
            print(f"  [{safe_name}] [{idx}/{total}] Architect {task_name} ... OK ({elapsed:.1f}s)")
        except Exception as e:
            results["arch_fail"] += 1
            err_msg = f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            with open(arch_file, "w", encoding="utf-8") as f:
                f.write(err_msg)
            print(f"  [{safe_name}] [{idx}/{total}] Architect {task_name} ... FAIL ({e})")

    return model_key, results


def main():
    parser = argparse.ArgumentParser(description="Multi-model Plan & Architect benchmark")
    parser.add_argument("--llm-config", default=os.path.join(PROJECT_ROOT, "config", "llm.yaml"))
    parser.add_argument("--task-config", default=os.path.join(PROJECT_ROOT, "config", "tasks", "all_tasks.yaml"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "plan_test"))
    parser.add_argument("--task-filter", help="Comma-separated task names")
    parser.add_argument("--model-filter", help="Comma-separated model key substrings")
    parser.add_argument("--num-models", type=int, default=7, help="Number of models to test (from top of llm.yaml)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (default: 0.7)")
    parser.add_argument("--critic", action="store_true", help="Enable Critic review for plans (saved to plan_critic/)")
    parser.add_argument("--max-critic-rounds", type=int, default=3, help="Max Planner-Critic revision rounds (default: 3)")
    args = parser.parse_args()

    llm_config, task_config = load_configs(args.llm_config, args.task_config)
    model_keys = get_first_n_models(llm_config, args.num_models)

    if args.model_filter:
        filters = [f.strip() for f in args.model_filter.split(",") if f.strip()]
        model_keys = [k for k in model_keys if any(f in k for f in filters)]

    tasks = task_config.get("tasks", [])
    if args.task_filter:
        names = {n.strip() for n in args.task_filter.split(",") if n.strip()}
        tasks = [t for t in tasks if t["name"] in names]

    if not model_keys:
        print("No models selected. Check --model-filter or llm.yaml.")
        sys.exit(1)
    if not tasks:
        print("No tasks selected. Check --task-filter or auto_tasks.yaml.")
        sys.exit(1)

    print(f"Models ({len(model_keys)}): {model_keys}")
    print(f"Tasks  ({len(tasks)}): {[t['name'] for t in tasks]}")
    print(f"Output : {args.output_dir}")
    print(f"{'='*60}")

    start = time.time()

    with ThreadPoolExecutor(max_workers=len(model_keys)) as executor:
        futures = {
            executor.submit(run_model, mk, tasks, llm_config, args.output_dir,
                            args.temperature, args.critic, args.max_critic_rounds): mk
            for mk in model_keys
        }
        all_results = {}
        for future in as_completed(futures):
            mk = futures[future]
            try:
                model_key, results = future.result()
                all_results[model_key] = results
            except Exception as e:
                print(f"\n[FATAL] Model {mk} thread crashed: {e}")
                all_results[mk] = {"plan_ok": 0, "plan_fail": -1, "arch_ok": 0, "arch_fail": -1, "arch_skip": 0}

    total_time = time.time() - start

    # --- Summary ---
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'Plan OK':>8} {'Plan F':>8} {'Plan C':>8} {'Arch OK':>8} {'Arch F':>8} {'Arch S':>8} {'Arch C':>8}")
    print("-" * 106)
    for mk in model_keys:
        r = all_results.get(mk, {})
        print(f"{mk:<40} {r.get('plan_ok',0):>8} {r.get('plan_fail',0):>8} {r.get('plan_cached',0):>8} "
              f"{r.get('arch_ok',0):>8} {r.get('arch_fail',0):>8} {r.get('arch_skip',0):>8} {r.get('arch_cached',0):>8}")
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
