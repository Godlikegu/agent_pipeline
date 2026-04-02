"""
code_cleaner/cli.py -- CLI entry point for the code_cleaner module.

Usage:
    python -m code_cleaner.cli env-setup --tasks-dir data/tasks --envs-dir ./test_conda_envs
    python -m code_cleaner.cli env-setup --help
"""
import os
import sys
import argparse
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_env_setup(args):
    """Execute the environment setup command."""
    from utils.config_loader import load_config
    from utils.llm_client import create_client
    from code_cleaner.environment import CondaEnvManager, EnvSetupAgent

    config = load_config(args.config)
    cc_cfg = config.get("code_cleaner", {})
    env_cfg = cc_cfg.get("env_setup", {})

    # Resolve parameters: CLI args > config > defaults
    envs_dir = args.envs_dir or env_cfg.get("envs_dir", "./test_conda_envs")
    conda_path = args.conda_path or env_cfg.get("conda_path")
    python_version = args.python_version or env_cfg.get("python_version", "3.10")
    force_rebuild = args.force_rebuild or cc_cfg.get("force_rebuild_env", False)

    # Create conda manager
    conda_manager = CondaEnvManager(
        conda_path=conda_path,
        envs_dir=envs_dir,
        python_version=python_version,
    )

    # Create LLM client (unless --no-llm)
    if args.no_llm:
        # Use a dummy agent that just logs without LLM
        print("[EnvSetup] LLM disabled. Will install without LLM-assisted diagnosis.")
        client, model_name = None, ""
    else:
        with open(args.llm_config, "r", encoding="utf-8") as f:
            llm_config = yaml.safe_load(f)
        client, model_name = create_client(llm_config, args.model)

    # Create agent
    agent = EnvSetupAgent(
        client=client,
        model_name=model_name,
        conda_manager=conda_manager,
        config=config,
        temperature=cc_cfg.get("llm_temperature", 0.0),
    )

    # Parse task filter
    task_filter = None
    if args.task_filter:
        task_filter = [t.strip() for t in args.task_filter.split(",") if t.strip()]

    # Run setup
    tasks_dir = str(Path(args.tasks_dir).resolve())
    results = agent.setup_all_tasks(
        tasks_dir=tasks_dir,
        force_rebuild=force_rebuild,
        task_filter=task_filter,
    )

    # Optionally write output YAML
    if args.output_yaml:
        tasks_list = []
        for name, python_path in results.items():
            if python_path:
                task_dir = os.path.join(tasks_dir, name)
                tasks_list.append({
                    "name": name,
                    "python_path": python_path,
                    "task_dir": task_dir,
                    "task_description_path": os.path.join(task_dir, "README.md"),
                })
        output = {"tasks": tasks_list}
        output_path = Path(args.output_yaml)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(output, f, default_flow_style=False)
        print(f"[EnvSetup] Task config written to: {output_path}")

    # Exit code
    failed = [n for n, p in results.items() if p is None]
    if failed:
        print(f"\n[EnvSetup] {len(failed)} task(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"\n[EnvSetup] All {len(results)} task(s) configured successfully.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Code Cleaner: Environment Setup + Code Cleaning Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- env-setup subcommand ---
    env_parser = subparsers.add_parser(
        "env-setup",
        help="Set up conda environments for tasks in data/tasks/"
    )
    env_parser.add_argument(
        "--tasks-dir", default="data/tasks",
        help="Directory containing task folders (default: data/tasks)"
    )
    env_parser.add_argument(
        "--envs-dir", default=None,
        help="Directory to store conda environments (default: from config or ./test_conda_envs)"
    )
    env_parser.add_argument(
        "--task-filter", default=None,
        help="Comma-separated task names to process (default: all)"
    )
    env_parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Force rebuild of existing environments"
    )
    env_parser.add_argument(
        "--conda-path", default=None,
        help="Path to conda executable (default: auto-detect)"
    )
    env_parser.add_argument(
        "--python-version", default=None,
        help="Python version for new environments (default: from config or 3.10)"
    )
    env_parser.add_argument(
        "--llm-config", default=str(PROJECT_ROOT / "config" / "llm.yaml"),
        help="Path to LLM config YAML"
    )
    env_parser.add_argument(
        "--model", default="cds/Claude-4.6-opus",
        help="LLM model key from llm config"
    )
    env_parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM-assisted diagnosis"
    )
    env_parser.add_argument(
        "--config", default=None,
        help="Override config path (default: config/default.yaml)"
    )
    env_parser.add_argument(
        "--output-yaml", default=None,
        help="Write task configs with python_path to YAML file"
    )

    # --- Legacy support for clean_code.sh flat args ---
    # If the first arg looks like a legacy flag, redirect to full pipeline
    if len(sys.argv) > 1 and sys.argv[1] in ("--github-url", "--local-repo", "--task-family",
                                                "--sandbox-root", "--llm-enabled", "--llm-required",
                                                "--force-rebuild-env"):
        print("[code_cleaner] Legacy arguments detected. Forwarding to full pipeline...")
        print("[code_cleaner] Full pipeline not yet implemented. Use 'env-setup' subcommand.")
        sys.exit(1)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "env-setup":
        run_env_setup(args)


if __name__ == "__main__":
    main()
