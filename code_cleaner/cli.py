from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from skills import create_skill_manager
from utils.config_loader import load_config
from utils.llm_client import create_client

from .cleaner import build_code_cleaner_from_config


def _parse_bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean flag value: {value}")


def _set_nested(config: dict, dotted_key: str, value) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        current = cursor.get(key)
        if not isinstance(current, dict):
            current = {}
            cursor[key] = current
        cursor = current
    cursor[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean and validate a GitHub repository or local codebase.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--github-url")
    source.add_argument("--local-repo")
    parser.add_argument("--paper-md")
    parser.add_argument("--task-family")
    parser.add_argument("--config", default=None, help="Optional config override YAML.")
    parser.add_argument(
        "--env-backend",
        choices=["auto", "conda", "venv"],
        default="auto",
        help="Environment backend policy. `auto` prefers conda and falls back to venv.",
    )
    parser.add_argument(
        "--python-version",
        default="3.10",
        help="Preferred Python version for auto-created environments when not pinned by the repo manifest.",
    )
    parser.add_argument(
        "--gpu-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="GPU runtime policy. `auto` prefers GPU when the host and repo support it.",
    )
    parser.add_argument(
        "--llm-config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "llm.yaml"),
        help="LLM config path used for code cleaning and optional skill extraction.",
    )
    parser.add_argument("--model", default="cds/Claude-4.6-opus")
    parser.add_argument("--no-provision", action="store_true", help="Skip environment provisioning.")
    parser.add_argument("--sandbox-root", help="Override paths.sandbox_root for repository environments.")
    parser.add_argument("--code-cleaner-root", help="Override paths.code_cleaner_root for cleaner artifacts.")
    parser.add_argument(
        "--llm-enabled",
        type=_parse_bool_flag,
        help="Override code_cleaner.llm_enabled with true/false.",
    )
    parser.add_argument(
        "--llm-required",
        type=_parse_bool_flag,
        help="Override code_cleaner.llm_required with true/false.",
    )
    parser.add_argument(
        "--force-rebuild-env",
        type=_parse_bool_flag,
        help="Override code_cleaner.force_rebuild_env with true/false.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.sandbox_root:
        _set_nested(config, "paths.sandbox_root", args.sandbox_root)
    if args.code_cleaner_root:
        _set_nested(config, "paths.code_cleaner_root", args.code_cleaner_root)
    if args.llm_enabled is not None:
        _set_nested(config, "code_cleaner.llm_enabled", args.llm_enabled)
    if args.llm_required is not None:
        _set_nested(config, "code_cleaner.llm_required", args.llm_required)
    if args.force_rebuild_env is not None:
        _set_nested(config, "code_cleaner.force_rebuild_env", args.force_rebuild_env)

    cleaner_cfg = config.get("code_cleaner", {})
    skills_enabled = config.get("skills", {}).get("enabled", False)
    llm_required = cleaner_cfg.get("llm_required", False)
    llm_enabled = cleaner_cfg.get("llm_enabled", llm_required)

    client = None
    model_name = None
    llm_config_path = Path(args.llm_config).expanduser().resolve()
    if llm_enabled or skills_enabled:
        if not llm_config_path.exists():
            if llm_required:
                raise FileNotFoundError(f"LLM cleaning requires config file: {llm_config_path}")
        else:
            llm_config = yaml.safe_load(llm_config_path.read_text(encoding="utf-8"))
            client, model_name = create_client(llm_config, args.model)

    skill_manager = None
    if skills_enabled:
        if client is not None and model_name is not None:
            skill_manager = create_skill_manager(config, client, model_name)
        else:
            skill_manager = create_skill_manager(config)

    cleaner = build_code_cleaner_from_config(
        config,
        skill_manager=skill_manager,
        provision_environment=not args.no_provision,
        env_backend=args.env_backend,
        python_version=args.python_version,
        gpu_mode=args.gpu_mode,
        llm_client=client,
        llm_model_name=model_name,
    )
    result = cleaner.clean(
        github_url=args.github_url,
        local_repo=args.local_repo,
        paper_md=args.paper_md,
        task_family=args.task_family,
    )

    print(
        json.dumps(
            {
                "repo_name": result.repo_name,
                "run_dir": str(result.run_dir),
                "cleaned_code_path": str(result.cleaned_code_path),
                "accepted": result.validation.accepted,
                "status": result.validation.status,
                "summary": result.summary,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.validation.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
