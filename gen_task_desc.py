"""
gen_task_desc.py — Generate task description from user prompt and optional sources.

Usage:
    python -m gen_task_desc \
        --user-prompt "Implement a sparse deconvolution algorithm" \
        --paper-markdown-path /data/paper.md \
        --cleaned-code-path /data/code_cleaned.py \
        --output-path ./data/task_descriptions/my_task.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from utils.config_loader import load_config
from utils.llm_client import create_client
from utils.task_description_utils import (
    load_or_generate_task_description,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate task description from user prompt and optional sources.")
    parser.add_argument("--user-prompt", required=True, help="Required user request (primary source of truth)")
    parser.add_argument("--paper-markdown-path", help="Optional paper markdown file")
    parser.add_argument("--cleaned-code-path", help="Optional cleaned code file")
    parser.add_argument("--gt-code-path", help="Optional ground truth code file")
    parser.add_argument("--readme-path", help="Optional README file")
    parser.add_argument("--config-paths", nargs="*", help="Optional config files")
    parser.add_argument("--test-paths", nargs="*", help="Optional test files")
    parser.add_argument("--task-name", default="generated-task", help="Task name for output file naming")
    parser.add_argument("--output-path", help="Output path for generated task description")
    parser.add_argument("--task-description-path", help="If provided, read existing description instead of generating")

    parser.add_argument("--config", default=None, help="Override config YAML")
    parser.add_argument("--llm-config",
                        default=str(Path(__file__).resolve().parent / "config" / "llm.yaml"))
    parser.add_argument("--model", default="cds/Claude-4.6-opus")
    args = parser.parse_args()

    config = load_config(args.config)

    task_info = {
        "name": args.task_name,
        "user_prompt": args.user_prompt,
        "paper_markdown_path": args.paper_markdown_path,
        "cleaned_code_path": args.cleaned_code_path,
        "gt_code_path": args.gt_code_path,
        "readme_path": args.readme_path,
        "task_description_path": args.task_description_path,
        "config_paths": args.config_paths or [],
        "test_paths": args.test_paths or [],
    }

    if args.output_path:
        task_info["_output_path"] = args.output_path

    llm_config = yaml.safe_load(Path(args.llm_config).expanduser().resolve().read_text(encoding="utf-8"))
    client, model_name = create_client(llm_config, args.model)

    task_description = load_or_generate_task_description(
        task_info=task_info,
        config=config,
        client=client,
        model_name=model_name,
    )

    output_path = args.output_path
    if not output_path:
        desc_dir = config.get("paths", {}).get("task_descriptions_dir", "./data/task_descriptions")
        output_path = str(Path(desc_dir) / f"{args.task_name}_description.md")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(task_description, encoding="utf-8")

    print(f"Task description saved to: {output_path}")
    print(f"Length: {len(task_description)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
