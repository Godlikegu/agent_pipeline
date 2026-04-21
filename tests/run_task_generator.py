"""Generate a task description from a local markdown file."""
from __future__ import annotations

import os
import sys
import yaml
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    markdown_path = Path(
        os.environ.get("MARKDOWN_PATH", repo_root / "data" / "paper_markdown" / "example.md")
    ).resolve()
    if not markdown_path.exists():
        print(f"[ERROR] Markdown file not found: {markdown_path}")
        return 2

    out_dir = Path(
        os.environ.get("TASK_DESC_OUTPUT_DIR", repo_root / "tests" / "output" / "task_desc")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "task_description.md"

    config_path = repo_root / "config" / "default.yaml"
    llm_config_path = repo_root / "config" / "llm.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return 3
    if not llm_config_path.exists():
        print(f"[ERROR] LLM config not found: {llm_config_path}")
        return 4

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(llm_config_path, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f)

    model_key = os.environ.get("MODEL_NAME", "example/default-model")
    if model_key not in llm_config.get("models", {}):
        model_key = next(iter(llm_config.get("models", {}).keys()), None)
    if not model_key:
        print("[ERROR] No model entry found in config/llm.yaml")
        return 5

    from utils.llm_client import create_client
    from agents.task_generator import TaskGeneratorAgent

    client, model_name = create_client(llm_config, model_key)
    task_gen_cfg = config.get("task_gen", {})
    user_prompt = (
        task_gen_cfg.get("default_user_prompt")
        or "Generate an implementation-ready task description for the inverse-problem pipeline."
    )

    generator = TaskGeneratorAgent.from_config(
        client=client,
        model_name=model_name,
        config=config,
    )

    print(f"[INFO] Input : {markdown_path}")
    print(f"[INFO] Output: {save_path}")
    print(f"[INFO] Model : {model_key}")
    print("[INFO] Generating task description...")

    result = generator.generate_from_markdown_path(
        markdown_path=str(markdown_path),
        user_prompt=user_prompt,
        save_path=str(save_path),
    )

    print(f"[OK] Task description saved to {result.task_description_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
