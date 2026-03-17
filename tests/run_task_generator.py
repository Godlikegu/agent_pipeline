"""从 test.md 生成任务描述并保存到 tests/output/task_desc/。"""
from __future__ import annotations

import sys
import yaml
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    markdown_path = Path("/home/guyuxuan/pipeline/tests/output/test.md").resolve()
    if not markdown_path.exists():
        print(f"[ERROR] Markdown 不存在: {markdown_path}")
        return 2

    out_dir = Path("/home/guyuxuan/pipeline/tests/output/task_desc").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "task_description.md"

    # 加载配置
    config_path = repo_root / "config" / "default.yaml"
    llm_config_path = repo_root / "config" / "llm.yaml"
    if not config_path.exists():
        print(f"[ERROR] 配置不存在: {config_path}")
        return 3
    if not llm_config_path.exists():
        print(f"[ERROR] LLM 配置不存在: {llm_config_path}")
        return 4

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(llm_config_path, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f)

    model_key = "cds/Claude-4.6-opus"
    if model_key not in llm_config.get("models", {}):
        model_key = next(iter(llm_config.get("models", {}).keys()), None)
    if not model_key:
        print("[ERROR] llm.yaml 中无可用模型")
        return 5

    from utils.llm_client import create_client
    from task_gen import TaskDescriptionGenerator

    client, model_name = create_client(llm_config, model_key)
    task_gen_cfg = config.get("task_gen", {})
    user_prompt = (
        task_gen_cfg.get("default_user_prompt")
        or "根据论文内容，生成一个适用于逆问题求解 pipeline 的任务描述，包含问题定义、输入输出约束和评估要求。"
    )

    generator = TaskDescriptionGenerator.from_config(
        client=client,
        model_name=model_name,
        config=task_gen_cfg,
    )

    print(f"[INFO] 输入: {markdown_path}")
    print(f"[INFO] 输出: {save_path}")
    print(f"[INFO] 模型: {model_key}")
    print("[INFO] 正在生成任务描述...")

    result = generator.generate_from_markdown_path(
        markdown_path=str(markdown_path),
        user_prompt=user_prompt,
        save_path=str(save_path),
    )

    print(f"[OK] 任务描述已保存: {result.task_description_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
