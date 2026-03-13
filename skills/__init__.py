"""
skills — 知识管理系统

提供 create_skill_manager() 工厂函数，根据 config/default.yaml 中的
skills.enabled 和 skills.mode 配置创建对应的 SkillManager 实例。

Pipeline 侧通过此工厂获取 skill_manager，无需判断 None，
统一调用 retrieve_knowledge / distill_and_store 等接口。
skills.enabled=false 时返回 NoSkillManager（空实现，无副作用）。
"""

from .manager import SkillManager
from .ablation import (
    NoSkillManager,
    InstanceOnlyManager,
    ExperienceOnlyManager,
    InstanceExpManager,
)


def create_skill_manager(config: dict, client=None, model_name: str = "") -> object:
    """
    根据配置创建 SkillManager 实例。

    Args:
        config: 完整配置字典（load_config() 返回值）。
        client: OpenAI-compatible LLM 客户端。
        model_name: LLM 模型名。

    Returns:
        SkillManager 或其消融变体。skills.enabled=false 时返回 NoSkillManager。
    """
    skills_cfg = config.get("skills", {})

    if not skills_cfg.get("enabled", True):
        return NoSkillManager()

    mode = skills_cfg.get("mode", "default")

    if mode == "none":
        return NoSkillManager()

    # 创建真实的 SkillManager
    db_path = config.get("paths", {}).get("skills_db", "./data/skills.db")
    real_manager = SkillManager(
        db_path=db_path,
        client=client,
        model_name=model_name,
        config=skills_cfg,
    )

    mode_map = {
        "default": real_manager,
        "instance": InstanceOnlyManager(real_manager),
        "experience": ExperienceOnlyManager(real_manager),
        "instance_exp": InstanceExpManager(real_manager),
    }

    return mode_map.get(mode, real_manager)
