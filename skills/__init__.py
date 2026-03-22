"""
Skill system — knowledge skills (general + task-specific) and code skills.

Skills are stored as Claude-compatible SKILL.md files.
Toggle via config `skills.enabled`.
"""
from .ablation import NoSkillManager


def create_skill_manager(config: dict, client=None, model_name: str = "") -> object:
    """Factory: returns FileSkillManager or NoSkillManager based on config."""
    skills_cfg = config.get("skills", {})
    if not skills_cfg.get("enabled", False):
        return NoSkillManager()

    from .file_manager import FileSkillManager

    paths_cfg = config.get("paths", {})
    merged = dict(skills_cfg)
    merged["_paths"] = {
        "active_dir": paths_cfg.get("skills_active_dir", "./.claude/skills"),
        "draft_dir": paths_cfg.get("skills_draft_dir", "./skills/library/drafts"),
        "registry": paths_cfg.get("skills_registry", "./skills/library/registry.json"),
        "code_pool": paths_cfg.get("skills_code_pool", "./skills/library/code_pool"),
        "embedding_model_dir": paths_cfg.get("skills_embedding_model_dir", "./skills/embeddings/all-MiniLM-L6-v2"),
    }
    return FileSkillManager(client=client, model_name=model_name, config=merged)
