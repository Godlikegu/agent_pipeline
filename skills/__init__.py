"""
Skill system -- knowledge skills (general + task-specific) and code skills.

Toggle via config:
  skills.retrieval_enabled  (read-only, default=true)
  skills.learning_enabled   (write, default=false, implies retrieval_enabled=true)
"""
from .ablation import NoSkillManager


def create_skill_manager(config: dict, client=None, model_name: str = "") -> object:
    """Factory: returns FileSkillManager or NoSkillManager based on config."""
    skills_cfg = config.get("skills", {})

    retrieval_enabled = skills_cfg.get("retrieval_enabled", False)
    learning_enabled = skills_cfg.get("learning_enabled", False)

    # Backward compat: old 'enabled' key
    if "enabled" in skills_cfg and "retrieval_enabled" not in skills_cfg:
        retrieval_enabled = skills_cfg["enabled"]

    # Write implies read
    if learning_enabled:
        retrieval_enabled = True

    if not retrieval_enabled and not learning_enabled:
        return NoSkillManager()

    from .file_manager import FileSkillManager

    paths_cfg = config.get("paths", {})
    merged = dict(skills_cfg)
    merged["retrieval_enabled"] = retrieval_enabled
    merged["learning_enabled"] = learning_enabled
    merged["_paths"] = {
        "active_dir": paths_cfg.get("skills_active_dir", "./.claude/skills"),
        "draft_dir": paths_cfg.get("skills_draft_dir", "./skills/library/drafts"),
        "registry": paths_cfg.get("skills_registry", "./skills/library/registry.json"),
        "code_pool": paths_cfg.get("skills_code_pool", "./skills/library/code_pool"),
        "embedding_model_dir": paths_cfg.get(
            "skills_embedding_model_dir", "./skills/embeddings/all-MiniLM-L6-v2"
        ),
        "trajectories_dir": paths_cfg.get("trajectories_dir", "./data"),
    }
    return FileSkillManager(client=client, model_name=model_name, config=merged)
