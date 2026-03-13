"""
Ablation Study Skill Manager Wrappers

Provides filtered SkillManager variants for ablation experiments.
All wrappers use a FROZEN database (read-only: no distillation, no score updates).

Four modes:
  1. NoSkillManager       — No persistent skill system at all (baseline)
  2. InstanceOnlyManager  — Only instance (few-shot) retrieval
  3. ExperienceOnlyManager— Only experience (pattern) retrieval
  4. InstanceExpManager    — Instance + Experience retrieval (no core knowledge)
"""

from typing import List, Dict, Optional, Any


class _FrozenManagerBase:
    """
    Base class that wraps a real SkillManager but freezes it:
      - retrieve_knowledge: filtered by subclass
      - update_scores: no-op
      - distill_and_store: no-op
    """

    # Subclasses set these to True/False to control which layers are active
    _enable_core: bool = False
    _enable_experience: bool = False
    _enable_instance: bool = False

    def __init__(self, real_manager):
        """
        Args:
            real_manager: A fully initialized SkillManager with a loaded DB.
        """
        self._real = real_manager
        # Expose storage for any code that accesses it directly
        self.storage = real_manager.storage

    # ---- Retrieval (filtered) ----

    def retrieve_knowledge(self, task_desc: str, agent_role: str = 'General', top_k: int = 3) -> Dict[str, List[Dict]]:
        """Retrieve knowledge from the real manager, then zero-out disabled layers."""
        full_results = self._real.retrieve_knowledge(task_desc, agent_role, top_k)

        filtered = {
            "core": full_results.get("core", []) if self._enable_core else [],
            "experience": full_results.get("experience", []) if self._enable_experience else [],
            "instance": full_results.get("instance", []) if self._enable_instance else [],
        }
        return filtered

    def format_knowledge_for_prompt(self, knowledge: Dict[str, List[Dict]]) -> str:
        """Delegate formatting to real manager."""
        return self._real.format_knowledge_for_prompt(knowledge)

    def get_embedding(self, text: str) -> List[float]:
        return self._real.get_embedding(text)

    def get_knowledge_details(self, knowledge_ids: List[str]) -> List[Dict]:
        return self._real.get_knowledge_details(knowledge_ids)

    # ---- Frozen operations (no-op) ----

    def update_scores(self, knowledge_ids: List[str], success: bool):
        """No-op: frozen database does not update scores."""
        pass

    def distill_and_store(self, trajectory: Dict[str, Any]) -> Dict[str, int]:
        """No-op: frozen database does not distill new knowledge."""
        return {'instances': 0, 'experiences': 0, 'core': 0}


# =============================================================================
# Concrete ablation variants
# =============================================================================

class NoSkillManager:
    """
    Mode 1: No skill system at all.

    This is NOT a wrapper — it's a standalone object that always returns empty.
    Setting skill_manager=None would also work, but this class allows the workflow
    code to call methods without None-checks failing.
    """

    def retrieve_knowledge(self, task_desc: str, agent_role: str = 'General', top_k: int = 3) -> Dict[str, List[Dict]]:
        return {"core": [], "experience": [], "instance": []}

    def format_knowledge_for_prompt(self, knowledge: Dict[str, List[Dict]]) -> str:
        return ""

    def update_scores(self, knowledge_ids: List[str], success: bool):
        pass

    def distill_and_store(self, trajectory: Dict[str, Any]) -> Dict[str, int]:
        return {'instances': 0, 'experiences': 0, 'core': 0}

    def get_knowledge_details(self, knowledge_ids: List[str]) -> List[Dict]:
        return []


class InstanceOnlyManager(_FrozenManagerBase):
    """Mode 2: Only instance (few-shot examples) retrieval."""
    _enable_core = False
    _enable_experience = False
    _enable_instance = True


class ExperienceOnlyManager(_FrozenManagerBase):
    """Mode 3: Only experience (pattern) retrieval."""
    _enable_core = False
    _enable_experience = True
    _enable_instance = False


class InstanceExpManager(_FrozenManagerBase):
    """Mode 4: Instance + Experience retrieval (no core knowledge)."""
    _enable_core = False
    _enable_experience = True
    _enable_instance = True
