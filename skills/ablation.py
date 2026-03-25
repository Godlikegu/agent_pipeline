"""
Ablation / no-op skill managers.

NoSkillManager is used when both retrieval_enabled and learning_enabled are false.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set


class NoSkillManager:
    """Null-object skill manager -- every method is a safe no-op."""

    retrieval_enabled = False
    learning_enabled = False

    def retrieve_knowledge(
        self, task_desc: str = "", agent_role: str = "Planner",
        top_k: int = 3, exclude_ids: Set[str] = None,
    ) -> Dict[str, List]:
        return {"knowledge_general": [], "knowledge_task_specific": [], "code": []}

    def format_knowledge_for_prompt(self, knowledge: Dict[str, List]) -> str:
        return ""

    def get_knowledge_details(self, knowledge_ids: List[str]) -> List[Dict]:
        return []

    def distill_from_trajectories(
        self, task_name: str = "", task_desc: str = "",
        trajectories: List[dict] = None, final_outcome: str = "",
    ) -> list:
        return []

    def promote_used_skills(self, used_ids: Set[str] = None, task_name: str = "") -> list:
        return []

    def cleanup_draft_skills(self, task_name: str = "") -> int:
        return 0
