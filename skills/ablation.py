"""
Ablation / no-op skill managers.

NoSkillManager is used when skills.enabled=false.
"""
from __future__ import annotations

from typing import Any, Dict, List


class NoSkillManager:
    """Null-object skill manager — every method is a safe no-op."""

    def retrieve_knowledge(self, task_desc: str = "", agent_role: str = "General", top_k: int = 3) -> Dict[str, List]:
        return {"knowledge_general": [], "knowledge_task_specific": [], "code": []}

    def format_knowledge_for_prompt(self, knowledge: Dict[str, List]) -> str:
        return ""

    def get_knowledge_details(self, knowledge_ids: List[str]) -> List[Dict]:
        return []

    def update_scores(self, knowledge_ids: List[str], success: bool = True) -> None:
        pass

    def distill_and_store(self, trajectory: Dict[str, Any]) -> Dict[str, int]:
        return {"knowledge_general": 0, "knowledge_task_specific": 0, "code": 0}
