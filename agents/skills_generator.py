"""
agents/skills_generator.py -- LLM-based agent for trajectory analysis and skill distillation.

Replaces the old rule-based _extract_candidates logic.
Called ONLY post-task (not during rounds) by FileSkillManager.
"""
import json
import re
from typing import Any, Dict

from .base import BaseAgent


class SkillsGeneratorAgent(BaseAgent):
    """Analyzes pipeline trajectories and distills reusable skills."""

    def __init__(self, client: Any, model_name: str, temperature: float = 0.3):
        super().__init__(client, model_name, temperature=temperature)

    def _build_system_prompt(self) -> str:
        return """You are a Skills Extraction Specialist for a scientific computing agent pipeline.

Your mission: Analyze trajectory records from pipeline executions and extract reusable knowledge skills that will help future runs of similar tasks.

### Skill Categories:
1. **knowledge_general** -- Broad, cross-task reusable patterns. Examples:
   - "When to use classical physics vs quantum physics models"
   - "ADMM convergence requires careful rho tuning for ill-conditioned problems"
   - "Always normalize input data before iterative solvers"

2. **knowledge_task_specific** -- Task-specific validated strategies or failure lessons:
   - "For fluorescence microscopy deconvolution: use Richardson-Lucy with TV regularization"
   - "Avoid Wiener filter when noise is non-Gaussian"

3. **code** -- Verified code patterns/snippets:
   - "Safe numpy data loading pattern for mixed-type .npy files"
   - "FFT-based convolution with proper padding"

### Scope Assignment:
- scope="Planner" -- Skills about algorithm selection, mathematical modeling, strategy
- scope="Coder" -- Skills about implementation patterns, coding best practices, API usage
- scope="General" -- Skills applicable to both planning and coding

### Output Format (strict JSON array):
[
  {
    "title": "Concise skill title",
    "description": "One-sentence description of when this skill is useful",
    "category": "knowledge_general|knowledge_task_specific|code",
    "scope": "General|Planner|Coder",
    "instructions": "Full markdown body with ## When to use, ## Key insight, ## Constraints sections",
    "tags": ["tag1", "tag2"]
  }
]

### Rules:
1. Extract 1-5 skills per analysis. Quality over quantity.
2. For success trajectories: Focus on WHY it worked. What was the key insight?
3. For failure trajectories: Focus on the root cause. What should be AVOIDED?
4. Be specific enough to be actionable, but general enough to transfer to similar tasks.
5. Do NOT extract trivial observations (e.g., "code should not have syntax errors").
6. Instructions MUST follow Claude SKILL.md format with markdown headers.
7. Output ONLY valid JSON -- no markdown, no explanation."""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        task_name = context.get("task_name", "unknown")
        task_desc = context.get("task_desc", "")[:2000]
        trajectories_json = context.get("trajectories_json", "[]")
        final_outcome = context.get("final_outcome", "unknown")

        return f"""### Task: {task_name}
### Final Outcome: {final_outcome}

### Task Description (abbreviated):
{task_desc}

### Trajectory Records:
{trajectories_json}

Analyze ALL rounds above. Extract reusable skills.
- If outcome is SUCCESS: focus on the winning strategy from the final successful round, and general patterns that led to success.
- If outcome is FAILURE: focus on repeated mistakes, avoidable patterns, and lessons learned across rounds.

Output a JSON array of extracted skills. Output ONLY valid JSON -- no markdown, no explanation."""

    def generate_merge(self, existing_skill: dict, new_skill: dict) -> str:
        """LLM-based merge of two similar skills into one improved skill."""
        prompt = f"""You are merging two similar skills into one improved skill.

### Existing Skill:
Title: {existing_skill.get('title', '')}
Instructions:
{existing_skill.get('instructions', '')}
Source Tasks: {existing_skill.get('source_tasks', [])}

### New Skill:
Title: {new_skill.get('title', '')}
Instructions:
{new_skill.get('instructions', '')}
Source Tasks: {new_skill.get('source_tasks', [])}

### Rules:
1. Combine insights from both skills into a single, improved skill.
2. Keep the more specific/actionable guidance from each.
3. Preserve information about which tasks validated this knowledge.
4. Output format: single JSON object with fields: title, description, instructions, tags

Output ONLY valid JSON -- no markdown, no explanation."""

        return self.call_llm(prompt, temperature=0.2, max_tokens=4000, max_loops=1)

    @staticmethod
    def parse_skills_output(raw: str) -> list:
        """Parse the LLM output into a list of skill dicts."""
        # Try to extract JSON array from the response
        # First, try direct parse
        text = raw.strip()
        if text.startswith("```"):
            # Remove markdown code fence
            text = re.sub(r"```(?:json)?\n?", "", text).strip()
            text = text.rstrip("`").strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in the text
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def parse_merge_output(raw: str) -> dict:
        """Parse the LLM merge output into a skill dict."""
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\n?", "", text).strip()
            text = text.rstrip("`").strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}
