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
        return """You are a Skills Extraction Specialist for a scientific computing agent pipeline that solves inverse problems (deconvolution, reconstruction, etc.).

Your mission: Analyze trajectory records from pipeline executions and extract **highly specific, actionable** knowledge skills that will directly improve future runs.

### Skill Categories:
1. **knowledge_general** -- Cross-task reusable patterns with SPECIFIC details. Examples:
   - "FISTA with TV regularization: use step_size=1/(L+lambda), lambda=0.001-0.01, 200-500 iterations for convergence"
   - "For FFT-based convolution: zero-pad to 2x size, use rfft2/irfft2, normalize by dividing by product of padded dimensions"
   - "When PSNR < baseline after deconvolution: reduce regularization weight by 10x, the solver is over-regularizing"

2. **knowledge_task_specific** -- Validated strategies WITH exact hyperparameters:
   - "For SIM task: ADMM with rho=0.5, TV_weight=0.001, 300 iterations achieves PSNR>23"
   - "Wiener filter diverges for this noise level; use FISTA with lambda=1e-4 instead"

3. **code** -- Verified code patterns with complete implementation:
   - Include the actual working code snippet (5-20 lines)
   - "Safe numpy loading: raw = np.load(path, allow_pickle=True); data = raw.item() if raw.ndim == 0 else raw"

### Scope Assignment:
- scope="Planner" -- Algorithm selection, mathematical formulation, hyperparameter choices
- scope="Coder" -- Implementation patterns, API usage, numerical stability tricks
- scope="General" -- Applicable to both

### CRITICAL: What makes a GOOD skill vs BAD skill:
GOOD: "Use FISTA with step_size=1/L where L=||H^T H||, lambda_tv=0.001, 300 iterations. Start from zero initialization, not from input data."
BAD: "Use an iterative optimization algorithm with appropriate parameters."

GOOD: "Sign convention: gradient update must be x_new = x - step*grad (subtraction). Using addition causes divergence."
BAD: "Be careful with signs in gradient updates."

GOOD (code): "FFT convolution with proper padding:\\n```python\\npad_h, pad_w = h + kh - 1, w + kw - 1\\nH = np.fft.rfft2(kernel, s=(pad_h, pad_w))\\nX = np.fft.rfft2(image, s=(pad_h, pad_w))\\nresult = np.fft.irfft2(H * X)[:h, :w]\\n```"
BAD: "Use FFT for convolution."

### Output Format (strict JSON array):
[
  {
    "title": "Concise but specific skill title",
    "description": "One-sentence: when to use this skill and what problem it solves",
    "category": "knowledge_general|knowledge_task_specific|code",
    "scope": "General|Planner|Coder",
    "instructions": "## When to use\\n- Specific trigger conditions\\n\\n## Key insight\\n- The actual actionable knowledge with EXACT numbers/formulas/code\\n\\n## Constraints\\n- When NOT to use this, edge cases",
    "tags": ["tag1", "tag2"]
  }
]

### Rules:
1. Extract 1-5 skills per analysis. Quality over quantity.
2. For SUCCESS trajectories: Extract the WINNING strategy -- exact algorithm, hyperparameters, and implementation that worked.
3. For FAILURE trajectories: Extract the ROOT CAUSE with specific details -- what parameter was wrong, what sign was flipped, what API was misused.
4. EVERY skill MUST include specific numbers, formulas, or code. No vague advice.
5. Do NOT extract trivial observations (e.g., "code should not have syntax errors").
6. If a success trajectory shows iteration-2 fix of iteration-1 failure, extract BOTH the failure lesson AND the winning fix.
7. Output ONLY valid JSON -- no markdown fence, no explanation."""

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
