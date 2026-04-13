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
- scope="General" -- **DEFAULT choice.** Use this for skills that contain BOTH algorithmic insight AND specific implementation details (formulas, code snippets, coordinate conventions, operator definitions). Most skills from code diff analysis should be "General" because they describe WHAT to implement AND HOW.
- scope="Planner" -- ONLY for pure high-level strategy skills with NO specific formulas or code (e.g., "use ADMM instead of gradient descent for this problem type")
- scope="Coder" -- ONLY for pure implementation tricks with no algorithmic insight (e.g., "use try/except for optional imports")

### CRITICAL: What makes a GOOD skill vs BAD skill:

Skills must capture IMPLICIT KNOWLEDGE — the reasoning and principles that domain experts
know intuitively but never state explicitly. The goal is to guide the agent's THINKING,
not dictate its code. The agent that receives these skills will NOT see the reference code.

GOOD: "When snapping continuous frequencies to a discrete FFT grid, use truncation (floor toward zero),
not rounding. Rounding can snap to a neighboring bin, creating a systematic mismatch between the
forward model and the measurement data. This mismatch corrupts gradients across all angles and
degrades reconstruction quality by NCC~0.3-0.4."
BAD: "Use int(c_a * norm_x) / norm_x to snap frequencies."
(Gives code but no reasoning — agent may 'improve' int() to round() thinking they're equivalent)

GOOD: "A smooth sigmoid cutoff requires its exp() argument to reach large magnitudes (>>1 inside
the passband, <<-1 outside). If the argument is clamped to a small range before reaching exp(),
the sigmoid saturates at ~0.5 instead of ~1.0, halving all amplitudes and breaking the forward model.
Ensure that any clamping preserves the dynamic range of the argument to exp()."
BAD: "Use torch.exp(torch.clamp(c_gamma - cutoff, max=0.01) * 10000) for the pupil."
(Shows exact code — agent may reorder operations without understanding WHY the order matters)

GOOD: "Regularization (TV, L2, etc.) introduces reconstruction bias that reduces correlation with
ground truth. Unless the task explicitly requests regularization or initial results show severe
noise/artifacts, start with zero regularization weight. Adding even small regularization (1e-4)
smooths fine features and reduces NCC."
BAD: "Use TV weight 1e-4 to 1e-3."
(Gives a number without explaining the principle of when/why to use it)

GOOD: "Sign convention: gradient update must be x_new = x - step*grad (subtraction). Using addition causes divergence."
BAD: "Be careful with signs in gradient updates."

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
1. Extract 3-6 skills per analysis. Each critical issue from code diff deserves its own skill — do NOT merge distinct issues.
2. For SUCCESS trajectories: Extract the WINNING strategy -- exact algorithm, hyperparameters, and implementation that worked.
3. For FAILURE trajectories: Extract the ROOT CAUSE with specific details -- what parameter was wrong, what sign was flipped, what API was misused. If a code diff analysis report is provided, use it as the PRIMARY source for identifying root causes.
4. EVERY skill MUST include specific numbers, formulas, or code. No vague advice.
5. Do NOT extract trivial observations (e.g., "code should not have syntax errors").
6. If a success trajectory shows iteration-2 fix of iteration-1 failure, extract BOTH the failure lesson AND the winning fix.
7. Output ONLY valid JSON -- no markdown fence, no explanation."""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        task_name = context.get("task_name", "unknown")
        task_desc = context.get("task_desc", "")
        trajectories_json = context.get("trajectories_json", "[]")
        final_outcome = context.get("final_outcome", "unknown")
        code_diff_report = context.get("code_diff_report", "")

        prompt = f"""### Task: {task_name}
### Final Outcome: {final_outcome}

### Task Description:
{task_desc}

### Trajectory Records:
{trajectories_json}
"""

        if code_diff_report:
            prompt += f"""
### Code Diff Analysis Report (reference implementation vs generated solver):
{code_diff_report}

Use this diff report as the PRIMARY source for understanding WHY the task failed/succeeded.
Focus especially on the `implicit_knowledge` field of each critical issue — this captures the
domain principles that the generated solver violated.

Extract skills that teach the PRINCIPLE behind each difference, not the specific code.
The agent that receives these skills must understand WHY a certain approach is correct,
so it can implement it correctly even without seeing the reference code.
Ask yourself: "What would a domain expert tell a competent programmer who keeps making this mistake?"
The answer is the skill content.

DO NOT copy code from the reference implementation into skills. Instead, explain the reasoning
that would lead a programmer to write the correct code independently.
"""

        prompt += """
Analyze ALL rounds above. Extract reusable skills.
- If outcome is SUCCESS: focus on the winning strategy from the final successful round, and general patterns that led to success.
- If outcome is FAILURE: focus on repeated mistakes, avoidable patterns, and lessons learned across rounds.

Extract 3-6 skills. Each code diff issue should become its own separate skill — do NOT compress multiple issues into one.
Output a JSON array of extracted skills. Output ONLY valid JSON -- no markdown, no explanation."""
        return prompt

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

        return self.call_llm(prompt, temperature=0.2, max_tokens=32768, max_loops=1)

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
