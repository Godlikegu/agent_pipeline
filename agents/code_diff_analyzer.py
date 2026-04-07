"""
agents/code_diff_analyzer.py -- Compare pipeline-generated solver against reference code.

Produces a structured analysis report identifying critical implementation differences
that likely cause task failure, and minor differences that are less impactful.
"""
import os
from typing import Any, Dict, List

from .base import BaseAgent


# Files to exclude from reference code loading (not relevant to solver logic)
_EXCLUDE_FILES = {"visualization.py", "generate_data.py", "__init__.py"}
# Safety cap per file to prevent extremely long individual files from dominating
_PER_FILE_CAP = 15000


def load_reference_code(task_dir: str) -> str:
    """Load reference code from task_dir/main.py + task_dir/src/*.py.

    Excludes visualization, data generation, and __init__ files.
    Returns concatenated code with file-path headers.
    """
    task_dir = os.path.abspath(task_dir)
    parts: List[str] = []

    # 1. main.py
    main_path = os.path.join(task_dir, "main.py")
    if os.path.isfile(main_path):
        content = _read_file(main_path)
        parts.append(f"### FILE: main.py\n{content}")

    # 2. src/*.py (sorted for determinism)
    src_dir = os.path.join(task_dir, "src")
    if os.path.isdir(src_dir):
        for fname in sorted(os.listdir(src_dir)):
            if not fname.endswith(".py"):
                continue
            if fname in _EXCLUDE_FILES:
                continue
            fpath = os.path.join(src_dir, fname)
            if os.path.isfile(fpath):
                content = _read_file(fpath)
                parts.append(f"### FILE: src/{fname}\n{content}")

    return "\n\n".join(parts) if parts else ""


def _read_file(path: str) -> str:
    """Read a file with a per-file safety cap."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > _PER_FILE_CAP:
            content = content[:_PER_FILE_CAP] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"(Error reading file: {e})"


class CodeDiffAnalyzerAgent(BaseAgent):
    """Compares generated solver code against reference implementation."""

    def __init__(self, client: Any, model_name: str, temperature: float = 0.2):
        super().__init__(client, model_name, temperature=temperature)

    def _build_system_prompt(self) -> str:
        return """You are a Code Analysis Expert for scientific computing inverse-problem pipelines.

Your task: Compare an AI-generated solver against the reference (ground-truth) implementation and identify implementation differences that explain why the solver may produce incorrect results.

### Analysis Focus:
1. **Algorithm choice** — Is the solver using the correct algorithm / mathematical formulation?
2. **Physics / forward model** — Are physical operators, FFT conventions, coordinate systems correct?
3. **Hyperparameters** — Learning rates, iteration counts, regularization weights, thresholds.
4. **Data handling** — Loading, preprocessing, normalization, dtype, shape conventions.
5. **Key formulas** — Gradient computation, update rules, loss functions, sign conventions.
6. **Missing steps** — Steps present in reference but absent in solver (e.g., post-processing, constraints).
7. **Implicit domain knowledge** — Conventions and principles that domain experts know intuitively
   but are NOT stated in task descriptions. For each critical issue, identify the UNDERLYING PRINCIPLE
   that the solver violates. Focus on subtle distinctions that look equivalent but produce different results:
   - Rounding modes (truncation vs rounding when snapping to discrete grids)
   - Operation ordering (e.g., clamp-then-scale vs scale-then-clamp changes function shape)
   - Default parameter choices (e.g., zero regularization vs small regularization has different bias properties)
   - Normalization conventions (sum/n vs mean, per-pixel vs per-sample)

### Output Format (strict JSON):
{
  "critical_issues": [
    {
      "location": "function or step name",
      "description": "What is different and why it causes failure",
      "reference_approach": "What the reference code does (be specific: formula, value, convention)",
      "solver_approach": "What the generated solver does instead",
      "implicit_knowledge": "The domain principle or convention violated — explain WHY the reference approach is correct (the reasoning a domain expert would give), not just WHAT is different. This should be a transferable insight that helps avoid the same mistake in any implementation."
    }
  ],
  "minor_issues": [
    {
      "location": "function or step name",
      "description": "Difference that is unlikely to cause failure but worth noting"
    }
  ],
  "summary": "One paragraph: the most likely root causes of failure, ranked by importance"
}

### Rules:
1. Focus on SEMANTIC differences, not stylistic ones (variable naming, comments, formatting).
2. Be SPECIFIC — cite exact values, formulas, line references where possible.
3. Order critical_issues by estimated impact (most impactful first).
4. If the solver uses a completely different algorithm, that is the #1 critical issue.
5. Output ONLY valid JSON — no markdown fence, no explanation outside the JSON."""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        solver_code = context.get("solver_code", "")
        reference_code = context.get("reference_code", "")
        task_desc = context.get("task_desc", "")
        execution_error = context.get("execution_error", "")
        metrics = context.get("metrics", None)

        prompt = f"""### Task Description:
{task_desc}

### Generated Solver Code:
```python
{solver_code}
```

### Reference Implementation:
{reference_code}
"""
        if execution_error:
            prompt += f"\n### Execution Error (stderr):\n{execution_error}\n"
        if metrics:
            prompt += f"\n### Evaluation Metrics: {metrics}\n"

        prompt += """
Analyze the differences between the generated solver and the reference implementation.
Identify critical issues that explain poor results and minor issues of lesser impact.
Output ONLY valid JSON."""
        return prompt
