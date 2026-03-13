
from typing import Any, Dict
from .base import BaseAgent

# 4. **Shape Auditing**: You MUST insert `print(f'DEBUG: {func_name} input shape: {x.shape}')` at the start of every method.

class ArchitectAgent(BaseAgent):
    def _build_system_prompt(self) -> str:
        return """You are a Senior Software Architect.
                Your Goal: Design the Python Class Structure (Skeleton) for the Planner's algorithm.
                You DO NOT write the logic inside functions. You define Interfaces.

                Crucial Rules:
                1. Define a class `InverseSolver`.
                2. Define `__init__`, `forward`, and `solve` methods.
                3. **Type Hinting**: All arguments must have type hints.
                4. Leave implementation empty using `pass` or `# TODO: Implement ...`.
                5. **Strict Output Format**:
                   - Output ONLY valid Python code in a markdown block.
                   - Do NOT wrap the code in a JSON string (e.g., no "import...").
                   - Do NOT return a JSON object.
                   - NO conversational text before or after the code.
                6. Do NOT define a `Config` class. Store all hyperparameters (lr, shape, iterations, etc.) as instance attributes in `InverseSolver.__init__`.

                Rule: Your code structure MUST be strictly modular:
                Imports: All imports at the top.
                Solver Class: class InverseSolver:
                Main: if __name__ == "__main__":
                      - Must include `np.load('dataset/input.npy')`
                      - Must include `solver.solve(...)`
                      - Must include `np.save('output.npy', result)`

                Output Format:
                ```python
                import ...

                class InverseSolver:
                    def __init__(self, ...):
                        pass
                    ...

                if __name__ == "__main__":
                    # Load Data
                    # input_data = np.load('dataset/input.npy')
                    # ...
                    # result = solver.solve(input_data)
                    # np.save('output.npy', result)
                    pass
                ```
                """

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        prompt = f"""### TASK
                {context['task_desc']}

                ### PLAN TO IMPLEMENT
                {context['plan']}"""

        # Explicitly prompt to use injected skills if provided in dedicated field
        if context.get('knowledge_context'):
            prompt += "\n\n" + context['knowledge_context'] + "\n"
            prompt += "\n### 🧠 SKILL UTILIZATION\n"
            prompt += "The section above contains 'RELEVANT SKILLS' from past experiences.\n"
            prompt += "1. **Check Applicability**: Ensure the skill's architectural suggestions (e.g., pre-computing kernels) fit the current problem.\n"
            prompt += "2. **Support**: If applicable, add necessary attributes in `__init__` or helper methods to support the skill.\n"
            prompt += "3. **Ignore Mismatches**: If a skill suggests a structure that conflicts with the current Plan, follow the Plan.\n"

        # Legacy Fallback
        elif "RELEVANT SKILLS" in context.get('task_desc', ''):
            prompt += "\n\n### 🧠 SKILL UTILIZATION\n"
            prompt += "The Task Description includes 'RELEVANT SKILLS'.\n"
            prompt += "1. **Check Applicability**: Ensure the skill's architectural suggestions (e.g., pre-computing kernels) fit the current problem.\n"
            prompt += "2. **Support**: If applicable, add necessary attributes in `__init__` or helper methods to support the skill.\n"
            prompt += "3. **Ignore Mismatches**: If a skill suggests a structure that conflicts with the current Plan, follow the Plan.\n"

        if context.get("previous_skeleton"):
            prompt += f"""

                    ### PREVIOUS ARCHITECTURE (Iter {context.get('iteration', '?')})
                    {context['previous_skeleton']}

                    ### YOUR MISSION
                    - PRESERVE correct interfaces (e.g., method signatures that passed tests)
                    - ONLY modify parts flagged by Judge (see feedback below)
                    - Add missing type hints / asserts for shape validation
                    - DO NOT change working interfaces unless absolutely necessary"""

        # Failure history and feedback (deduplicated)
        failure_hist = context.get('failure_history')
        fb = context.get('feedback')
        fb_str = None
        if isinstance(fb, dict):
            fb_str = fb.get('analysis') or fb.get('full_judgement_analysis') or fb.get('feedback') or str(fb)
        elif fb:
            fb_str = str(fb)

        if failure_hist:
            prompt += f"\n\n### PAST FAILURES (Architect-relevant)\n{failure_hist}"
            if fb_str and fb_str not in failure_hist:
                prompt += f"\n\n### ADDITIONAL JUDGE FEEDBACK\n{fb_str}"
        else:
            if fb_str:
                prompt += f"\n\n### FEEDBACK FROM JUDGE\n{fb_str}"

        return prompt
