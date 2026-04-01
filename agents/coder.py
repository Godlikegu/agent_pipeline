
from typing import Any, Dict
import re
from .base import BaseAgent
from utils.code_editor import CodeEditor


#  """You are a Senior Python Developer (Scientific Computing).
#     Your Goal: Implement or modify a SPECIFIC code segment based on current context.

#     Critical Rules:
#     1. Input Context ALWAYS includes:
#     - Package dependencies (what libraries are available)
#     - Current full code state (what already exists)
#     - Precise modification target (function/class/imports/etc.)
#     - Implementation plan (math/logic requirements)
#     2. Output: Return ONLY the code segment to be inserted/replaced - NO explanations, NO markdown.
#     3. Data Flow Integrity:
#     - PRESERVE all existing `print(shape)` debugging statements
#     - ADD new shape checks when tensor operations change dimensions (reshape/permute/concat)
#     4. Type Safety: Explicitly handle dtype conversions (e.g., `.float()` for float32 consistency)
#     5. NEVER output the entire file - ONLY the modified segment matching the target type.
#     6. **STRICT CONFIG TYPING**:
#     - When implementing `__init__` or other methods that take `config` as an argument, DO NOT use the type hint `config: Config` if `Config` is defined in the same file but later, or if it creates a circular reference.
#     - Use `config: 'Config'` (string forward reference) or `config: Any` or just `config` (no type hint) to avoid `NameError: name 'Config' is not defined`.
#     - **Reason**: The `Config` class might not be fully defined when `InverseSolver` is compiled, or `from __future__ import annotations` is missing.
#     7. **MANDATORY**: You MUST write the actual implementation logic. DO NOT return `pass` or `TODO` comments. You are the IMPLEMENTER. Replace existing `pass` with real code."""

class CoderAgent(BaseAgent):
    def _strip_markdown(self, code: str) -> str:
        # Extract content inside ```python ... ``` or ``` ... ```
        code_block_pattern = re.compile(r'```(?:python)?\n(.*?)\n```', re.DOTALL)
        matches = code_block_pattern.findall(code)
        if matches:
            return max(matches, key=len).strip()

        # Fallback for simple stripping if no blocks found (legacy behavior)
        code = re.sub(r'```python\n', '', code)
        code = re.sub(r'```\n', '', code)
        code = re.sub(r'\n```', '', code)
        return code.strip()

    def _build_system_prompt(self) -> str:
        return """You are a Senior Python Developer (Scientific Computing).
                Your Goal: Implement or modify a SPECIFIC code segment based on current context.

                Critical Rules:
                1. Input Context ALWAYS includes:
                - Package dependencies (what libraries are available)
                - Current full code state (what already exists)
                - Precise modification target (function/class/imports/etc.)
                - Implementation plan (math/logic requirements)
                2. Output: Return ONLY the code segment to be inserted/replaced - NO explanations, NO markdown.
                3. Type Safety: Explicitly handle dtype conversions (e.g., `.float()` for float32 consistency)
                4. NEVER output the entire file - ONLY the modified segment matching the target type.
                5. **MANDATORY**: You MUST write the actual implementation logic. DO NOT return `pass` or `TODO` comments. You are the IMPLEMENTER. Replace existing `pass` with real code.
                6. **DATA FILES**: Load input data from `dataset/` directory (e.g., `np.load('dataset/raw_data.npz')` and access keys). Load physical parameters from `dataset/meta_data.json`. Save the final reconstruction to `output.npy`. Do NOT assume any files exist outside `dataset/`.
                7. **SELF-CONTAINED**: Your code must be fully self-contained. Do NOT import from local project files. Only use installed pip packages and standard library.
                8. **__init__ IMPLEMENTATION**: When implementing __init__, you MUST set all instance variables. NEVER leave __init__ with just `pass`.
                9. **API COMPATIBILITY**: When using library functions/classes, do NOT pass keyword arguments you are unsure about. Use only well-documented, standard parameters. If a function call fails with 'unexpected keyword argument', remove that kwarg. Wrap risky API calls in try/except and have a working fallback.
                10. **CLASS STRUCTURE**: All class definitions must be at the top level of the file (module scope). NEVER nest one class definition inside another class's method. Check indentation carefully.
                11. **ORCHESTRATOR METHODS**: When a class has a setup/initialization method (e.g., `_setup`, `initialize`, `_precompute`) that is called by `__init__` or other methods, it MUST contain actual calls to all necessary helper methods. NEVER leave an orchestrator method as just `pass` or TODO comments while the helper methods it should call are already implemented. Verify the call chain is complete: `__init__` → `_setup()` → `[_helper1(), _helper2(), ...]`.
                12. **PERFORMANCE**: Do NOT use `torch.utils.checkpoint` (gradient checkpointing) unless the code explicitly runs out of GPU memory. Checkpointing doubles compute time by re-running forward pass during backward. For most inverse problems, standard autograd is fast enough and preferred. Keep the forward loop simple: plain Python for-loop with in-place FFT operations."""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        # 统一提取关键上下文
        package_list = context.get('package_list', 'No specific packages mentioned')
        current_code = context.get('current_code', '# No current code provided')
        plan = context.get('plan', 'No implementation plan provided')
        task_desc = context.get('task_desc', '')
        feedback = context.get('feedback', '')
        target_type = context.get('target_type', 'function')
        target_name = context.get('target_function', context.get('target_name', target_type))

        # Extract Skill Section if provided in dedicated field
        skill_section = ""
        skill_instruction = ""

        if context.get('knowledge_context'):
            skill_section = context['knowledge_context']
            skill_block = f"\n\n{skill_section}\n(Use these skills to guide your implementation style/logic)"
            skill_instruction = "\n### 🧠 SKILL UTILIZATION\nReview the 'RELEVANT SKILLS' above.\n"
            skill_instruction += "- If a skill provides a specific implementation pattern relevant to the current plan, you SHOULD follow it.\n"
            skill_instruction += "- However, if the skill contradicts the specific requirements of the current plan or task, PRIORITIZE the current Plan/Task over the skill.\n"
            skill_instruction += "- Skills are historical references, not absolute laws. Use judgment."

        # Legacy Fallback
        elif "### 🧠 RELEVANT SKILLS" in task_desc:
            # Extract until the next major header or end of relevant section
            try:
                start_idx = task_desc.find("### 🧠 RELEVANT SKILLS")
                # Assume skills block ends at next "###" or double newline if not found
                # But task_desc usually starts with skills now.
                # Let's just take the first 1000 chars as a heuristic for skills context
                # or split by double newline.
                skill_part = task_desc[start_idx:]
                # Find next header (e.g. "### Task Description" which might be original)
                # The injection prepends: skills + "\n\n" + original_desc
                # So we can split by "\n\n" and take the first part
                skill_section = skill_part.split("\n\n###")[0]
                if len(skill_section) > 2000: skill_section = skill_section[:2000] + "..."

                skill_block = f"\n\n{skill_section}\n(Use these skills to guide your implementation style/logic)"
                skill_instruction = "\n### 🧠 SKILL UTILIZATION\nReview the 'RELEVANT SKILLS' above.\n"
                skill_instruction += "- If a skill provides a specific implementation pattern relevant to the current plan, you SHOULD follow it.\n"
                skill_instruction += "- However, if the skill contradicts the specific requirements of the current plan or task, PRIORITIZE the current Plan/Task over the skill.\n"
                skill_instruction += "- Skills are historical references, not absolute laws. Use judgment."
            except:
                skill_block = ""
                skill_instruction = ""
        else:
            skill_block = ""
            skill_instruction = ""

        # 构建任务描述
        if target_type == 'function':
            task_desc = f"Implement the function `{target_name}`"
        elif target_type == 'class':
            task_desc = f"Implement the class `{target_name}`"
        elif target_type == 'imports':
            task_desc = "Replace ONLY the import statements at the top of the file"
        elif target_type == 'main_block':
            data_layout = context.get('data_layout', 'dataset/raw_data.npz (input), dataset/meta_data.json (parameters)')
            task_desc = f"Replace ONLY the code inside `if __name__ == '__main__':` block. Data layout: {data_layout}. SAVE result to `output.npy`."

        else:
            data_layout = context.get('data_layout', '')
            if 'full_rewrite' in target_type:
                task_desc = f"Rewrite the ENTIRE solver file from scratch. Data layout: {data_layout}"
            else:
                task_desc = f"Modify the `{target_type}` segment"

        # 构建反馈部分（如果有），并合并 failure_history（去重）
        failure_history = context.get('failure_history', '')
        fb_str = ''
        if isinstance(feedback, dict):
            fb_str = feedback.get('analysis') or feedback.get('full_judgement_analysis') or feedback.get('feedback') or str(feedback)
        elif feedback:
            fb_str = str(feedback)

        if failure_history:
            # If feedback text already appears in failure history, avoid duplication
            if fb_str and fb_str in failure_history:
                feedback_section = f"\n\n{failure_history}"
            else:
                feedback_section = f"\n\n{failure_history}\n\nPrevious Feedback:\n{fb_str}" if fb_str else f"\n\n{failure_history}"
        else:
            feedback_section = f"\n\nPrevious Feedback:\n{fb_str}" if fb_str else ""

        # Shape constraints section
        shape_section = ""
        input_shape = context.get('input_shape')
        output_shape = context.get('output_shape')
        if input_shape or output_shape:
            shape_section = "\n⚠️ CRITICAL SHAPE CONSTRAINTS:\n"
            if input_shape:
                shape_section += f"- Input data shape: {input_shape}\n"
            if output_shape:
                shape_section += f"- Output MUST have shape: {output_shape} (match ground truth EXACTLY)\n"
                shape_section += f"- If your result has a different shape, RESIZE/RESHAPE it to {output_shape} before np.save('output.npy', result)\n"

        return f"""Package Dependencies Available:
{package_list}

Implementation Plan:
{plan}
{skill_block}

Current Full Code State (DO NOT duplicate this entire code):
```python
{current_code}
```

Task: {task_desc}
{feedback_section}
{skill_instruction}
{shape_section}

Output ONLY the exact code segment to replace/insert - nothing else.
CRITICAL REMINDERS:
- Data layout: {context.get('data_layout', 'dataset/raw_data.npz for input, dataset/meta_data.json for parameters')}. Save final result to `output.npy`.
- Only use files listed in the data layout above. Do NOT assume other files exist.
- __init__ MUST have real implementation with self.xxx = ... (NEVER just pass).
- All numerical arrays should be np.float64 or np.float32 (avoid object arrays).
- HYPERPARAMETER FIDELITY: Use EXACTLY the hyperparameter values specified in the Plan. Do NOT add adaptive/warm-start features not in the Plan.
- SIGN CORRECTNESS: Double-check signs in gradient updates and proximal operators. If plan says `x = z - tau * grad`, code MUST use subtraction, not addition.
- IMPORT SAFETY: Before using any library API, verify the function/class exists. Use try/except ImportError for optional imports. Do NOT use keyword arguments unless you are certain they exist in the installed version.
- NUMERICAL PRECISION: For physics-based forward models with iterative propagation (many sequential steps), ALWAYS use float64/complex128 (torch.float64 / torch.complex128). NEVER use float32/complex64 — it causes NaN after ~100 FFT steps. This is NON-NEGOTIABLE.
- LEARNING RATE: If the Plan specifies a fixed learning rate value (e.g., lr=50), use EXACTLY that value. Do NOT implement learning rate auto-calibration, adaptive scheduling, or gradient-based lr estimation. Use a simple fixed lr with `x -= lr * grad`.
- SANITY CHECK: If the plan specifies a zero-contrast or trivial-input verification step, you MUST implement it. Print the result so the user can verify correctness before optimization begins.
"""

    def implement_and_merge(self, context: Dict[str, Any]) -> str:
        """
        Enhanced implementation with full context preservation and retry logic for syntax errors.
        """
        target_type = context.get('target_type', 'function')
        target_name = context.get('target_function', '') if target_type == 'function' else ''

        # FIX: Prioritize current_full_code (evolving state), fallback to skeleton (initial state)
        full_code = context.get('current_full_code')
        if not full_code or not full_code.strip():
            full_code = context['skeleton_code']

        # 构建统一的上下文（所有类型都包含关键信息）
        context_for_llm = {
            'package_list': context.get('package_list', 'torch, numpy, math'),
            'plan': context['plan'],
            'task_desc': context.get('task_desc', ''),
            'current_code': full_code,
            'target_type': target_type,
            'target_function': target_name,
            'feedback': context.get('feedback', ''),
            'knowledge_context': context.get('knowledge_context', ''),
        }

        # 生成代码（LLM 基于完整上下文生成）
        max_retries = 3

        for attempt in range(max_retries):
            raw_response = self.generate(context_for_llm)
            new_code = self._strip_markdown(raw_response)

            # Debug Log
            print(f"[CoderAgent] Generated code for {target_name or target_type} (Attempt {attempt+1}):\n{new_code[:100]}...")

            # 1. Validation: Check for 'pass' or 'TODO'
            is_invalid = False
            if target_type in ['function', 'class', 'main_block']:
                if "TODO" in new_code:
                    print(f"[CoderAgent] Detected 'TODO' in output.")
                    is_invalid = True

                # Check for "pass" only body (ignoring docstrings)
                if re.search(r'^\s*pass\s*$', new_code, re.MULTILINE):
                     print(f"[CoderAgent] Detected 'pass' statement in output.")
                     is_invalid = True

            if is_invalid:
                if attempt < max_retries - 1:
                    print(f"[CoderAgent] Rejecting output. Retrying with stricter prompt...")
                    current_feedback = context_for_llm.get('feedback')
                    if current_feedback is None:
                        current_feedback = ""

                    context_for_llm['feedback'] = (
                        current_feedback +
                        f"\n\nCRITICAL ERROR IN PREVIOUS ATTEMPT (Attempt {attempt+1}): "
                        "You returned code containing 'TODO' or 'pass'. "
                        "You MUST implement the full logic. Do not use placeholders. "
                        "WRITE THE ACTUAL CODE."
                    )
                    continue

            # 2. Validation: Try to Merge and Check Syntax
            print(f"[CoderAgent] Editing type: {target_type} | Target: {target_name or target_type}")
            try:
                result = ""
                if target_type == 'imports':
                    result = CodeEditor.replace_imports(full_code, new_code)
                elif target_type == 'main_block':
                    result = CodeEditor.replace_main_block(full_code, new_code)
                elif target_type == 'full_rewrite':
                    result = new_code
                elif target_type == 'class':
                    result = CodeEditor.replace_class(full_code, target_name, new_code)
                else:  # function (default)
                    result = CodeEditor.replace_function(full_code, target_name, new_code)

                # Verify non-empty
                if not result.strip():
                    raise ValueError("Code merge resulted in empty output")

                # If we get here, everything is good!
                return result

            except ValueError as e:
                # Syntax/Validation error from CodeEditor
                print(f"[CoderAgent] Merge/Syntax Error (Attempt {attempt+1}): {e}")

                if attempt < max_retries - 1:
                    # Retry with feedback
                    current_feedback = context_for_llm.get('feedback')
                    if current_feedback is None:
                        current_feedback = ""

                    context_for_llm['feedback'] = (
                        current_feedback +
                        f"\n\nCRITICAL SYNTAX ERROR IN PREVIOUS ATTEMPT (Attempt {attempt+1}): "
                        f"Your code caused a syntax/indentation error when merged: {e}. "
                        "Please fix indentation and syntax. Ensure you use 4 spaces for indentation and correct Python syntax."
                    )
                    continue
                else:
                    print(f"[CoderAgent] Max retries reached. Returning original code.")
                    return full_code

            except Exception as e:
                print(f"[CoderAgent] Unexpected error during merge: {e}")
                # For unexpected errors, retrying might not help, return original
                return full_code

        # Fallback (should not be reached if loop logic is correct)
        return full_code
