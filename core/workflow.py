import os
import sys
import shutil
import subprocess
import json
import time
import ast
from typing import List, Dict, Tuple, Any, Optional
import re, hashlib, datetime

from .workflow_base import InverseProblemBase
from utils.text_utils import extract_json, extract_python

class InverseProblemWorkflow(InverseProblemBase):
    def __init__(self, task_name: str, task_desc: str, gt_code_path: str, python_path: str, working_dir: str, client: Any, model_name: str, config: dict = None, root_output_dir: str = None, skill_manager: Any = None, max_retries: int = None):
        super().__init__(task_name, task_desc, gt_code_path, python_path, working_dir, client, model_name, config, root_output_dir, skill_manager, max_retries)

    def run(self):
        from .sandbox import setup_sandbox
        from .executor import phase_0_preparation

        setup_sandbox(self.sandbox_dir, self.gt_code_path, self._log)

        input_shape, output_shape, baseline_metrics = phase_0_preparation(
            sandbox_dir=self.sandbox_dir,
            python_path=self.python_path,
            task_desc=self.task_desc,
            package_list=self.package_list,
            data_gen_agent=self.data_gen_agent,
            eval_gen_agent=self.eval_gen_agent,
            write_file_fn=self._write_file,
            log_fn=self._log,
            data_gen_timeout=self.data_gen_timeout,
            gt_code_snippet_limit=self.gt_code_snippet_limit,
            syntax_check_timeout=self.syntax_check_timeout,
        )
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.baseline_metrics = baseline_metrics

        # ------------------------------------------------------------------
        # Skill Injection (Moved to per-agent context building)
        # ------------------------------------------------------------------
        self._log(">>> [Knowledge System] Hierarchical Memory Activated. Skills will be injected per-agent.")

        # 初始状态
        feedback = None
        ticket = "Planner" # 默认从 Plan 开始
        # action_sequence removed in favor of self.trajectory_steps

        while self.retry_count < self.max_retries:
            iter_id = self.retry_count + 1
            self._log(f"\n{'='*20} Iteration {iter_id} (Ticket: {ticket}) {'='*20}")


            # =========================================================
            # Stage 1: Planning (with Critic Loop)
            # =========================================================
            if ticket == "Planner":
                self._log(">>> [Agent] Planner...")
                # action_sequence.append(f"Iteration {iter_id}: Planner") removed

                # plan_ctx = {'task_desc': self.task_desc, 'feedback': feedback}
                plan_ctx = self._build_context_with_memory(
                    base_context={
                        'task_desc': self.task_desc,
                        'feedback': feedback,
                        'shape_info': f"Input shape: {self.input_shape}, Expected output shape: {self.output_shape}" if hasattr(self, 'input_shape') and self.input_shape else None
                    },
                    agent_role="Planner",
                    current_ticket="Planner"
                )

                draft_plan = self.planner.generate(plan_ctx)

                # --- Critic Loop ---
                # Planner 阶段的 Critic Loop（现有代码只需微调解析方式）
                critic_valid = False
                for critic_retry in range(3):
                    critic_resp_str = self.critic.generate({
                        'task_desc': self.task_desc,
                        'plan': draft_plan
                    })

                    try:
                        critic_resp = json.loads(critic_resp_str)  # 现在 100% 安全
                        if critic_resp["decision"] == "PASS":
                            critic_valid = True
                            break
                        else:
                            # 使用结构化字段构建反馈
                            feedback = f"Critic rejected: {critic_resp['reason']}"
                            if critic_resp.get("suggestion"):
                                feedback += f" | Fix: {critic_resp['suggestion']}"
                            draft_plan = self.planner.generate({'task_desc': self.task_desc, 'feedback': feedback})
                    except Exception as e:
                        # 理论上不会触发（因 CriticAgent 已保证 JSON 合法性）
                        self._log(f"[System] Critic JSON parse failed (unexpected): {e}")
                        break

                if not critic_valid:
                    self._log("[System] Critic rejected plan after max retries. Proceeding with caution...")

                self.current_plan = draft_plan

                # ✅ UPDATED RECORD STEP
                self._record_step(iter_id, "Planner", input_data=plan_ctx, output_data={"plan": self.current_plan})

                self._save_artifact(f"iter_{iter_id}_plan.md", self.current_plan)
                ticket = "Architect" # Pass baton

            # =========================================================
            # Stage 2: Architecture
            # =========================================================
            if ticket == "Architect":
                self._log(">>> [Agent] Architect...")
                # action_sequence.append(f"Iteration {iter_id}: Architect") removed

                # Custom Retrieval Query for Architect: Task + Plan
                architect_query = f"{self.task_desc}\n\nPLAN CONTEXT:\n{self.current_plan}"

                arch_ctx = self._build_context_with_memory(
                    base_context={
                        'task_desc': self.task_desc, # Still pass original for prompt
                        'plan': self.current_plan,
                        'previous_skeleton': self.current_skeleton if self.current_skeleton.strip() else None,
                        'feedback': feedback.get('feedback') if isinstance(feedback, dict) and feedback.get('ticket') == 'Architect' else None
                    },
                    agent_role="Architect",
                    current_ticket="Architect",
                    retrieval_query=architect_query # Pass custom query
                )

                # Architect 生成包含 Class 定义和 pass 方法的骨架
                for attempt in range(3):
                    arch_resp = self.architect.generate(arch_ctx)

                    # Try extracting JSON first (legacy support)
                    extracted_json = extract_json(arch_resp)
                    try:
                        arch_json = json.loads(extracted_json)
                        if isinstance(arch_json, dict) and 'skeleton_code' in arch_json:
                             self.current_skeleton = arch_json['skeleton_code']
                        else:
                             # Maybe it's just the code string
                             self.current_skeleton = arch_resp
                    except:
                        # Fallback: assume response is code (with markdown)
                        self.current_skeleton = arch_resp

                    # Ensure skeleton is clean code (Python)
                    self.current_skeleton = extract_python(self.current_skeleton)

                    # Validate
                    is_valid, err_msg = self._validate_skeleton(self.current_skeleton)
                    if is_valid:
                        self._log("  [System] Skeleton validated successfully.")
                        break
                    else:
                        self._log(f"  [System] Skeleton validation failed (Attempt {attempt+1}): {err_msg}")
                        # Debug: self._log first few lines of invalid content
                        debug_snippet = self.current_skeleton[:300].replace('\n', '\\n')
                        self._log(f"  [Debug] Invalid Content Start: {debug_snippet}...")
                        arch_ctx['feedback'] = f"Previous skeleton was invalid. Error: {err_msg}\nPlease output ONLY valid Python code containing `class Config` and `class InverseSolver`."
                else:
                    raise RuntimeError("Failed to generate valid skeleton after 3 attempts.")

                # [Hard Logic] Parse function list using AST (No Hallucination)
                self.function_list = self._parse_functions_from_skeleton(self.current_skeleton)
                print(f"[System] Extracted Function List: {self.function_list}")

                # Initialize current_code with the skeleton so we start with a valid structure
                self.current_code = self.current_skeleton

                # ✅ UPDATED RECORD STEP
                self._record_step(iter_id, "Architect", input_data=arch_ctx, output_data={"skeleton": self.current_skeleton})

                self._save_artifact(f"iter_{iter_id}_skeleton.py", self.current_skeleton)
                ticket = "Coder" # Pass baton

            # =========================================================
            # Stage 3: Coding (The Edit Loop)
            # =========================================================
            if ticket == "Coder":
                self._log(">>> [Agent] Coder...")
                # action_sequence.append(f"Iteration {iter_id}: Coder") removed

                # 1. 确定生成任务列表 (Task List)
                # 默认是全量生成 (Fresh Start)
                coding_tasks = [
                    ('imports', None),
                    *[( 'function', func_name ) for func_name in self.function_list],
                    ('main_block', None)
                ]

                # 2. 检查是否是"修补模式"
                # 如果有 feedback，且 feedback 指向了特定的函数或模块
                is_patch_mode = False
                if isinstance(feedback, dict) and feedback.get('ticket_assigned_to') == 'Coder':
                    # 假设 Judge 的 feedback 中包含 'fix_target' 字段
                    target = feedback.get('fix_target')

                    # Normalizing target: Handle "InverseSolver.solve method" -> "solve"
                    if target:
                        # Common patterns: "InverseSolver.solve", "solve method", "function solve"
                        match = re.search(r'\b(solve|__init__|forward|[_a-zA-Z0-9]+)\b', target)
                        if match:
                             # If the extracted name is in our function list (or is main/imports), use it
                             candidate = match.group(1)
                             if candidate in self.function_list or candidate in ['imports', 'main_block']:
                                 target = candidate
                             # Special case: "InverseSolver.solve" -> "solve"
                             elif "." in target:
                                 parts = target.split('.')
                                 if parts[-1] in self.function_list:
                                     target = parts[-1]

                    # 简单的推断逻辑：如果 analysis 里提到了具体的函数名
                    if not target:
                        for func in self.function_list:
                            if func in feedback.get('analysis', ''):
                                target = func
                                break

                    if target:
                        self._log(f"  [System] Smart Patch Mode Activated. Target: {target}")
                        is_patch_mode = True
                        # 只保留需要修改的目标
                        if target == 'imports':
                            coding_tasks = [('imports', None)]
                        elif target == 'main_block' or target == 'main':
                            coding_tasks = [('main_block', None)]
                        elif target in self.function_list:
                            coding_tasks = [('function', target)]
                        else:
                            # If ambiguous, fall back to full rebuild but maybe hint the Coder?
                            # For now, let's try to be smart about "solve method and main execution block"
                            # If multiple targets mentioned, we might need a list.
                            # BUT current architecture supports list of tasks.

                            # Improved: Check for multiple targets in string
                            detected_targets = []
                            if 'main' in str(target).lower() or 'execution' in str(target).lower():
                                detected_targets.append(('main_block', None))

                            for func in self.function_list:
                                if func in str(target):
                                    detected_targets.append(('function', func))

                            if detected_targets:
                                coding_tasks = detected_targets
                                self._log(f"  [System] Multi-target patch detected: {coding_tasks}")
                            else:
                                # 全量修复：按依赖顺序构建任务队列
                                self._log(f"  [System] Target '{target}' ambiguous or unknown. Performing full rebuild...")
                                coding_tasks = [
                                    ('imports', None),                                      # 1. 基础依赖
                                    *[
                                        ('function', func_name)
                                        for func_name in self.function_list
                                        if func_name not in {'imports', 'main_block'}  # 避免重复
                                    ],                                                      # 3. 所有业务函数
                                    ('main_block', None)                                    # 4. 入口逻辑
                                ]
                                self._log(f"  [System] Full rebuild tasks: {len(coding_tasks)} items")

                # 3. 执行 Coding Loop
                # Reset to skeleton ONLY if we are doing a full rebuild (Full Fresh Start)
                if not is_patch_mode:
                    print("  [System] Full rebuild mode: Resetting code to skeleton state.")
                    self.current_code = self.current_skeleton

                # --- CODE SIZE GUARDRAIL REMOVED PER USER REQUEST ---
                # The mechanism that forced a full rebuild when code exceeded 12000 chars has been disabled.
                # Coder is now allowed to patch large files, but be warned: context window overflow risks are higher.

                # --- STUCK DETECTION: If same error repeated 3+ times, force Planner rewrite ---
                if len(self.failure_history) >= 3:
                    recent_errors = [h.get('analysis', '')[:80] for h in self.failure_history[-3:]]
                    if len(set(recent_errors)) == 1:
                        self._log("  ⚠️ [STUCK DETECTION] Same error 3x in a row. Forcing Planner rewrite next iteration.")
                        # Clear failure history to break the loop
                        self.failure_history = self.failure_history[-1:]

                # --- STUCK DETECTION 2: If same ticket assigned 4+ times, escalate ---
                if len(self.failure_history) >= 4:
                    recent_tickets = [h.get('ticket_assigned_to', '') for h in self.failure_history[-4:]]
                    if len(set(recent_tickets)) == 1 and recent_tickets[0] == 'Coder':
                        self._log("  ⚠️ [STUCK DETECTION] Coder assigned 4x in a row. Escalating to Planner for new approach.")
                        self.failure_history[-1]['ticket_assigned_to'] = 'Planner'
                        self.failure_history[-1]['feedback'] = "The Coder has failed to fix the code after 4 attempts with the current algorithm. Please propose a COMPLETELY DIFFERENT and SIMPLER approach. Avoid the same library/method that keeps failing."

                for task_type, task_name in coding_tasks:
                    print(f"  [Coder] Processing {task_type}" + (f": {task_name}" if task_name else "") + "...")

                    # --- Custom Retrieval Query for Coder ---
                    # Focus on Function Signature + Task Name (Constraint) + Plan Summary

                    func_signature = None
                    if task_type == 'function' and self.current_skeleton:
                         # Use robust AST-based extraction
                         func_signature = self._extract_function_signature(self.current_skeleton, task_name)

                    # Extract Plan Summary (First 500 chars)
                    plan_summary = self.current_plan[:500] if self.current_plan else "No Plan Available"

                    if func_signature:
                        # Use precise signature + docstring + Plan
                        coder_query = f"Python Implementation for:\n{func_signature}\n\nContext: {self.task_name}\nPlan: {plan_summary}"
                    else:
                        # Fallback
                        coder_query = f"Python Implementation for function: {task_name if task_name else 'Global Scope'}\nContext: {self.task_name}\nPlan: {plan_summary}"

                    if is_patch_mode and feedback.get('analysis'):
                         coder_query += f"\nError Analysis: {feedback.get('analysis')}"

                    ctx = {
                        'target_type': task_type,
                        'skeleton_code': self.current_skeleton,
                        'current_full_code': self.current_code,      # ✅ 上一轮完整实现（自身历史）
                        'plan': self.current_plan,
                        'task_desc': self.task_desc,                 # ✅ Pass task_desc (with skills) to Coder
                        'package_list': self.package_list,
                        'feedback': feedback.get('feedback') if is_patch_mode and isinstance(feedback, dict) else None,
                        'analysis': feedback.get('analysis') if is_patch_mode and isinstance(feedback, dict) else None, # ✅ Inject Analysis
                        'fix_target': target if is_patch_mode else None,
                    }

                    if task_type == 'function':
                        ctx['target_function'] = task_name

                    # ====== 关键增强：为Coder注入历史失败 ======
                    ctx = self._build_context_with_memory(
                        base_context=ctx,
                        agent_role="Coder",
                        current_ticket="Coder",
                        retrieval_query=coder_query # Pass custom query
                    )

                    # 调用 Coder Agent
                    # implement_and_merge 内部会自动处理 EDIT 逻辑，保留其他代码不变
                    self.current_code = self.coder.implement_and_merge(ctx)

                    # ✅ UPDATED RECORD STEP
                    self._record_step(iter_id, "Coder", input_data=ctx, output_data={
                        "task_type": task_type,
                        "target": task_name,
                        "code": self.current_code
                    })

                    # 增量保存: 每完成一个任务，就更新 solver.py，允许用户实时查看进度
                    self.current_code = extract_python(self.current_code)
                    solver_path = os.path.join(self.sandbox_dir, "solver.py")
                    with open(solver_path, "w") as f:
                        f.write(self.current_code)

                # 4. 保存文件 (Final Save after loop)
                self.current_code = extract_python(self.current_code)

                # --- POST-CODING VALIDATION: Reject empty __init__ and detect nested classes ---
                self._log("  [System] Post-coding validation...")
                try:
                    import ast as _ast
                    _tree = _ast.parse(self.current_code)

                    # CHECK 1: Detect nested class definitions (classes inside other classes/functions)
                    for _node in _ast.walk(_tree):
                        if isinstance(_node, _ast.ClassDef):
                            for _child in _ast.walk(_node):
                                if isinstance(_child, _ast.ClassDef) and _child is not _node:
                                    # Found a class nested inside another class
                                    self._log(f"  ⚠️ [VALIDATION] Nested class detected: '{_child.name}' inside '{_node.name}'. Forcing full rewrite...")
                                    _nest_ctx = {
                                        'target_type': 'full_rewrite',
                                        'skeleton_code': self.current_skeleton,
                                        'current_full_code': self.current_skeleton,
                                        'plan': self.current_plan,
                                        'task_desc': self.task_desc,
                                        'package_list': self.package_list,
                                        'feedback': f"CRITICAL STRUCTURE ERROR: Class '{_child.name}' is nested INSIDE class '{_node.name}'. ALL classes must be at module top-level. Fix the indentation so each class is at column 0. Do NOT put class definitions inside methods or other classes."
                                    }
                                    _nest_ctx = self._build_context_with_memory(_nest_ctx, "Coder", "Coder")
                                    self.current_code = self.coder.implement_and_merge(_nest_ctx)
                                    self.current_code = extract_python(self.current_code)
                                    break
                            else:
                                continue
                            break

                    # Re-parse after potential fix
                    _tree = _ast.parse(self.current_code)

                    # CHECK 2: Reject empty __init__
                    for _node in _ast.walk(_tree):
                        if isinstance(_node, _ast.ClassDef) and _node.name == "InverseSolver":
                            for _method in _node.body:
                                if isinstance(_method, _ast.FunctionDef) and _method.name == "__init__":
                                    # Check if body is just 'pass' or docstring + pass
                                    real_stmts = [s for s in _method.body
                                                  if not (isinstance(s, _ast.Pass) or
                                                         (isinstance(s, _ast.Expr) and isinstance(s.value, (_ast.Constant, _ast.Str))))]
                                    if not real_stmts:
                                        self._log("  ⚠️ [VALIDATION] __init__ is empty (only pass/docstring). Forcing full rewrite...")
                                        _init_ctx = {
                                            'target_type': 'function',
                                            'target_function': '__init__',
                                            'skeleton_code': self.current_skeleton,
                                            'current_full_code': self.current_code,
                                            'plan': self.current_plan,
                                            'task_desc': self.task_desc,
                                            'package_list': self.package_list,
                                            'feedback': "CRITICAL: __init__ body is EMPTY (just 'pass'). You MUST implement it with all necessary instance variables (self.xxx = ...). Read the plan carefully and initialize all required parameters, arrays, and configuration.",
                                            'analysis': None,
                                            'fix_target': '__init__'
                                        }
                                        _init_ctx = self._build_context_with_memory(_init_ctx, "Coder", "Coder")
                                        for _retry in range(3):
                                            self.current_code = self.coder.implement_and_merge(_init_ctx)
                                            self.current_code = extract_python(self.current_code)
                                            # Re-check
                                            _tree2 = _ast.parse(self.current_code)
                                            _fixed = False
                                            for _n2 in _ast.walk(_tree2):
                                                if isinstance(_n2, _ast.ClassDef) and _n2.name == "InverseSolver":
                                                    for _m2 in _n2.body:
                                                        if isinstance(_m2, _ast.FunctionDef) and _m2.name == "__init__":
                                                            _real2 = [s for s in _m2.body
                                                                      if not (isinstance(s, _ast.Pass) or
                                                                             (isinstance(s, _ast.Expr) and isinstance(s.value, (_ast.Constant, _ast.Str))))]
                                                            if _real2:
                                                                _fixed = True
                                            if _fixed:
                                                self._log(f"  ✅ __init__ validated after {_retry + 1} retries")
                                                break
                                            else:
                                                _init_ctx['feedback'] = f"ATTEMPT {_retry+1} STILL FAILED: __init__ is STILL empty. You MUST add self.xxx = value lines. Do NOT use pass."
                except Exception as _e:
                    self._log(f"  [Warning] Post-coding validation error: {_e}")

                solver_path = os.path.join(self.sandbox_dir, "solver.py")
                with open(solver_path, "w") as f:
                    f.write(self.current_code)

                # --- Syntax Check Loop ---
                self._log("  [System] Checking Syntax...")
                # After generating code and saving to solver.py...

                # === SYNTAX GUARDRAIL: Internal loop until clean ===
                syntax_retry = 0
                MAX_SYNTAX_RETRY = 5
                while syntax_retry < MAX_SYNTAX_RETRY:
                    from .sandbox import run_cmd as _run_cmd
                    syn_ok, _, syn_err = _run_cmd(self.python_path, self.sandbox_dir, "solver.py", check_syntax_only=True, syntax_check_timeout=self.syntax_check_timeout)
                    if syn_ok:
                        self._log(f"  ✅ Syntax check passed (Attempt {syntax_retry+1}).")
                        break

                    self._log(f"  ❌ Syntax Error (Attempt {syntax_retry+1}/{MAX_SYNTAX_RETRY})")

                    # Strategy: After 2 failed patches, do full rewrite from skeleton
                    if syntax_retry >= 2:
                        self._log(f"  [System] Syntax patch failed {syntax_retry+1}x. Switching to FULL REWRITE from skeleton.")
                        ctx = {
                            'target_type': 'full_rewrite',
                            'skeleton_code': self.current_skeleton,
                            'current_full_code': self.current_skeleton,  # Start fresh from skeleton
                            'plan': self.current_plan,
                            'task_desc': self.task_desc,
                            'package_list': self.package_list,
                            'feedback': f"CRITICAL: Previous code had PERSISTENT syntax errors after {syntax_retry+1} attempts. Start FRESH from the skeleton and implement ALL functions. Error was:\n{syn_err[-500:]}"
                        }
                    else:
                        ctx = {
                            'target_type': 'full_rewrite',
                            'skeleton_code': self.current_skeleton,
                            'plan': self.current_plan,
                            'feedback': f"SYNTAX ERROR (Attempt {syntax_retry+1}):\n{syn_err}"
                        }
                    ctx = self._build_context_with_memory(ctx, "Coder", "Coder")
                    self.current_code = self.coder.implement_and_merge(ctx)

                    # ✅ UPDATED RECORD STEP
                    self._record_step(iter_id, "Coder", input_data=ctx, output_data={
                        "task_type": "full_rewrite_syntax",
                        "error": syn_err,
                        "code": self.current_code
                    })

                    # Rewrite file for next check
                    self.current_code = extract_python(self.current_code)
                    with open(os.path.join(self.sandbox_dir, "solver.py"), "w") as f:
                        f.write(self.current_code)

                    syntax_retry += 1
                else:
                    # Exhausted retries - don't abort, escalate to Planner for a completely different approach
                    failure_record = {
                        "iteration": self.retry_count + 1,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ticket_assigned_to": "Planner",
                        "fix_target": None,
                        "analysis": f"Persistent syntax errors after {MAX_SYNTAX_RETRY} retries. The current approach generates uncompilable code. Need a SIMPLER algorithm.",
                        "evidence": syn_err[-500:],
                        "feedback": "The previous approach generates code with persistent syntax errors. Please propose a SIMPLER algorithm that is easier to implement correctly. Prefer classical methods (e.g., least squares, Tikhonov regularization, simple iterative methods) over complex architectures."
                    }
                    self.failure_history.append(failure_record)
                    self._log(f"  ⚠️ Syntax errors persist after {MAX_SYNTAX_RETRY} retries. Escalating to Planner for simpler approach.")
                    ticket = "Planner"
                    feedback = failure_record
                    self.retry_count += 1
                    self._reset_downstream_state("Planner")
                    continue


                # 如果语法没问题，进入运行阶段
                self._save_artifact(f"iter_{iter_id}_solver.py", self.current_code)

                # --- STATIC ANALYSIS: Detect dangerous patterns before execution ---
                try:
                    _dangerous_patterns = []
                    # Pattern 1: .item() without ndim check
                    if '.item()' in self.current_code and 'ndim' not in self.current_code:
                        _dangerous_patterns.append(("UNSAFE_ITEM", "Code calls .item() without checking ndim first. Use: `data = raw.item() if raw.ndim == 0 else raw`"))
                    # Pattern 2: Hardcoded file paths that don't exist
                    import re as _re
                    _ext_files = _re.findall(r"['\"]([^'\"]*\.(tif|yaml|h5|csv|mat|fits|json))['\"]", self.current_code)
                    _ext_files = [f for f, _ in _ext_files if 'dataset/' not in f and 'output' not in f and 'eval_' not in f]
                    if _ext_files:
                        _dangerous_patterns.append(("EXTERNAL_FILES", f"Code references non-existent files: {_ext_files[:3]}. Only dataset/input.npy exists."))

                    if _dangerous_patterns:
                        self._log(f"  ⚠️ [STATIC ANALYSIS] Found {len(_dangerous_patterns)} dangerous pattern(s): {[p[0] for p in _dangerous_patterns]}")
                        _fix_feedback = "CRITICAL CODE ISSUES DETECTED (fix before execution):\n"
                        for _pname, _pdesc in _dangerous_patterns:
                            _fix_feedback += f"- {_pname}: {_pdesc}\n"
                        _fix_ctx = {
                            'target_type': 'full_rewrite',
                            'skeleton_code': self.current_skeleton,
                            'current_full_code': self.current_code,
                            'plan': self.current_plan,
                            'task_desc': self.task_desc,
                            'package_list': self.package_list,
                            'feedback': _fix_feedback
                        }
                        _fix_ctx = self._build_context_with_memory(_fix_ctx, "Coder", "Coder")
                        self.current_code = self.coder.implement_and_merge(_fix_ctx)
                        self.current_code = extract_python(self.current_code)
                        with open(os.path.join(self.sandbox_dir, "solver.py"), "w") as f:
                            f.write(self.current_code)
                        self._log(f"  ✅ [STATIC ANALYSIS] Code patched for detected issues.")
                except Exception as _sa_err:
                    self._log(f"  [Warning] Static analysis error: {_sa_err}")

                ticket = "Execution"

            # =========================================================
            # Stage 4: Execution & Judgment
            # =========================================================
            if ticket == "Execution":
                self._log(">>> [System] Executing...")
                # action_sequence.append(f"Iteration {iter_id}: Execution") removed

                # Cleanup stale output to prevent false positives in evaluation
                output_path = "output.npy" # relative to sandbox
                if os.path.exists(os.path.join(self.sandbox_dir, output_path)):
                    try:
                        os.remove(os.path.join(self.sandbox_dir, output_path))
                        self._log("  [System] Removed stale output.npy before execution.")
                    except OSError as e:
                        self._log(f"  [Warning] Failed to remove stale output.npy: {e}")

                from .sandbox import run_cmd as _run_cmd
                success, stdout, stderr = _run_cmd(self.python_path, self.sandbox_dir, "solver.py", timeout=self.execution_timeout)

                # --- QUICK-FIX RETRY: If crashes immediately with common fixable errors, patch and retry once ---
                if not success and stderr:
                    _quick_fix_patterns = [
                        ('can only convert an array of size 1', 'Replace `.item()` calls with safe pattern: `data = raw.item() if raw.ndim == 0 else raw`'),
                        ('unexpected keyword argument', 'Remove the unsupported keyword argument from the function call. Only use standard well-documented parameters.'),
                        ('No module named', 'The imported module does not exist. Use an alternative or implement the functionality manually with numpy/scipy.'),
                        ('cannot import name', 'The function/class does not exist in this version of the library. Check the correct import path or use an alternative.'),
                    ]
                    _matched_fix = None
                    for _pattern, _fix_hint in _quick_fix_patterns:
                        if _pattern in stderr:
                            _matched_fix = _fix_hint
                            break

                    if _matched_fix and self.retry_count < self.max_retries - 1:
                        self._log(f"  ⚠️ [QUICK-FIX] Detected fixable crash. Attempting auto-patch...")
                        _qf_ctx = {
                            'target_type': 'full_rewrite',
                            'skeleton_code': self.current_skeleton,
                            'current_full_code': self.current_code,
                            'plan': self.current_plan,
                            'task_desc': self.task_desc,
                            'package_list': self.package_list,
                            'feedback': f"RUNTIME CRASH:\n{stderr[-800:]}\n\nFIX: {_matched_fix}\nRewrite the code to fix this specific error. Keep the same algorithm but fix the bug."
                        }
                        _qf_ctx = self._build_context_with_memory(_qf_ctx, "Coder", "Coder")
                        self.current_code = self.coder.implement_and_merge(_qf_ctx)
                        self.current_code = extract_python(self.current_code)
                        with open(os.path.join(self.sandbox_dir, "solver.py"), "w") as f:
                            f.write(self.current_code)
                        # Retry execution
                        success, stdout, stderr = _run_cmd(self.python_path, self.sandbox_dir, "solver.py", timeout=self.execution_timeout)
                        logs = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                        self._log(f"  [QUICK-FIX] Retry result: {'Success' if success else 'Still failing'}")

                logs = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                self._save_artifact(f"iter_{iter_id}_exec_log.txt", logs)

                # Metrics Calculation
                metrics = None
                eval_success = False

                if success:
                    self._log("  [System] Execution Success. Running Evaluation...")

                    # --- OUTPUT SHAPE AUTO-FIX ---
                    try:
                        import numpy as np
                        output_path_full = os.path.join(self.sandbox_dir, "output.npy")
                        gt_path_full = os.path.join(self.sandbox_dir, "dataset", "gt_output.npy")
                        if os.path.exists(output_path_full) and os.path.exists(gt_path_full):
                            pred = np.load(output_path_full, allow_pickle=True)
                            gt = np.load(gt_path_full, allow_pickle=True)
                            if pred.shape != gt.shape:
                                self._log(f"  ⚠️ [Shape Fix] pred={pred.shape} vs gt={gt.shape}. Attempting auto-fix...")
                                # Try common fixes: squeeze, reshape, resize
                                fixed = False
                                # 1. Squeeze singleton dims
                                pred_squeezed = np.squeeze(pred)
                                if pred_squeezed.shape == gt.shape:
                                    np.save(output_path_full, pred_squeezed.astype(np.float64))
                                    self._log(f"  ✅ [Shape Fix] Squeezed {pred.shape} -> {pred_squeezed.shape}")
                                    fixed = True
                                # 2. If just one extra/missing batch dim
                                elif len(pred.shape) == len(gt.shape) + 1 and pred.shape[0] == 1:
                                    np.save(output_path_full, pred[0].astype(np.float64))
                                    self._log(f"  ✅ [Shape Fix] Removed batch dim {pred.shape} -> {pred[0].shape}")
                                    fixed = True
                                elif len(pred.shape) + 1 == len(gt.shape) and gt.shape[0] == 1:
                                    np.save(output_path_full, pred[np.newaxis].astype(np.float64))
                                    fixed = True
                                # 3. Same total elements, different shape -> reshape
                                elif pred.size == gt.size:
                                    np.save(output_path_full, pred.reshape(gt.shape).astype(np.float64))
                                    self._log(f"  ✅ [Shape Fix] Reshaped {pred.shape} -> {gt.shape}")
                                    fixed = True
                                if not fixed:
                                    self._log(f"  ❌ [Shape Fix] Cannot auto-fix shape mismatch")
                    except Exception as shape_err:
                        self._log(f"  [Warning] Shape auto-fix error: {shape_err}")

                    # 运行 eval_script.py output.npy
                    e_ok, e_out, e_err = _run_cmd(self.python_path, self.sandbox_dir, "eval_script.py", args=["output.npy"])
                    self._log(f"  [Eval] Return Code: {'Success' if e_ok else 'Failure'}")
                    self._log(f"  [Eval] STDOUT:\n{e_out}\n  [Eval] STDERR:\n{e_err}")
                    if e_ok:
                        try:
                            metrics = json.loads(e_out)
                            eval_success = True
                            self._log(f"  [Metrics] {metrics}")
                        except:
                            logs += f"\n\n[Eval Error] JSON Parse failed: {e_out}"
                    else:
                        logs += f"\n\n[Eval Error] Script failed: {e_err}"
                else:
                    self._log(f"  [System] Execution Failed. Skipping Evaluation. Error: {stderr}")

                # ✅ UPDATED RECORD STEP
                self._record_step(iter_id, "Execution", input_data="solver.py", output_data={
                    "success": success,
                    "eval_success": eval_success,
                    "metrics": metrics,
                    "stdout": stdout,
                    "stderr": stderr
                })

                if eval_success and metrics:
                    curr_psnr = metrics.get('psnr', 0)

                    # 动态阈值计算：
                    # 1. 基础要求：Baseline 的 80%
                    # 2. 保底要求：至少 20.0 dB
                    # 取二者中的较大值
                    base_psnr = self.baseline_metrics.get('psnr', 0)
                    base_threshold = base_psnr * self.baseline_ratio
                    min_guaranteed_threshold = self.min_guaranteed_psnr

                    threshold = max(base_threshold, min_guaranteed_threshold)

                    self._log(f"  [Eval] Threshold Logic: max({base_threshold:.2f}, {min_guaranteed_threshold}) = {threshold:.2f} | Baseline PSNR: {base_psnr:.2f}")

                    if curr_psnr >= threshold:
                        self._log(f"🎉 SUCCESS! PSNR {metrics['psnr']} >= {threshold}")
                        self._save_snapshot(self.retry_count + 1, "final_success", {
                            "metrics": metrics,
                            "threshold": threshold,
                            "solver_code_path": f"iter_{self.retry_count+1:02d}_solver.py"
                        })

                        # --- SKILL DISTILLATION (SUCCESS) ---
                        try:
                            # Update credit scores for used knowledge
                            self.skill_manager.update_scores(list(self.used_knowledge_ids), success=True)

                            trajectory = {
                                "exp_id": self.exp_id,
                                "task_name": self.task_name,
                                "task_desc": self.task_desc,
                                "domain": "General", # Default
                                "difficulty": "Unknown",
                                "outcome": "success",
                                "quality_score": metrics.get('psnr', 0.0) if metrics else 0.0,
                                "steps": self.trajectory_steps,
                                "used_knowledge_ids": list(self.used_knowledge_ids),
                                "retrieval_key": f"Trajectory: {self.task_name} (Success)",
                                "timestamp": int(time.time()),
                                "final_plan": self.current_plan,
                                "final_skeleton": self.current_skeleton,
                                "final_code": self.current_code,
                                "final_reward": metrics,
                                "key_logs": logs[-2000:]
                            }
                            # Store the stats
                            self.distillation_stats = self.skill_manager.distill_and_store(trajectory)
                        except Exception as e:
                            self._log(f"  ⚠️ Skill distillation failed: {e}")

                        # Generate final report
                        self.generate_knowledge_report(success=True)
                        return True  # EXIT WORKFLOW - No Judge needed
                    else:
                        self._log(f"⚠️ Metrics below threshold: PSNR {metrics['psnr']} < {threshold}")


                # --- Judge Agent ---
                self._log("\n>>> [Agent] Judge Analyzing Failure Root Cause...")

                # Use clean task desc (no global skills injected)
                clean_task_desc = self.task_desc

                judge_base_ctx = {
                    'task_desc': clean_task_desc,
                    'logs': logs[-1000:],
                    'metrics': metrics,
                    'baseline_metrics': self.baseline_metrics,
                    'current_code_snippet': self.current_code
                }

                # Custom Retrieval Query for Judge: Task + Error Logs + Low Metrics
                # Focus on "Diagnosis" skills
                judge_query = f"{clean_task_desc}\n\nEXECUTION LOGS (Error Focus):\n{stderr[-500:]}"
                if metrics:
                    judge_query += f"\nMETRICS: {metrics}"

                # Inject Knowledge (Core & Experience) using the standard pipeline
                judge_ctx = self._build_context_with_memory(
                    base_context=judge_base_ctx,
                    agent_role="Judge",
                    current_ticket="Judge",
                    retrieval_query=judge_query # Pass custom query
                )

                judgment = self.judge.generate(judge_ctx)
                self._save_artifact(f"iter_{iter_id}_judge.json", judgment)
                # 解析 Judge 的 JSON 输出
                try:
                    judgment = extract_json(judgment)
                    result = json.loads(judgment)
                    # Enforce evidence field (critical for traceability)
                    if 'evidence' not in result:
                        result['evidence'] = 'MISSING_EVIDENCE_FALLBACK'
                        result['analysis'] = '[SYSTEM OVERRIDE] Judge omitted evidence field. Defaulting to Coder.'
                        result['ticket_assigned_to'] = 'Coder'

                    self._log(f"  [Judge] Ticket: {result['ticket_assigned_to']} | Analysis: {result.get('analysis', 'N/A')}")

                    # ✅ SPECIAL RETRIEVAL KEY GENERATION FOR JUDGE
                    # retrieval_key = f"{error_type}: {execution_summary} in {code_context}"
                    error_type = result.get('ticket_assigned_to', 'Unknown_Error')
                    # Try to find specific error type in analysis (heuristic)
                    if "NaN" in result.get('analysis', ''): error_type = "NaN_Error"
                    elif "Shape" in result.get('analysis', ''): error_type = "Shape_Mismatch"

                    execution_summary = result.get('evidence', 'No evidence')[:50]
                    code_context = result.get('fix_target', 'unknown_context')

                    judge_retrieval_key = f"{error_type}: {execution_summary} in {code_context}"

                    # Compact Output Structure
                    judge_output = {
                        "retrieval_key": judge_retrieval_key,
                        "error_type": error_type,
                        "error_category": result.get('ticket_assigned_to', 'General'),
                        "execution_summary": result.get('evidence', ''),
                        "judgement_summary": result.get('analysis', '')[:200],
                        "outcome": "pending",
                        # Full refs
                        "full_judgement_analysis": result.get('analysis', ''),
                        "ticket": result.get('ticket_assigned_to'),
                        "fix_target": result.get('fix_target'),
                        "feedback": result.get('feedback')
                    }

                    # ✅ UPDATED RECORD STEP
                    self._record_step(iter_id, "Judge", input_data=judge_ctx, output_data=judge_output, retrieval_key=judge_retrieval_key)

                    # Record failure history BEFORE snapshot
                    failure_record = {
                        "iteration": self.retry_count + 1,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ticket_assigned_to": result['ticket_assigned_to'],
                        "fix_target": result.get('fix_target'),
                        "analysis": result.get('analysis', ''),
                        "evidence": result.get('evidence', ''),
                        "feedback": result.get('feedback', ''),
                        "metrics": metrics,
                        "logs_snippet": logs[-500:]
                    }
                    self.failure_history.append(failure_record)
                    self._log(f"  [System] Recorded failure history for {result['ticket_assigned_to']}")


                    # self._reset_sandbox_to_phase0_state()

                    # 失败，分发 Ticket
                    # ticket = result['ticket_assigned_to']

                    new_ticket = result['ticket_assigned_to']
                    self._reset_downstream_state(new_ticket)  # ← 新增调用
                    ticket = new_ticket

                    feedback = result
                    self.retry_count += 1

                except Exception as e:
                    self._log(f"Judge output parse error: {e}")
                    # Fallback
                    ticket = "Coder"
                    feedback = {'analysis': "Judge output invalid, defaulting to check code."}
                    self.retry_count += 1

        self._log("❌ Failed.")

        # --- SKILL DISTILLATION (FAILURE) ---
        try:
            # Update credit scores for used knowledge (Negative Feedback)
            self.skill_manager.update_scores(list(self.used_knowledge_ids), success=False)

            if self.failure_history:
                last_failure = self.failure_history[-1]
                trajectory = {
                    "exp_id": self.exp_id,
                    "task_name": self.task_name,
                    "task_desc": self.task_desc,
                    "domain": "General", # Default
                    "difficulty": "Unknown",
                    "outcome": "failure",
                    "quality_score": 0.0,
                    "steps": self.trajectory_steps,
                    "used_knowledge_ids": list(self.used_knowledge_ids),
                    "retrieval_key": f"Trajectory: {self.task_name} (Failure)",
                    "timestamp": int(time.time()),
                    "final_plan": self.current_plan,
                    "final_skeleton": self.current_skeleton,
                    "final_code": self.current_code,
                    "final_reward": last_failure.get('metrics', 0),
                    "key_logs": last_failure.get('analysis', '') + "\n" + last_failure.get('evidence', '')
                }
                # Store the stats
                self.distillation_stats = self.skill_manager.distill_and_store(trajectory)
        except Exception as e:
            self._log(f"  ⚠️ Skill distillation failed: {e}")

        self.failure_history.clear()

        # Generate final report
        self.generate_knowledge_report(success=False)
        return False


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    workflow = InverseProblemWorkflow(
        task_desc="Recover a 64x64 image from 50% randomly missing pixels. Use ADMM.",
        gt_code_path="./gt_repo",
        python_path="/usr/bin/python3",
        working_dir="./workspace_run_1"
    )
    workflow.run()
