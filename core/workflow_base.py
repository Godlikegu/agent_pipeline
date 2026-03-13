
import os
import sys
import shutil
import subprocess
import json
import time
import ast
import re
import datetime
from typing import List, Dict, Tuple, Any, Optional

from agents.planner import PlannerAgent, CriticAgent
from agents.architect import ArchitectAgent
from agents.coder import CoderAgent
from agents.judge import JudgeAgent
from agents.sandbox_agents import DataGenAgent, EvalGenAgent, get_installed_libraries

from utils.text_utils import extract_json, extract_python, highlight_target_in_code, format_failure_histories
from .sandbox import setup_sandbox, run_cmd, reset_sandbox_to_phase0_state
from .executor import phase_0_preparation, load_data_shapes

class InverseProblemBase:
    def __init__(self, task_name: str, task_desc: str, gt_code_path: str, python_path: str, working_dir: str, client: Any, model_name: str, config: dict = None, root_output_dir: str = None, skill_manager: Any = None, max_retries: int = None):
        self.task_name = task_name
        self.task_desc = task_desc
        self.gt_code_path = gt_code_path
        self.python_path = python_path

        # Config
        self.config = config or {}
        pipeline_cfg = self.config.get('pipeline', {})
        eval_cfg = self.config.get('evaluation', {})
        skills_cfg = self.config.get('skills', {})
        retrieval_cfg = skills_cfg.get('retrieval', {})
        paths_cfg = self.config.get('paths', {})

        # Directory structure
        self.root_dir = os.path.abspath(working_dir)
        _root_output = root_output_dir or paths_cfg.get('sandbox_root', '/data/yjh/end_sandbox')
        self.sandbox_dir = os.path.join(_root_output, f"{task_name}_sandbox")

        self.package_list = get_installed_libraries(self.python_path)

        self.client = client
        self.model_name = model_name

        # GT Code Reference (not used — no answer leaking)
        self.gt_code_reference = ""

        # Skill Manager
        self.skill_manager = skill_manager

        # Initialize Agents
        self.planner = PlannerAgent(client, model_name)
        self.critic = CriticAgent(client, model_name)
        self.architect = ArchitectAgent(client, model_name)
        self.coder = CoderAgent(client, model_name)
        self.judge = JudgeAgent(client, model_name)
        self.data_gen_agent = DataGenAgent(client, model_name)
        self.eval_gen_agent = EvalGenAgent(client, model_name)

        # Memory State
        self.current_plan = ""
        self.current_skeleton = ""
        self.current_code = ""
        self.function_list = []
        self.failure_history: List[Dict] = []
        self.trajectory_steps: List[Dict] = []

        self.max_retries = max_retries if max_retries is not None else pipeline_cfg.get('max_retries', 5)
        self.retry_count = 0

        # Pipeline parameters from config
        self.max_history_len = pipeline_cfg.get('max_history_len', 3)
        self.data_gen_timeout = pipeline_cfg.get('data_gen_timeout', 180)
        self.gt_code_snippet_limit = pipeline_cfg.get('gt_code_snippet_limit', 4000)
        self.execution_timeout = pipeline_cfg.get('execution_timeout', 600)
        self.syntax_check_timeout = pipeline_cfg.get('syntax_check_timeout', 30)
        self.code_size_guard = pipeline_cfg.get('code_size_guard', 12000)

        # Evaluation parameters from config
        self.min_guaranteed_psnr = eval_cfg.get('min_guaranteed_psnr', 20.0)
        self.baseline_ratio = eval_cfg.get('baseline_ratio', 0.8)

        # Skills retrieval parameters from config
        self.top_k_coder = retrieval_cfg.get('top_k_coder', 4)
        self.top_k_default = retrieval_cfg.get('top_k_default', 3)

        # Knowledge System: Track used knowledge items
        self.used_knowledge_ids = set()

        # Track newly generated knowledge during this run
        self.distillation_stats = {'instances': 0, 'experiences': 0, 'core': 0}

        # Experiment ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Subclasses should append suffix (e.g. _test_) if needed, but base uses standard
        self.exp_id = f"{self.task_name}_{timestamp}"

        # Log directory
        self.snapshot_dir = os.path.join(_root_output, self.model_name, self.exp_id)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.log_file = os.path.join(self.snapshot_dir, "workflow.log")
        open(self.log_file, 'a').close()

        self._log(f"Workflow initialized. Exp ID: {self.exp_id}")
        self._log(f"Sandbox Directory: {self.sandbox_dir}")
        self._log(f"Snapshot Directory: {self.snapshot_dir}")

    def _log(self, message: str):
        timestamp = time.strftime("[%H:%M:%S]")
        formatted_msg = f"{timestamp} {message}"
        print(formatted_msg)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(formatted_msg + "\n")

    def _generate_retrieval_key(self, role: str, input_data: Any, output_data: Any) -> str:
        """
        Auto-generate retrieval key based on input/output content.
        """
        try:
            if role == "Planner":
                # Input usually contains task_desc. Output is plan.
                # Key: Task Summary
                desc = input_data.get('task_desc', '') if isinstance(input_data, dict) else str(input_data)
                return f"Plan for task: {desc[:100]}..."

            elif role == "Architect":
                # Input: Plan. Output: Skeleton.
                # Key: Plan Summary
                plan = input_data.get('plan', '') if isinstance(input_data, dict) else str(input_data)
                return f"Architecture for plan: {plan[:100]}..."

            elif role == "Coder":
                # Input: Task info. Output: Code.
                # Key: Task Target
                target = input_data.get('target_function') or input_data.get('target_type') or 'unknown'
                return f"Implement {target} in {self.task_name}"

            elif role == "Execution":
                return f"Execution of {self.task_name}"

            elif role == "Judge":
                # Fallback if not provided explicitly
                return f"Judge analysis for {self.task_name}"

            return f"{role} step for {self.task_name}"
        except:
            return f"Step {role} in {self.task_name}"

    def _record_step(self, iteration: int, role: str, input_data: Any, output_data: Any, retrieval_key: str = None):
        """
        Records a detailed step in the trajectory with input, output and retrieval key.
        """
        if retrieval_key is None:
            retrieval_key = self._generate_retrieval_key(role, input_data, output_data)

        step = {
            "step_id": len(self.trajectory_steps) + 1,
            "iteration": iteration,
            "role": role,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data,
            "retrieval_key": retrieval_key
        }
        self.trajectory_steps.append(step)

    def _save_artifact(self, filename: str, content: str) -> str:
        path = os.path.join(self.snapshot_dir, f"{self.exp_id}_{filename}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def _save_snapshot(self, iteration: int, stage: str, content: dict):
        path = os.path.join(self.snapshot_dir, f"iter_{iteration:03d}_{stage}.json")
        with open(path, 'w') as f:
            json.dump({
                "exp_id": self.exp_id,
                "iteration": iteration,
                "stage": stage,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **content
            }, f, indent=2)

    def _parse_functions_from_skeleton(self, skeleton_code: str) -> List[str]:
        func_list = []
        try:
            tree = ast.parse(skeleton_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "InverseSolver":
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            func_list.append(item.name)
        except Exception as e:
            self._log(f"[System] AST Parsing Error: {e}")
        return func_list

    def _validate_skeleton(self, code: str) -> Tuple[bool, str]:
        if not code.strip(): return False, "Empty code."
        try:
            tree = ast.parse(code)
            has_solver = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "InverseSolver":
                    has_solver = True
            if not has_solver: return False, "Missing 'class InverseSolver'."
            return True, ""
        except SyntaxError as e: return False, f"Syntax Error: {e}"
        except Exception as e: return False, f"Validation Error: {e}"

    def _extract_function_signature(self, code: str, func_name: str) -> Optional[str]:
        """
        Extracts the full function signature (including decorators and docstring) for a given function name
        from the provided code (usually skeleton).
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Found the function. Extract source lines.
                    # AST nodes have lineno (1-based).
                    # We want to capture:
                    # 1. Decorators (if any)
                    # 2. def func(...) -> ...:
                    # 3. Docstring (if strictly the first statement)

                    lines = code.split('\n')
                    start_line = node.lineno - 1

                    # Adjust start_line to include decorators
                    if node.decorator_list:
                        start_line = node.decorator_list[0].lineno - 1

                    # Find end of signature (:)
                    # This is tricky with multi-line args.
                    # We can approximate by looking for the docstring or the first statement.

                    end_line = node.lineno # At least the def line

                    # Check for docstring
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # If docstring exists, the signature block effectively ends after the docstring?
                        # Or do we want just the signature?
                        # User requested "signature + docstring".

                        # Find the docstring node in body
                        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                             doc_node = node.body[0]
                             end_line = doc_node.end_lineno if hasattr(doc_node, 'end_lineno') else doc_node.lineno
                    else:
                        # No docstring, try to find the colon of def?
                        # Or just take the first few lines until body starts?
                        if node.body:
                            # The body starts at the first statement
                            # We want everything BEFORE the first statement (excluding docstring if we already handled it)
                            # Actually, let's just grab the text from start_line to the line before first body statement
                            first_stmt = node.body[0]
                            end_line = first_stmt.lineno - 2 # Line before the statement
                            if end_line < start_line: end_line = start_line # Fallback
                        else:
                            end_line = start_line # Empty body

                    # Extract snippet
                    snippet = "\n".join(lines[start_line : end_line + 1])
                    return snippet.strip()

        except Exception as e:
            self._log(f"[System] Signature extraction failed for {func_name}: {e}")

        return None

    def _clean_code(self, code: str) -> str:
        return extract_python(code)

    def _write_file(self, filename: str, content: str):
        content = extract_python(content)
        path = os.path.join(self.sandbox_dir, filename)
        with open(path, "w", encoding='utf-8') as f:
            f.write(content)

    def _build_context_with_memory(self, base_context: Dict[str, Any], agent_role: str, current_ticket: str, retrieval_query: str = None) -> Dict[str, Any]:
        if len(self.failure_history) > self.max_history_len:
            self.failure_history = sorted(self.failure_history, key=lambda x: x.get('iteration', 0), reverse=True)[:self.max_history_len]

        context = base_context.copy()

        # 1. Previous Output Injection (state from prior iteration — NO error analysis here)
        if agent_role == "Planner":
            # Inject hard constraints (shapes)
            constraints = []
            if hasattr(self, 'input_shape') and self.input_shape: constraints.append(f"Input shape: {self.input_shape}")
            if hasattr(self, 'output_shape') and self.output_shape: constraints.append(f"Output shape: {self.output_shape}")

            if constraints:
                constraint_block = "### 🔑 HARD CONSTRAINTS (NON-NEGOTIABLE)\n" + "\n".join([f"• {c}" for c in constraints]) + "\n\n"
                context["task_desc"] = constraint_block + context.get("task_desc", "")

            # Inject previous plan as reference (not error info — that goes in Layer 2)
            if self.current_plan.strip():
                context["previous_plan"] = self.current_plan

        elif agent_role == "Architect" and self.current_skeleton.strip():
            context["plan"] = context.get("plan", "")
            context["previous_skeleton"] = self.current_skeleton

        elif agent_role == "Coder" and self.current_code.strip():
            if base_context.get("fix_target"):
                target = base_context["fix_target"]
                context["current_full_code"] = highlight_target_in_code(self.current_code, target)

        # 2. Failure History Injection (deduplicated — uses dedicated field, not mixed into task_desc/feedback)
        if agent_role not in ["Judge", "Critic"]:
            relevant_histories = [h for h in self.failure_history if h.get("ticket_assigned_to") == current_ticket]
            if relevant_histories:
                history_section = format_failure_histories(relevant_histories)
                # Always use a dedicated field to avoid polluting task_desc or duplicating feedback
                context["failure_history"] = history_section

        # 3. Knowledge Injection — always call skill_manager (NoSkillManager handles the empty case)
        try:
            # Determine Retrieval Query
            if retrieval_query:
                query_text = retrieval_query
            else:
                # Fallback to default (Task Desc)
                query_text = self.task_desc.split("### 🛡️ CORE KNOWLEDGE")[0].strip()

            top_k = self.top_k_coder if agent_role == "Coder" else self.top_k_default

            knowledge = self.skill_manager.retrieve_knowledge(
                task_desc=query_text,
                agent_role=agent_role,
                top_k=top_k
            )

            # Track usage
            for k_type in ['core', 'experience', 'instance']:
                for item in knowledge.get(k_type, []):
                    if 'id' in item:
                        self.used_knowledge_ids.add(item['id'])

            knowledge_prompt = self.skill_manager.format_knowledge_for_prompt(knowledge)

            if knowledge_prompt:
                # Use dedicated knowledge_context field
                context["knowledge_context"] = knowledge_prompt

        except Exception as e:
            self._log(f"  [System] Knowledge injection failed: {e}")

        return context

    def _reset_downstream_state(self, ticket: str):
        if ticket == "Planner":
            self._log("  [State] Resetting downstream state: skeleton + code (Planner ticket)")
            self.current_skeleton = ""
            self.current_code = ""
            self.function_list = []
        elif ticket == "Architect":
            self._log("  [State] Resetting downstream state: code only (Architect ticket)")
            self.current_code = ""

    def generate_knowledge_report(self, success: bool):
        """Generates a final report of knowledge usage and success."""
        report = "\n" + "="*50 + "\n"
        report += "📊 KNOWLEDGE USAGE REPORT\n"
        report += "="*50 + "\n"
        report += f"Task Outcome: {'✅ SUCCESS' if success else '❌ FAILURE'}\n"
        report += f"Knowledge Items Used: {len(self.used_knowledge_ids)}\n"

        # Always call skill_manager (NoSkillManager handles the empty case)
        if self.used_knowledge_ids:
            try:
                # We need a method in SkillManager to get details by IDs.
                # Since we don't have it yet, we will rely on a new method or try to fetch from DB manually?
                # Best practice: Add method to SkillManager.
                # Assuming `get_knowledge_details` exists.
                if hasattr(self.skill_manager, 'get_knowledge_details'):
                    details = self.skill_manager.get_knowledge_details(list(self.used_knowledge_ids))

                    # Group by Type
                    by_type = {"core": [], "experience": [], "instance": []}
                    for item in details:
                        k_type = item.get('type', 'unknown')
                        if k_type in by_type: by_type[k_type].append(item)

                    for k_type, items in by_type.items():
                        if items:
                            report += f"\n[{k_type.upper()}]\n"
                            for item in items:
                                score_change = "+0.1" if success else "-0.2"
                                report += f"  - {item['name']} (Score: {item.get('credit_score', 1.0):.2f} -> {score_change})\n"
                else:
                    report += "  (Details unavailable: SkillManager.get_knowledge_details not implemented)\n"
            except Exception as e:
                report += f"  Error generating details: {e}\n"
        else:
            report += "  No knowledge items were injected.\n"

        report += "="*50 + "\n"
        self._log(report)
