"""
core/workflow_base.py -- Pipeline workflow base class.

Assumes the sandbox is already set up with:
  - data/gt_output.npy, data/input_data/, data/meta_data.json
  - eval_script.py
The workflow starts from task_description and uses skills for generation.
"""
import os
import sys
import json
import time
import ast
import shutil
import datetime
import textwrap
from typing import List, Dict, Tuple, Any, Optional

from agents.planner import PlannerAgent, CriticAgent
from agents.architect import ArchitectAgent
from agents.coder import CoderAgent
from agents.judge import JudgeAgent
from agents.sandbox_agents import DataGenAgent, EvalGenAgent, get_installed_libraries

from utils.text_utils import extract_json, extract_python, highlight_target_in_code, format_failure_histories


class WorkflowBase:
    """Base class for the agent pipeline workflow."""

    def __init__(
        self,
        task_name: str,
        task_desc: str,
        sandbox_dir: str,
        python_path: str,
        client: Any,
        model_name: str,
        config: dict = None,
        skill_manager: Any = None,
        max_retries: int = None,
        eval_thresholds: dict = None,
        task_dir: str = None,
    ):
        self.task_name = task_name
        self.task_desc = task_desc
        self.sandbox_dir = sandbox_dir
        self.python_path = python_path
        self.task_dir = task_dir

        self.config = config or {}
        pipeline_cfg = self.config.get("pipeline", {})
        eval_cfg = self.config.get("evaluation", {})
        skills_cfg = self.config.get("skills", {})
        retrieval_cfg = skills_cfg.get("retrieval", {})
        paths_cfg = self.config.get("paths", {})

        self.package_list = get_installed_libraries(self.python_path)

        self.client = client
        self.model_name = model_name

        self.skill_manager = skill_manager

        agents_cfg = self.config.get("agents", {})
        self.planner = PlannerAgent(client, model_name,
                                    temperature=agents_cfg.get("planner", {}).get("temperature", 0.7))
        self.critic = CriticAgent(client, model_name,
                                  temperature=agents_cfg.get("critic", {}).get("temperature", 0.7))
        self.architect = ArchitectAgent(client, model_name,
                                        temperature=agents_cfg.get("architect", {}).get("temperature", 0.7))
        self.coder = CoderAgent(client, model_name,
                                temperature=agents_cfg.get("coder", {}).get("temperature", 0.7))
        self.judge = JudgeAgent(client, model_name,
                                temperature=agents_cfg.get("judge", {}).get("temperature", 0.7))
        self.data_gen_agent = DataGenAgent(client, model_name)
        self.eval_gen_agent = EvalGenAgent(client, model_name)

        self.current_plan = ""
        self.current_skeleton = ""
        self.current_code = ""
        self.function_list: List[str] = []
        self.failure_history: List[Dict] = []
        self.trajectory_steps: List[Dict] = []

        self.max_retries = max_retries if max_retries is not None else pipeline_cfg.get("max_retries", 5)
        self.retry_count = 0

        self.max_history_len = pipeline_cfg.get("max_history_len", 3)
        self.execution_timeout = pipeline_cfg.get("execution_timeout", 600)
        self.syntax_check_timeout = pipeline_cfg.get("syntax_check_timeout", 30)
        self.code_size_guard = pipeline_cfg.get("code_size_guard", 12000)

        self.min_ncc = eval_cfg.get("min_ncc", 0.85)
        self.max_nrmse = eval_cfg.get("max_nrmse", 0.5)

        # Per-task thresholds override global config
        self._eval_boundaries = {}
        if eval_thresholds:
            if "min_ncc" in eval_thresholds:
                self.min_ncc = eval_thresholds["min_ncc"]
            if "max_nrmse" in eval_thresholds:
                self.max_nrmse = eval_thresholds["max_nrmse"]
            if "eval_boundaries" in eval_thresholds:
                self._eval_boundaries = eval_thresholds["eval_boundaries"]

        self.top_k_planner = retrieval_cfg.get("top_k_planner", 3)
        self.top_k_coder = retrieval_cfg.get("top_k_coder", 4)

        # Best result tracking across iterations
        self.best_primary_score = -float('inf')
        self.best_ncc = -1.0  # backward compat
        self.best_iteration = 0
        self.best_passed = False
        self.best_metrics = {}

        self.used_knowledge_ids: set = set()
        self.distillation_stats = {"knowledge_general": 0, "knowledge_task_specific": 0, "code": 0}

        # ---- Trajectory recording ----
        self.round_trajectories: List[Dict] = []
        self._current_round: Dict = {}
        self._round_skills_injected: Dict[int, set] = {}  # round_id -> set of skill IDs

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_id = f"{self.task_name}_{timestamp}"

        snapshot_root = paths_cfg.get("sandbox_root", "./data/end_sandbox")
        safe_model_name = self.model_name.replace("/", "_").replace("\\", "_")
        self.snapshot_dir = os.path.join(snapshot_root, safe_model_name, self.exp_id)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.log_file = os.path.join(self.snapshot_dir, "workflow.log")
        open(self.log_file, "a").close()

        self._log(f"Workflow initialized. Exp ID: {self.exp_id}")
        self._log(f"Sandbox Directory: {self.sandbox_dir}")

    # ---- logging ----
    def _log(self, message: str):
        ts = time.strftime("[%H:%M:%S]")
        formatted = f"{ts} {message}"
        # Safe print that handles non-ASCII on Windows GBK console
        try:
            print(formatted)
        except UnicodeEncodeError:
            import sys
            enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
            sys.stdout.write(formatted.encode(enc, errors="replace").decode(enc, errors="replace") + "\n")
            sys.stdout.flush()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    # ---- round trajectory lifecycle ----
    def _start_round(self, round_id: int):
        """Initialize a new round trajectory record."""
        self._current_round = {
            "round_id": round_id,
            "task_name": self.task_name,
            "success": False,
            "planner_output": "",
            "critic_output": "",
            "architect_output": "",
            "coder_output": "",
            "execution_success": False,
            "execution_stdout": "",
            "execution_stderr": "",
            "eval_metrics": None,
            "judge_output": None,
            "skills_used_ids": [],
            "timestamp": time.time(),
        }
        self._round_skills_injected[round_id] = set()

    def _record_round_agent(self, agent_name: str, output: Any):
        """Record an agent's output into the current round trajectory."""
        if not self._current_round:
            return
        key_map = {
            "Planner": "planner_output",
            "Critic": "critic_output",
            "Architect": "architect_output",
            "Coder": "coder_output",
        }
        key = key_map.get(agent_name)
        if key:
            if isinstance(output, dict):
                text = json.dumps(output, default=str)[:5000]
            else:
                text = str(output)[:5000]
            # For Coder, append (multiple calls per round)
            if agent_name == "Coder" and self._current_round.get(key):
                self._current_round[key] += f"\n---\n{text}"
            else:
                self._current_round[key] = text

    def _record_round_execution(self, success: bool, stdout: str, stderr: str, metrics: dict):
        """Record execution results into the current round trajectory."""
        if not self._current_round:
            return
        self._current_round["execution_success"] = success
        self._current_round["execution_stdout"] = (stdout or "")[:3000]
        self._current_round["execution_stderr"] = (stderr or "")[:3000]
        self._current_round["eval_metrics"] = metrics

    def _record_round_judge(self, judge_output: dict):
        """Record judge output into the current round trajectory."""
        if not self._current_round:
            return
        self._current_round["judge_output"] = judge_output

    def _finalize_round(self, success: bool):
        """Finalize current round and add to trajectories list."""
        if not self._current_round:
            return
        self._current_round["success"] = success
        round_id = self._current_round.get("round_id", 0)
        self._current_round["skills_used_ids"] = list(
            self._round_skills_injected.get(round_id, set())
        )
        self.round_trajectories.append(self._current_round)
        self._current_round = {}

    # ---- legacy trajectory recording (kept for snapshot compatibility) ----
    def _record_step(self, iteration: int, role: str, input_data: Any, output_data: Any, retrieval_key: str = None):
        if retrieval_key is None:
            retrieval_key = self._generate_retrieval_key(role, input_data, output_data)
        self.trajectory_steps.append({
            "step_id": len(self.trajectory_steps) + 1,
            "iteration": iteration,
            "role": role,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data,
            "retrieval_key": retrieval_key,
        })

    def _generate_retrieval_key(self, role: str, input_data: Any, output_data: Any) -> str:
        try:
            if role == "Planner":
                desc = input_data.get("task_desc", "")[:100] if isinstance(input_data, dict) else str(input_data)[:100]
                return f"Plan for: {desc}"
            if role == "Architect":
                plan = input_data.get("plan", "")[:100] if isinstance(input_data, dict) else str(input_data)[:100]
                return f"Architecture for: {plan}"
            if role == "Coder":
                target = (input_data or {}).get("target_function") or (input_data or {}).get("target_type") or "unknown"
                return f"Implement {target} in {self.task_name}"
            if role == "Execution":
                return f"Execution of {self.task_name}"
            if role == "Judge":
                return f"Judge analysis for {self.task_name}"
        except Exception:
            pass
        return f"{role} step for {self.task_name}"

    # ---- artifacts ----
    def _save_artifact(self, filename: str, content: str) -> str:
        path = os.path.join(self.snapshot_dir, f"{self.exp_id}_{filename}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _save_snapshot(self, iteration: int, stage: str, content: dict):
        path = os.path.join(self.snapshot_dir, f"iter_{iteration:03d}_{stage}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"exp_id": self.exp_id, "iteration": iteration, "stage": stage,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **content}, f, indent=2)

    # ---- code helpers ----
    def _parse_functions_from_skeleton(self, skeleton_code: str) -> List[str]:
        func_list = []
        try:
            tree = ast.parse(skeleton_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            func_list.append(item.name)
        except Exception as e:
            self._log(f"[System] AST Parsing Error: {e}")
        return func_list

    def _validate_skeleton(self, code: str) -> Tuple[bool, str]:
        if not code.strip():
            return False, "Empty code."
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax Error: {e}"
        except Exception as e:
            return False, f"Validation Error: {e}"

    def _extract_function_signature(self, code: str, func_name: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    lines = code.split("\n")
                    start = node.lineno - 1
                    if node.decorator_list:
                        start = node.decorator_list[0].lineno - 1
                    end = node.lineno
                    docstring = ast.get_docstring(node)
                    if docstring and node.body:
                        first = node.body[0]
                        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                            end = first.end_lineno if hasattr(first, "end_lineno") else first.lineno
                    elif node.body:
                        end = max(start, node.body[0].lineno - 2)
                    return "\n".join(lines[start: end + 1]).strip()
        except Exception as e:
            self._log(f"[System] Signature extraction failed for {func_name}: {e}")
        return None

    def _clean_code(self, code: str) -> str:
        return extract_python(code)

    def _write_file(self, filename: str, content: str):
        content = extract_python(content)
        path = os.path.join(self.sandbox_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # ---- context building with skills injection ----
    def _build_context_with_memory(self, base_context: Dict[str, Any], agent_role: str, current_ticket: str, retrieval_query: str = None) -> Dict[str, Any]:
        if len(self.failure_history) > self.max_history_len:
            self.failure_history = sorted(self.failure_history, key=lambda x: x.get("iteration", 0), reverse=True)[:self.max_history_len]

        context = base_context.copy()

        if agent_role == "Planner":
            constraints = []
            if hasattr(self, "input_shape") and self.input_shape:
                constraints.append(f"Input shape: {self.input_shape}")
            if hasattr(self, "output_shape") and self.output_shape:
                constraints.append(f"Output shape: {self.output_shape}")
            if constraints:
                block = "### HARD CONSTRAINTS\n" + "\n".join(f"- {c}" for c in constraints) + "\n\n"
                context["task_desc"] = block + context.get("task_desc", "")
            if self.current_plan.strip():
                context["previous_plan"] = self.current_plan

        elif agent_role == "Architect" and self.current_skeleton.strip():
            context["previous_skeleton"] = self.current_skeleton

        elif agent_role == "Coder" and self.current_code.strip():
            if base_context.get("fix_target"):
                context["current_full_code"] = highlight_target_in_code(self.current_code, base_context["fix_target"])

        if agent_role not in ["Judge", "Critic"]:
            relevant = [h for h in self.failure_history if h.get("ticket_assigned_to") == current_ticket]

            # For Coder and Architect: also inject runtime error history from OTHER tickets.
            # This ensures that after a Planner reset (full code regeneration), the Coder
            # still sees previous runtime errors and avoids repeating the same mistakes.
            if agent_role in ("Coder", "Architect"):
                _ERROR_KEYWORDS = ("error", "traceback", "exception", "crash",
                                   "runtime", "unsupported", "not implemented")
                runtime_error_history = [
                    h for h in self.failure_history
                    if h not in relevant
                    and (h.get("evidence") or h.get("analysis", ""))
                    and any(kw in (h.get("evidence", "") + " " + h.get("analysis", "")).lower()
                            for kw in _ERROR_KEYWORDS)
                ]
                relevant.extend(runtime_error_history)

            if relevant:
                context["failure_history"] = format_failure_histories(relevant)

        # Skills injection: ONLY for Planner and Coder
        if agent_role in ("Planner", "Coder"):
            try:
                query_text = retrieval_query or self.task_desc.split("### GENERAL KNOWLEDGE")[0].strip()
                top_k = self.top_k_coder if agent_role == "Coder" else self.top_k_planner

                # For Coder: do NOT exclude skills already injected in this round.
                # The Coder needs the FULL skill set for each function implementation,
                # because each Coder call is independent and may need different skills.
                # Only exclude for Planner (which is called once per round).
                round_id = self._current_round.get("round_id", 0)
                if agent_role == "Planner":
                    already_injected = self._round_skills_injected.get(round_id, set())
                else:
                    already_injected = set()  # Coder gets all skills every time

                knowledge = self.skill_manager.retrieve_knowledge(
                    task_desc=query_text,
                    agent_role=agent_role,
                    top_k=top_k,
                    exclude_ids=already_injected,
                )

                for cat_items in knowledge.values():
                    for item in cat_items:
                        if "id" in item:
                            self.used_knowledge_ids.add(item["id"])
                            self._round_skills_injected.setdefault(round_id, set()).add(item["id"])

                knowledge_prompt = self.skill_manager.format_knowledge_for_prompt(knowledge)
                if knowledge_prompt:
                    context["knowledge_context"] = knowledge_prompt

                    # Log which skills were injected
                    skill_names = []
                    for cat_items in knowledge.values():
                        for item in cat_items:
                            tier = item.get("tier", "?")
                            skill_names.append(f"{item.get('name', '?')} [{tier}]")
                    if skill_names:
                        self._log(f"  [Skills] Injected for {agent_role}: {', '.join(skill_names)}")
            except Exception as e:
                self._log(f"  [System] Knowledge injection failed: {e}")

        return context

    # ---- structural validation ----
    def _structural_validate_and_fix(self, code: str) -> Tuple[str, List[str]]:
        """Detect and fix structural code issues that pass syntax checks but crash at runtime.

        Returns (fixed_code, list_of_issues_found). If no issues, returns (code, []).
        """
        issues = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, []  # syntax errors handled elsewhere

        lines = code.split("\n")

        # 1. Detect nested class definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.ClassDef):
                        issues.append(
                            f"Nested class '{child.name}' inside '{node.name}' "
                            f"(line {child.lineno})"
                        )

        # 2. Detect orphaned methods (self parameter at module scope)
        top_level_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                if args and args[0] == "self":
                    issues.append(
                        f"Orphaned method '{node.name}' with 'self' at module scope "
                        f"(line {node.lineno}) — should be inside a class"
                    )

        # 3. Detect duplicate class definitions
        class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        seen = set()
        for name in class_names:
            if name in seen:
                issues.append(f"Duplicate class definition '{name}'")
            seen.add(name)

        if not issues:
            return code, []

        self._log(f"  [Structural] Found {len(issues)} issues: {issues}")

        # Attempt auto-fix
        try:
            code = self._structural_autofix(code, tree)
            # Verify the fix parses
            ast.parse(code)
            self._log("  [Structural] Auto-fix applied successfully.")
        except Exception as e:
            self._log(f"  [Structural] Auto-fix failed: {e}")

        return code, issues

    def _structural_autofix(self, code: str, tree: ast.Module) -> str:
        """Attempt to fix structural issues by rewriting the AST."""
        lines = code.split("\n")

        # Collect all top-level class ranges
        top_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

        # Fix 1: Flatten nested classes — merge inner class body into outer class
        for cls_node in top_classes:
            nested = [n for n in cls_node.body if isinstance(n, ast.ClassDef)]
            if not nested:
                continue
            # Strategy: remove the inner class header and dedent its methods
            # so they become part of the outer class
            for inner in nested:
                inner_start = inner.lineno - 1  # 0-indexed
                inner_end = inner.end_lineno  # exclusive
                # Find methods in inner class
                inner_methods = [n for n in inner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if inner_methods:
                    # Remove the inner class def line + any decorators
                    # Keep the method bodies, dedented by one level (4 spaces)
                    new_lines = []
                    for i in range(inner_start, inner_end):
                        line = lines[i]
                        # Skip the 'class Foo:' line itself
                        if i == inner_start:
                            continue
                        # Dedent by 4 spaces (remove one nesting level)
                        if line.startswith("        "):  # 8 spaces -> 4 spaces
                            new_lines.append(line[4:])
                        else:
                            new_lines.append(line)
                    # Replace the inner class range with dedented methods
                    lines[inner_start:inner_end] = new_lines

        code = "\n".join(lines)
        # Re-parse after nested class fix
        tree = ast.parse(code)
        lines = code.split("\n")

        # Fix 2: Move orphaned self-methods into the last class before __main__
        top_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        orphaned_funcs = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                if args and args[0] == "self":
                    orphaned_funcs.append(node)

        if orphaned_funcs and top_classes:
            target_class = top_classes[-1]
            # Find the end of the target class
            class_end = target_class.end_lineno  # 1-indexed, inclusive

            # Extract orphaned method lines, indent them to class level (4 spaces)
            insert_lines = []
            remove_ranges = []
            for func in orphaned_funcs:
                start = func.lineno - 1
                end = func.end_lineno
                func_lines = lines[start:end]
                # Indent each line by 4 spaces
                indented = ["    " + l if l.strip() else l for l in func_lines]
                insert_lines.append("")  # blank separator
                insert_lines.extend(indented)
                remove_ranges.append((start, end))

            # Remove orphaned functions (reverse order to preserve indices)
            for start, end in reversed(remove_ranges):
                del lines[start:end]

            # Re-find class end after removals
            # Reparse to get correct positions
            temp_code = "\n".join(lines)
            temp_tree = ast.parse(temp_code)
            for n in temp_tree.body:
                if isinstance(n, ast.ClassDef) and n.name == target_class.name:
                    class_end = n.end_lineno
                    break

            # Insert at class end
            for i, line in enumerate(insert_lines):
                lines.insert(class_end + i, line)

            code = "\n".join(lines)
            tree = ast.parse(code)
            lines = code.split("\n")

        # Fix 3: Merge duplicate class definitions
        top_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        class_groups = {}
        for cls in top_classes:
            class_groups.setdefault(cls.name, []).append(cls)

        for name, defs in class_groups.items():
            if len(defs) <= 1:
                continue
            # Keep the first definition, merge methods from later ones
            primary = defs[0]
            primary_methods = {n.name for n in primary.body if isinstance(n, ast.FunctionDef)}
            for dup in defs[1:]:
                dup_start = dup.lineno - 1
                dup_end = dup.end_lineno
                # Extract new methods not in primary
                new_methods_lines = []
                for item in dup.body:
                    if isinstance(item, ast.FunctionDef) and item.name not in primary_methods:
                        method_lines = lines[item.lineno - 1:item.end_lineno]
                        new_methods_lines.append("")
                        new_methods_lines.extend(method_lines)
                        primary_methods.add(item.name)
                # Remove the duplicate class
                lines[dup_start:dup_end] = []
                # Insert new methods at end of primary class
                if new_methods_lines:
                    # Re-parse to find current primary end
                    temp_code = "\n".join(lines)
                    temp_tree = ast.parse(temp_code)
                    for n in temp_tree.body:
                        if isinstance(n, ast.ClassDef) and n.name == name:
                            insert_at = n.end_lineno
                            for i, line in enumerate(new_methods_lines):
                                lines.insert(insert_at + i, line)
                            break

        return "\n".join(lines)

    # ---- state management ----
    def _reset_downstream_state(self, ticket: str):
        if ticket == "Planner":
            self._log("  [State] Reset: skeleton + code (Planner ticket)")
            self.current_skeleton = ""
            self.current_code = ""
            self.function_list = []
        elif ticket == "Architect":
            self._log("  [State] Reset: code only (Architect ticket)")
            self.current_code = ""

    def generate_knowledge_report(self, success: bool):
        report = "\n" + "=" * 50 + "\n"
        report += "KNOWLEDGE USAGE REPORT\n" + "=" * 50 + "\n"
        report += f"Outcome: {'SUCCESS' if success else 'FAILURE'}\n"
        report += f"Knowledge items used: {len(self.used_knowledge_ids)}\n"
        if self.used_knowledge_ids and hasattr(self.skill_manager, "get_knowledge_details"):
            try:
                details = self.skill_manager.get_knowledge_details(list(self.used_knowledge_ids))
                for item in details:
                    report += f"  - {item.get('name', 'unknown')} [{item.get('tier', '?')}]\n"
            except Exception as e:
                report += f"  Error: {e}\n"
        report += "=" * 50 + "\n"
        self._log(report)
