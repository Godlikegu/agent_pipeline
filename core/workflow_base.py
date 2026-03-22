"""
core/workflow_base.py — Pipeline workflow base class.

Assumes the sandbox is already set up (by code_cleaner) with:
  - dataset/input.npy, dataset/gt_output.npy, dataset/baseline.npy
  - eval_script.py
The workflow starts from task_description and uses skills for generation.
"""
import os
import sys
import json
import time
import ast
import datetime
from typing import List, Dict, Tuple, Any, Optional

from agents.planner import PlannerAgent, CriticAgent
from agents.architect import ArchitectAgent
from agents.coder import CoderAgent
from agents.judge import JudgeAgent
from agents.sandbox_agents import DataGenAgent, EvalGenAgent, get_installed_libraries

from utils.text_utils import extract_json, extract_python, highlight_target_in_code, format_failure_histories
from code_cleaner.test_harness import load_data_shapes


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
    ):
        self.task_name = task_name
        self.task_desc = task_desc
        self.sandbox_dir = sandbox_dir
        self.python_path = python_path

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

        self.planner = PlannerAgent(client, model_name)
        self.critic = CriticAgent(client, model_name)
        self.architect = ArchitectAgent(client, model_name)
        self.coder = CoderAgent(client, model_name)
        self.judge = JudgeAgent(client, model_name)
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

        self.min_guaranteed_psnr = eval_cfg.get("min_guaranteed_psnr", 20.0)
        self.baseline_ratio = eval_cfg.get("baseline_ratio", 0.8)

        self.top_k_coder = retrieval_cfg.get("top_k_coder", 4)
        self.top_k_default = retrieval_cfg.get("top_k_default", 3)

        self.used_knowledge_ids: set = set()
        self.distillation_stats = {"knowledge_general": 0, "knowledge_task_specific": 0, "code": 0}

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_id = f"{self.task_name}_{timestamp}"

        snapshot_root = paths_cfg.get("sandbox_root", "/data/guyuxuan/agent/end_sandbox")
        self.snapshot_dir = os.path.join(snapshot_root, self.model_name, self.exp_id)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.log_file = os.path.join(self.snapshot_dir, "workflow.log")
        open(self.log_file, "a").close()

        self._log(f"Workflow initialized. Exp ID: {self.exp_id}")
        self._log(f"Sandbox Directory: {self.sandbox_dir}")

    # ---- logging ----
    def _log(self, message: str):
        ts = time.strftime("[%H:%M:%S]")
        formatted = f"{ts} {message}"
        print(formatted)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    # ---- trajectory recording ----
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
        with open(path, "w") as f:
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
                        if isinstance(first, ast.Expr) and isinstance(first.value, (ast.Str, ast.Constant)):
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
            if relevant:
                context["failure_history"] = format_failure_histories(relevant)

        try:
            query_text = retrieval_query or self.task_desc.split("### GENERAL KNOWLEDGE")[0].strip()
            top_k = self.top_k_coder if agent_role == "Coder" else self.top_k_default
            knowledge = self.skill_manager.retrieve_knowledge(task_desc=query_text, agent_role=agent_role, top_k=top_k)

            for cat_items in knowledge.values():
                for item in cat_items:
                    if "id" in item:
                        self.used_knowledge_ids.add(item["id"])

            knowledge_prompt = self.skill_manager.format_knowledge_for_prompt(knowledge)
            if knowledge_prompt:
                context["knowledge_context"] = knowledge_prompt
        except Exception as e:
            self._log(f"  [System] Knowledge injection failed: {e}")

        return context

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
                    report += f"  - {item.get('name', 'unknown')} (score: {item.get('credit_score', 0):.2f})\n"
            except Exception as e:
                report += f"  Error: {e}\n"
        report += "=" * 50 + "\n"
        self._log(report)
