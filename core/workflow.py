"""
core/workflow.py -- Main pipeline workflow.

Assumes the sandbox is pre-configured with:
  - dataset/gt_output.npy, dataset/input_data/, dataset/meta_data.json
  - eval_script.py
Starts from task_description -> Planner -> Critic -> Architect -> Coder -> Execute -> Judge.
Evaluates using NCC (Normalized Cross-Correlation) and NRMSE (Normalized RMSE).

Trajectory recording:
  - Each round (Planner->...->Judge) is recorded as a RoundTrajectory.
  - After all rounds complete, trajectories are written to data/<task>/trajectories/.
  - If learning_enabled, skills are distilled post-task by SkillsGeneratorAgent.
"""
import os
import json
import time
import re
import numpy as np
from typing import Any

from .workflow_base import WorkflowBase
from .sandbox import run_cmd
from utils.text_utils import extract_json, extract_python


class PipelineWorkflow(WorkflowBase):
    def __init__(self, task_name: str, task_desc: str, sandbox_dir: str, python_path: str,
                 client: Any, model_name: str, config: dict = None, skill_manager: Any = None,
                 max_retries: int = None, eval_thresholds: dict = None, task_dir: str = None):
        super().__init__(task_name, task_desc, sandbox_dir, python_path, client, model_name,
                         config, skill_manager, max_retries, eval_thresholds=eval_thresholds,
                         task_dir=task_dir)

    def run(self) -> bool:
        self.input_shape, self.output_shape = self._load_data_shapes()
        self.data_layout = self._build_data_layout()

        feedback = None
        ticket = "Planner"

        while self.retry_count < self.max_retries:
            iter_id = self.retry_count + 1
            self._log(f"\n{'='*20} Iteration {iter_id} (Ticket: {ticket}) {'='*20}")

            # Start round trajectory
            self._start_round(iter_id)

            # ==================== Planner ====================
            if ticket == "Planner":
                self._log(">>> [Agent] Planner...")
                plan_ctx = self._build_context_with_memory(
                    base_context={
                        "task_desc": self.task_desc,
                        "feedback": feedback,
                        "shape_info": f"Input: {self.input_shape}, Output: {self.output_shape}" if self.input_shape else None,
                        "data_layout": self.data_layout,
                        "package_list": self.package_list,
                    },
                    agent_role="Planner",
                    current_ticket="Planner",
                )
                draft_plan = self.planner.generate(plan_ctx)

                critic_valid = False
                critic_resp_str = ""
                for _ in range(3):
                    critic_resp_str = self.critic.generate({"task_desc": self.task_desc, "plan": draft_plan})
                    try:
                        critic_resp = json.loads(critic_resp_str)
                        if critic_resp["decision"] == "PASS":
                            critic_valid = True
                            break
                        fb = f"Critic rejected: {critic_resp['reason']}"
                        if critic_resp.get("suggestion"):
                            fb += f" | Fix: {critic_resp['suggestion']}"
                        draft_plan = self.planner.generate({"task_desc": self.task_desc, "feedback": fb})
                    except Exception as e:
                        self._log(f"[System] Critic parse failed: {e}")
                        break

                if not critic_valid:
                    self._log("[System] Critic rejected plan. Proceeding anyway.")

                self.current_plan = draft_plan
                self._record_step(iter_id, "Planner", input_data=plan_ctx, output_data={"plan": self.current_plan})
                self._record_round_agent("Planner", self.current_plan)
                self._record_round_agent("Critic", critic_resp_str)
                self._save_artifact(f"iter_{iter_id}_plan.md", self.current_plan)
                ticket = "Architect"

            # ==================== Architect ====================
            if ticket == "Architect":
                self._log(">>> [Agent] Architect...")
                arch_query = f"{self.task_desc}\n\nPLAN:\n{self.current_plan}"
                arch_ctx = self._build_context_with_memory(
                    base_context={
                        "task_desc": self.task_desc,
                        "plan": self.current_plan,
                        "previous_skeleton": self.current_skeleton if self.current_skeleton.strip() else None,
                        "feedback": feedback.get("feedback") if isinstance(feedback, dict) and feedback.get("ticket") == "Architect" else None,
                        "data_layout": self.data_layout,
                        "package_list": self.package_list,
                    },
                    agent_role="Architect",
                    current_ticket="Architect",
                    retrieval_query=arch_query,
                )
                for attempt in range(3):
                    arch_resp = self.architect.generate(arch_ctx)
                    extracted = extract_json(arch_resp)
                    try:
                        arch_json = json.loads(extracted)
                        if isinstance(arch_json, dict) and "skeleton_code" in arch_json:
                            self.current_skeleton = arch_json["skeleton_code"]
                        else:
                            self.current_skeleton = arch_resp
                    except Exception:
                        self.current_skeleton = arch_resp
                    self.current_skeleton = extract_python(self.current_skeleton)
                    is_valid, err = self._validate_skeleton(self.current_skeleton)
                    if is_valid:
                        self._log("  Skeleton validated.")
                        break
                    self._log(f"  Skeleton invalid (attempt {attempt+1}): {err}")
                    arch_ctx["feedback"] = f"Previous skeleton invalid: {err}. Output valid Python code."
                else:
                    raise RuntimeError("Failed to generate valid skeleton after 3 attempts.")

                self.function_list = self._parse_functions_from_skeleton(self.current_skeleton)
                self._log(f"  Functions: {self.function_list}")
                self.current_code = self.current_skeleton
                self._record_step(iter_id, "Architect", input_data=arch_ctx, output_data={"skeleton": self.current_skeleton})
                self._record_round_agent("Architect", self.current_skeleton)
                self._save_artifact(f"iter_{iter_id}_skeleton.py", self.current_skeleton)
                ticket = "Coder"

            # ==================== Coder ====================
            if ticket == "Coder":
                self._log(">>> [Agent] Coder...")
                coding_tasks, is_patch_mode, target = self._build_coding_tasks(feedback)

                if not is_patch_mode:
                    self.current_code = self.current_skeleton

                self._detect_stuck()

                for task_type, task_name_item in coding_tasks:
                    self._log(f"  [Coder] {task_type}" + (f": {task_name_item}" if task_name_item else ""))
                    coder_query = self._build_coder_query(task_type, task_name_item, is_patch_mode, feedback)
                    ctx = {
                        "target_type": task_type,
                        "skeleton_code": self.current_skeleton,
                        "current_full_code": self.current_code,
                        "plan": self.current_plan,
                        "task_desc": self.task_desc,
                        "package_list": self.package_list,
                        "data_layout": self.data_layout,
                        "feedback": feedback.get("feedback") if is_patch_mode and isinstance(feedback, dict) else None,
                        "analysis": feedback.get("analysis") if is_patch_mode and isinstance(feedback, dict) else None,
                        "fix_target": target if is_patch_mode else None,
                    }
                    if task_type == "function":
                        ctx["target_function"] = task_name_item
                    ctx = self._build_context_with_memory(ctx, "Coder", "Coder", retrieval_query=coder_query)
                    self.current_code = self.coder.implement_and_merge(ctx)
                    self._record_step(iter_id, "Coder", input_data=ctx,
                                      output_data={"task_type": task_type, "target": task_name_item, "code": self.current_code})
                    self.current_code = extract_python(self.current_code)
                    with open(os.path.join(self.sandbox_dir, "solver.py"), "w", encoding="utf-8") as f:
                        f.write(self.current_code)

                self.current_code = extract_python(self.current_code)
                with open(os.path.join(self.sandbox_dir, "solver.py"), "w", encoding="utf-8") as f:
                    f.write(self.current_code)

                self._record_round_agent("Coder", self.current_code)

                syntax_ok = self._syntax_check_loop(iter_id)
                if not syntax_ok:
                    self._finalize_round(success=False)
                    ticket = "Planner"
                    self.retry_count += 1
                    self._reset_downstream_state("Planner")
                    continue

                self._save_artifact(f"iter_{iter_id}_solver.py", self.current_code)
                ticket = "Execution"

            # ==================== Execution & Judge ====================
            if ticket == "Execution":
                self._log(">>> [System] Executing...")
                output_file = os.path.join(self.sandbox_dir, "output.npy")
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except OSError:
                        pass

                success, stdout, stderr = run_cmd(self.python_path, self.sandbox_dir, "solver.py", timeout=self.execution_timeout)

                if not success and stderr:
                    success, stdout, stderr = self._quick_fix(success, stdout, stderr)

                logs = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                self._save_artifact(f"iter_{iter_id}_exec_log.txt", logs)

                metrics = None
                eval_success = False

                if success:
                    self._log("  Execution success. Running eval...")
                    self._auto_fix_output_shape()
                    e_ok, e_out, e_err = run_cmd(self.python_path, self.sandbox_dir, "eval_script.py", args=["output.npy"])
                    if e_ok:
                        try:
                            metrics = json.loads(e_out)
                            eval_success = True
                            self._log(f"  Metrics: {metrics}")
                        except Exception:
                            logs += f"\n\n[Eval Error] JSON parse failed: {e_out}"
                    else:
                        logs += f"\n\n[Eval Error] Script failed: {e_err}"
                else:
                    self._log(f"  Execution failed: {stderr}")

                self._record_step(iter_id, "Execution", input_data="solver.py",
                                  output_data={"success": success, "eval_success": eval_success, "metrics": metrics})
                self._record_round_execution(success, stdout, stderr, metrics)

                if eval_success and metrics:
                    if self._check_threshold(metrics):
                        self._finalize_round(success=True)
                        self._on_success(metrics, logs)
                        return True

                result = self._judge(logs, metrics, stderr, iter_id)
                if result is None:
                    self._finalize_round(success=False)
                    ticket = "Coder"
                    feedback = {"analysis": "Judge output invalid."}
                    self.retry_count += 1
                    continue
                self._record_round_judge(result)
                self._finalize_round(success=False)
                self._reset_downstream_state(result["ticket_assigned_to"])
                ticket = result["ticket_assigned_to"]
                feedback = result
                self.retry_count += 1

        self._log("FAILED after max retries.")
        self._on_failure()
        return False

    # ---- helpers ----
    def _load_data_shapes(self):
        """Load input/output shapes from sandbox dataset."""
        input_shape = None
        output_shape = None
        dataset_dir = os.path.join(self.sandbox_dir, "dataset")
        try:
            gt_path = os.path.join(dataset_dir, "gt_output.npy")
            if os.path.exists(gt_path):
                gt = np.load(gt_path, allow_pickle=True)
                output_shape = gt.shape
                self._log(f"  GT output shape: {output_shape}, dtype: {gt.dtype}")
            # Scan dataset/ for input .npz and .npy files
            for fname in sorted(os.listdir(dataset_dir)):
                fpath = os.path.join(dataset_dir, fname)
                if fname.startswith("raw_data") and fname.endswith(".npz"):
                    npz = np.load(fpath)
                    for key in npz.keys():
                        arr = npz[key]
                        input_shape = arr.shape
                        self._log(f"  Input data: {fname}['{key}'], shape: {input_shape}, dtype: {arr.dtype}")
                        break
                    break
                elif fname.endswith(".npy") and fname != "gt_output.npy":
                    arr = np.load(fpath, allow_pickle=True)
                    input_shape = arr.shape
                    self._log(f"  Input data: {fname}, shape: {input_shape}, dtype: {arr.dtype}")
                    break
        except Exception as e:
            self._log(f"  Warning: failed to load data shapes: {e}")
        return input_shape, output_shape

    def _build_data_layout(self) -> str:
        """Scan sandbox dataset/ and build a human-readable layout string for agent context."""
        dataset_dir = os.path.join(self.sandbox_dir, "dataset")
        parts = []

        # Try data_info.json first (generated by setup_task_sandbox)
        data_info_path = os.path.join(dataset_dir, "data_info.json")
        if os.path.exists(data_info_path):
            with open(data_info_path, "r", encoding="utf-8") as f:
                data_info = json.load(f)
            for fname, info in data_info.items():
                if fname == "gt_output.npy":
                    continue  # Don't expose GT path to solver
                if isinstance(info, dict) and "shape" in info:
                    parts.append(f"dataset/{fname} shape={info['shape']} dtype={info['dtype']}")
                else:
                    for key, kinfo in info.items():
                        parts.append(f"dataset/{fname} key='{key}' shape={kinfo['shape']} dtype={kinfo['dtype']}")
        else:
            # Fallback: scan files
            for fname in sorted(os.listdir(dataset_dir)):
                fpath = os.path.join(dataset_dir, fname)
                if fname in ("gt_output.npy", "gt_key.txt", "data_info.json"):
                    continue
                if fname.endswith(".npz") and os.path.isfile(fpath):
                    try:
                        npz = np.load(fpath)
                        for k in npz.keys():
                            arr = npz[k]
                            parts.append(f"dataset/{fname} key='{k}' shape={arr.shape} dtype={arr.dtype}")
                    except Exception:
                        parts.append(f"dataset/{fname}")
                elif os.path.isfile(fpath):
                    parts.append(f"dataset/{fname}")

        meta_path = os.path.join(dataset_dir, "meta_data.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta_summary = ", ".join(f"{k}={v}" for k, v in meta.items()
                                         if k != "description")
                parts.append(f"dataset/meta_data.json ({meta_summary})")
            except Exception:
                parts.append("dataset/meta_data.json (physical parameters)")
        if not parts:
            parts.append("No data files found in dataset/")
        layout = "; ".join(parts) + ". Save output to output.npy"
        self._log(f"  Data layout: {layout}")
        return layout

    def _build_coding_tasks(self, feedback):
        coding_tasks = [("imports", None), *[("function", fn) for fn in self.function_list], ("main_block", None)]
        is_patch_mode = False
        target = None

        if isinstance(feedback, dict) and feedback.get("ticket_assigned_to") == "Coder":
            raw_target = feedback.get("fix_target", "")

            # Parse comma-separated targets from Judge
            resolved_targets = []
            if raw_target:
                candidates = [t.strip() for t in raw_target.split(",") if t.strip()]
                for candidate in candidates:
                    # Extract identifier from each candidate
                    match = re.search(r"\b([_a-zA-Z0-9]+)\b", candidate)
                    if match:
                        name = match.group(1)
                        if name in self.function_list or name in ["imports", "main_block", "main"]:
                            resolved_targets.append(name)
                        elif "." in candidate:
                            parts = candidate.split(".")
                            if parts[-1] in self.function_list:
                                resolved_targets.append(parts[-1])

            # Fallback: search analysis text for function names
            if not resolved_targets:
                analysis_text = feedback.get("analysis", "") + " " + feedback.get("evidence", "")
                for func in self.function_list:
                    if func in analysis_text:
                        resolved_targets.append(func)

            # Deduplicate while preserving order
            seen = set()
            unique_targets = []
            for t in resolved_targets:
                if t not in seen:
                    seen.add(t)
                    unique_targets.append(t)
            resolved_targets = unique_targets

            if resolved_targets:
                target = ",".join(resolved_targets)
                self._log(f"  Patch mode: {target}")
                is_patch_mode = True
                patch_tasks = []
                for t in resolved_targets:
                    if t in self.function_list:
                        patch_tasks.append(("function", t))
                    elif t in ("imports", "main_block", "main"):
                        patch_tasks.append(("main_block" if t in ("main_block", "main") else t, None))
                if patch_tasks:
                    coding_tasks = patch_tasks
                else:
                    # Fallback: try substring matching
                    detected = []
                    for t in resolved_targets:
                        if "main" in t.lower():
                            detected.append(("main_block", None))
                        for fn in self.function_list:
                            if fn in t:
                                detected.append(("function", fn))
                    if detected:
                        coding_tasks = detected
                    else:
                        is_patch_mode = False

        return coding_tasks, is_patch_mode, target

    def _build_coder_query(self, task_type, task_name_item, is_patch_mode, feedback):
        sig = None
        if task_type == "function" and self.current_skeleton:
            sig = self._extract_function_signature(self.current_skeleton, task_name_item)
        plan_summary = self.current_plan[:500] if self.current_plan else "No Plan"
        if sig:
            q = f"Python Implementation for:\n{sig}\n\nContext: {self.task_name}\nPlan: {plan_summary}"
        else:
            q = f"Python Implementation for: {task_name_item or 'Global'}\nContext: {self.task_name}\nPlan: {plan_summary}"
        if is_patch_mode and isinstance(feedback, dict) and feedback.get("analysis"):
            q += f"\nError: {feedback['analysis']}"
        return q

    def _detect_stuck(self):
        if len(self.failure_history) >= 3:
            recent = [h.get("analysis", "")[:80] for h in self.failure_history[-3:]]
            if len(set(recent)) == 1:
                self._log("  STUCK: Same error 3x. Breaking loop.")
                self.failure_history = self.failure_history[-1:]

        if len(self.failure_history) >= 4:
            recent_tickets = [h.get("ticket_assigned_to", "") for h in self.failure_history[-4:]]
            if len(set(recent_tickets)) == 1 and recent_tickets[0] == "Coder":
                self._log("  STUCK: Coder 4x. Escalating to Planner.")
                self.failure_history[-1]["ticket_assigned_to"] = "Planner"
                self.failure_history[-1]["feedback"] = "Coder failed 4x. Propose a DIFFERENT and SIMPLER approach."

        # Metrics-based stuck detection: consecutive very low NCC indicates fundamental algorithm failure
        if len(self.failure_history) >= 2:
            recent_ncc = []
            for h in self.failure_history[-2:]:
                m = h.get("metrics")
                if isinstance(m, dict) and "ncc" in m:
                    recent_ncc.append(m["ncc"])
            if len(recent_ncc) >= 2 and all(ncc < 0.1 for ncc in recent_ncc):
                self._log("  STUCK (metrics): NCC < 0.1 for 2 consecutive iterations. "
                          "Fundamental algorithm issue detected. Escalating to Planner with reset guidance.")
                self.failure_history[-1]["ticket_assigned_to"] = "Planner"
                self.failure_history[-1]["feedback"] = (
                    "CRITICAL: Reconstruction has no correlation with ground truth for multiple iterations. "
                    "This is a FUNDAMENTAL algorithm/implementation issue. You MUST: "
                    "(1) Verify ALL forward model operator formulas have correct signs — check EACH operator. "
                    "(2) Verify loss normalization matches the original plan exactly. "
                    "(3) Verify masks are smooth (sigmoid/exponential) where specified, not hard binary cutoffs. "
                    "(4) Simplify: remove any features not in the original plan (no adaptive lr, no extra scheduling). "
                    "(5) Propose a SIMPLER implementation with fewer moving parts."
                )

    def _syntax_check_loop(self, iter_id: int) -> bool:
        max_syn = 5
        for attempt in range(max_syn):
            ok, _, err = run_cmd(self.python_path, self.sandbox_dir, "solver.py",
                                 check_syntax_only=True, syntax_check_timeout=self.syntax_check_timeout)
            if ok:
                self._log(f"  Syntax OK (attempt {attempt+1})")
                return True
            self._log(f"  Syntax error (attempt {attempt+1}/{max_syn})")
            if attempt >= 2:
                ctx = {
                    "target_type": "full_rewrite",
                    "skeleton_code": self.current_skeleton,
                    "current_full_code": self.current_skeleton,
                    "plan": self.current_plan,
                    "task_desc": self.task_desc,
                    "package_list": self.package_list,
                    "data_layout": self.data_layout,
                    "feedback": f"PERSISTENT syntax errors after {attempt+1} attempts. Start FRESH.\n{err[-500:]}",
                }
            else:
                ctx = {
                    "target_type": "full_rewrite",
                    "skeleton_code": self.current_skeleton,
                    "plan": self.current_plan,
                    "data_layout": self.data_layout,
                    "feedback": f"SYNTAX ERROR (attempt {attempt+1}):\n{err}",
                }
            ctx = self._build_context_with_memory(ctx, "Coder", "Coder")
            self.current_code = extract_python(self.coder.implement_and_merge(ctx))
            with open(os.path.join(self.sandbox_dir, "solver.py"), "w", encoding="utf-8") as f:
                f.write(self.current_code)
            self._record_step(iter_id, "Coder", input_data=ctx,
                              output_data={"task_type": "syntax_fix", "error": err, "code": self.current_code})

        record = {
            "iteration": self.retry_count + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ticket_assigned_to": "Planner",
            "analysis": f"Persistent syntax errors after {max_syn} retries.",
            "evidence": err[-500:],
            "feedback": "Propose a SIMPLER algorithm that is easier to implement.",
        }
        self.failure_history.append(record)
        return False

    def _quick_fix(self, success, stdout, stderr):
        patterns = [
            ("can only convert an array of size 1", "Replace .item() with safe pattern: data = raw.item() if raw.ndim == 0 else raw"),
            ("unexpected keyword argument", "Remove unsupported keyword argument."),
            ("No module named", "Module not available. Use numpy/scipy alternative."),
            ("cannot import name", "Import path incorrect. Check the correct API."),
        ]
        for pat, hint in patterns:
            if pat in stderr:
                self._log(f"  Quick-fix: {pat}")
                ctx = {
                    "target_type": "full_rewrite",
                    "skeleton_code": self.current_skeleton,
                    "current_full_code": self.current_code,
                    "plan": self.current_plan,
                    "task_desc": self.task_desc,
                    "package_list": self.package_list,
                    "data_layout": self.data_layout,
                    "feedback": f"RUNTIME CRASH:\n{stderr[-800:]}\n\nFIX: {hint}",
                }
                ctx = self._build_context_with_memory(ctx, "Coder", "Coder")
                self.current_code = extract_python(self.coder.implement_and_merge(ctx))
                with open(os.path.join(self.sandbox_dir, "solver.py"), "w", encoding="utf-8") as f:
                    f.write(self.current_code)
                return run_cmd(self.python_path, self.sandbox_dir, "solver.py", timeout=self.execution_timeout)
        return success, stdout, stderr

    def _auto_fix_output_shape(self):
        try:
            import numpy as np
            out_path = os.path.join(self.sandbox_dir, "output.npy")
            gt_path = os.path.join(self.sandbox_dir, "dataset", "gt_output.npy")
            if not (os.path.exists(out_path) and os.path.exists(gt_path)):
                return
            pred = np.load(out_path, allow_pickle=True)
            gt = np.load(gt_path, allow_pickle=True)
            if pred.shape == gt.shape:
                return
            self._log(f"  Shape fix: pred={pred.shape} vs gt={gt.shape}")
            squeezed = np.squeeze(pred)
            if squeezed.shape == gt.shape:
                np.save(out_path, squeezed.astype(np.float64))
                return
            if pred.size == gt.size:
                np.save(out_path, pred.reshape(gt.shape).astype(np.float64))
        except Exception as e:
            self._log(f"  Shape fix error: {e}")

    def _check_threshold(self, metrics: dict) -> bool:
        ncc = metrics.get("ncc", 0)
        nrmse = metrics.get("nrmse", float('inf'))
        self._log(f"  NCC={ncc:.4f} (min={self.min_ncc}), NRMSE={nrmse:.4f} (max={self.max_nrmse})")
        return ncc >= self.min_ncc and nrmse <= self.max_nrmse

    def _judge(self, logs, metrics, stderr, iter_id) -> dict | None:
        self._log(">>> [Agent] Judge...")
        judge_ctx = self._build_context_with_memory(
            base_context={
                "task_desc": self.task_desc,
                "logs": logs[-1000:],
                "metrics": metrics,
                "current_code_snippet": self.current_code,
                "data_layout": self.data_layout,
                "package_list": self.package_list,
                "plan": getattr(self, 'current_plan', None),
                "eval_thresholds": f"NCC >= {self.min_ncc}, NRMSE <= {self.max_nrmse}",
            },
            agent_role="Judge",
            current_ticket="Judge",
            retrieval_query=f"{self.task_desc}\n\nERROR:\n{stderr[-500:]}\nMETRICS: {metrics}",
        )

        result = None
        for attempt in range(2):
            if attempt > 0:
                self._log("  Judge retry (previous output was not valid JSON)...")
                judge_ctx["feedback"] = (
                    "Your previous output was NOT valid JSON and could not be parsed. "
                    "You MUST output ONLY a single valid JSON object. "
                    "Do NOT include unescaped newlines or special characters inside string values. "
                    "Keep all string values concise and on a single line."
                )
            judgment = self.judge.generate(judge_ctx)
            self._save_artifact(f"iter_{iter_id}_judge.json", judgment)
            try:
                extracted = extract_json(judgment)
                result = json.loads(extracted)
                break
            except Exception as e:
                # Try to repair common JSON issues before giving up
                try:
                    repaired = self._repair_json(extracted if 'extracted' in dir() else judgment)
                    result = json.loads(repaired)
                    self._log(f"  Judge JSON repaired successfully.")
                    break
                except Exception:
                    pass
                self._log(f"  Judge parse error (attempt {attempt+1}): {e}")

        if result is None:
            return None

        if "evidence" not in result:
            result["evidence"] = "MISSING"
            result["ticket_assigned_to"] = "Coder"
        self._log(f"  Judge -> {result['ticket_assigned_to']}: {result.get('analysis', 'N/A')[:100]}")
        self._record_step(iter_id, "Judge", input_data=judge_ctx, output_data=result)
        self.failure_history.append({
            "iteration": self.retry_count + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ticket_assigned_to": result["ticket_assigned_to"],
            "fix_target": result.get("fix_target"),
            "analysis": result.get("analysis", ""),
            "evidence": result.get("evidence", ""),
            "feedback": result.get("feedback", ""),
            "metrics": metrics,
        })
        return result

    @staticmethod
    def _repair_json(text: str) -> str:
        """Attempt to repair common JSON issues from LLM output."""
        # Remove control characters that break JSON (except \n \r \t)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)

        # Fix unescaped newlines inside JSON string values
        # Strategy: find string boundaries and escape internal newlines
        in_string = False
        escape_next = False
        chars = list(cleaned)
        for i, ch in enumerate(chars):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string and ch == '\n':
                chars[i] = '\\n'
            elif in_string and ch == '\r':
                chars[i] = '\\r'
            elif in_string and ch == '\t':
                chars[i] = '\\t'
        repaired = ''.join(chars)

        # Try parsing; if it fails, try truncating at the last complete key-value
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

        # Try to close truncated JSON by finding last valid closing point
        # Find last complete "key": "value" and close the object
        last_brace = repaired.rfind('}')
        if last_brace > 0:
            candidate = repaired[:last_brace + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        return repaired

    # ---- Trajectory Writing ----
    def _write_trajectories(self, final_outcome: str):
        """Write all round trajectories to data/<task>/trajectories/<exp_id>.json"""
        paths_cfg = self.config.get("paths", {})
        traj_root = paths_cfg.get("trajectories_dir", "./data")
        traj_dir = os.path.join(traj_root, self.task_name, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        payload = {
            "exp_id": self.exp_id,
            "task_name": self.task_name,
            "task_desc": self.task_desc[:3000],
            "final_outcome": final_outcome,
            "total_rounds": len(self.round_trajectories),
            "all_skills_used_ids": list(self.used_knowledge_ids),
            "rounds": self.round_trajectories,
        }

        path = os.path.join(traj_dir, f"{self.exp_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            self._log(f"  [Trajectories] Written to {path}")
        except Exception as e:
            self._log(f"  [Trajectories] Failed to write: {e}")

    # ---- Post-task Skills Analysis ----
    def _post_task_skills_analysis(self, final_outcome: str):
        """After task ends, analyze trajectories and manage skills lifecycle."""
        if not getattr(self.skill_manager, "learning_enabled", False):
            self._log("  [Skills] Learning disabled, skipping post-task analysis.")
            return

        try:
            # 0. Generate code diff analysis report (if reference code exists)
            code_diff_report = ""
            solver_code = ""
            solver_path = os.path.join(self.sandbox_dir, "solver.py")
            if os.path.isfile(solver_path):
                try:
                    with open(solver_path, "r", encoding="utf-8") as f:
                        solver_code = f.read()
                except Exception as e:
                    self._log(f"  [Skills] Failed to read solver.py: {e}")

            if self.task_dir and solver_code:
                from agents.code_diff_analyzer import CodeDiffAnalyzerAgent, load_reference_code
                reference_code = load_reference_code(self.task_dir)
                if reference_code:
                    self._log("  [Skills] Generating code diff analysis report...")
                    # Collect last execution error and metrics for context
                    last_error = ""
                    last_metrics = None
                    if self.failure_history:
                        last = self.failure_history[-1]
                        last_error = last.get("evidence", "")
                        last_metrics = last.get("metrics")
                    diff_analyzer = CodeDiffAnalyzerAgent(self.client, self.model_name)
                    try:
                        diff_raw = diff_analyzer.generate({
                            "solver_code": solver_code,
                            "reference_code": reference_code,
                            "task_desc": self.task_desc,
                            "execution_error": last_error,
                            "metrics": last_metrics,
                        })
                        code_diff_report = diff_raw
                        self._log(f"  [Skills] Code diff report generated ({len(code_diff_report)} chars)")
                        self._save_artifact("code_diff_report.json", code_diff_report)
                    except Exception as e:
                        self._log(f"  [Skills] Code diff analysis failed: {e}")
                else:
                    self._log("  [Skills] No reference code found in task_dir, skipping diff analysis.")

            # 1. Distill new skills from trajectories + code diff report
            new_skills = self.skill_manager.distill_from_trajectories(
                task_name=self.task_name,
                task_desc=self.task_desc,
                trajectories=self.round_trajectories,
                final_outcome=final_outcome,
                code_diff_report=code_diff_report,
            )

            new_skill_ids = {s.id for s in new_skills}
            self._log(f"  [Skills] Distilled {len(new_skills)} new draft skills:")
            for s in new_skills:
                self._log(f"    - {s.title} [{s.category}] [{s.scope}]")

            if final_outcome == "success":
                # 2. Check first-run success (no prior skills were used)
                # "First-run" means no draft/permanent skills existed or none were used
                prior_skills_used = self.used_knowledge_ids - new_skill_ids
                is_first_run_no_skills = len(prior_skills_used) == 0

                if is_first_run_no_skills and new_skills:
                    # Promote ALL newly distilled skills to permanent
                    self._log("  [Skills] First-run success with no prior skills used. Promoting all new skills.")
                    for s in new_skills:
                        self.skill_manager.store.promote_to_permanent(s.id)
                else:
                    # 3. Promote only USED draft skills to permanent
                    promoted = self.skill_manager.promote_used_skills(
                        self.used_knowledge_ids, self.task_name
                    )
                    self._log(f"  [Skills] Promoted {len(promoted)} used draft skills to permanent.")

                # 4. Cleanup: delete remaining drafts for this task, but EXCLUDE newly distilled ones
                deleted = self.skill_manager.cleanup_draft_skills(self.task_name, exclude_ids=new_skill_ids)
                self._log(f"  [Skills] Cleaned up {deleted} remaining draft skills for task '{self.task_name}'.")

            else:
                self._log("  [Skills] Task failed. Draft skills retained for future reference.")

        except Exception as e:
            self._log(f"  [Skills] Post-task analysis failed: {e}")
            import traceback
            self._log(traceback.format_exc())

    # ---- Success/Failure handlers ----
    def _on_success(self, metrics, logs):
        self._save_snapshot(self.retry_count + 1, "final_success", {"metrics": metrics})
        self._write_trajectories("success")
        self._post_task_skills_analysis("success")
        self.generate_knowledge_report(success=True)

    def _on_failure(self):
        # Finalize last round if still open
        if self._current_round:
            self._finalize_round(success=False)
        self._write_trajectories("failure")
        self._post_task_skills_analysis("failure")
        self.failure_history.clear()
        self.generate_knowledge_report(success=False)
