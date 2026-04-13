"""
core/workflow.py -- Main pipeline workflow.

Assumes the sandbox is pre-configured with:
  - data/gt_output.npy, data/input_data/, data/meta_data.json
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
import shutil
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

                # Structural validation (nested classes, orphaned methods, etc.)
                self.current_code, struct_issues = self._structural_validate_and_fix(self.current_code)
                if struct_issues:
                    with open(os.path.join(self.sandbox_dir, "solver.py"), "w", encoding="utf-8") as f:
                        f.write(self.current_code)

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
                output_file = os.path.join(self.sandbox_dir, "output.npz")
                npy_file = os.path.join(self.sandbox_dir, "output.npy")
                for f in (output_file, npy_file):
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except OSError:
                            pass

                success, stdout, stderr = run_cmd(self.python_path, self.sandbox_dir, "solver.py", timeout=self.execution_timeout)

                if not success and stderr:
                    success, stdout, stderr = self._quick_fix(success, stdout, stderr)

                # Recover from timeout if solver saved a checkpoint output.npz
                if not success and "TIMEOUT" in stderr:
                    if os.path.exists(output_file):
                        self._log("  Execution timed out, but checkpoint output.npz found. Treating as partial success.")
                        success = True
                        stderr = f"TIMEOUT (recovered from checkpoint)\n{stderr}"

                logs = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                self._save_artifact(f"iter_{iter_id}_exec_log.txt", logs)

                metrics = None
                eval_success = False

                if success:
                    # Backward compat: auto-convert output.npy → output.npz
                    if not os.path.exists(output_file) and os.path.exists(npy_file):
                        import numpy as np
                        arr = np.load(npy_file, allow_pickle=True)
                        np.savez(output_file, output=arr)
                        self._log("  Auto-converted output.npy → output.npz (key='output')")

                    self._log("  Execution success. Running eval...")
                    self._auto_fix_output_shape()
                    e_ok, e_out, e_err = run_cmd(self.python_path, self.sandbox_dir, "eval_script.py", args=["output.npz"])
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
                    # Check for missing required metrics — regenerate eval_script if needed
                    missing = self._detect_missing_metrics(metrics)
                    if missing:
                        self._log(f"  [System] Required metrics missing: {missing}")
                        regen_ok = self._regenerate_eval_script(missing, metrics)
                        if regen_ok:
                            # Re-run eval with regenerated script
                            e_ok, e_out, e_err = run_cmd(self.python_path, self.sandbox_dir, "eval_script.py", args=["output.npz"])
                            if e_ok:
                                try:
                                    metrics = json.loads(e_out)
                                    self._log(f"  Metrics (after regen): {metrics}")
                                except Exception:
                                    self._log(f"  [Eval Error] JSON parse failed after regen: {e_out}")
                            else:
                                self._log(f"  [Eval Error] Regenerated script failed: {e_err}")

                    passed = self._check_threshold(metrics)

                    # Determine primary metric for best-result tracking
                    primary_score = self._get_primary_score(metrics)

                    # --- Metric Regression Guard ---
                    # Only protect good solutions (score >= 0.5). For poor initial solutions,
                    # exploration is more valuable than preservation.
                    regression_detected = False
                    if self.best_primary_score >= 0.5:  # Only apply when we have a working solution
                        # Detect significant regression: score dropped by >50% or sign flipped
                        is_regression = False
                        if self.best_primary_score > 0 and primary_score < self.best_primary_score * 0.5:
                            is_regression = True  # Positive score dropped by >50%
                        elif self.best_primary_score > 0 and primary_score < 0:
                            is_regression = True  # Score flipped from positive to negative
                        elif self.best_primary_score < 0 and primary_score < self.best_primary_score * 1.5:
                            is_regression = True  # Negative score got worse by >50%

                        if is_regression:
                            self._log(f"  REGRESSION: score={primary_score:.4f} << best={self.best_primary_score:.4f} (iter {self.best_iteration}). Reverting to best code.")
                            regression_detected = True
                            # Find and restore the best solver code from snapshot
                            best_solver_path = self._find_best_solver_artifact()
                            if best_solver_path:
                                with open(best_solver_path, "r", encoding="utf-8") as bf:
                                    self.current_code = bf.read()
                                with open(os.path.join(self.sandbox_dir, "solver.py"), "w", encoding="utf-8") as f:
                                    f.write(self.current_code)
                                self._log(f"  Reverted solver.py to iter {self.best_iteration} code.")
                            logs += (f"\n\nREGRESSION DETECTED: Current score={primary_score:.4f} is much worse "
                                     f"than best score={self.best_primary_score:.4f} (iter {self.best_iteration}). "
                                     f"Code has been reverted to the best version. "
                                     f"Make a SMALL, CONSERVATIVE change to improve upon the best result.")

                    # Track best result across iterations
                    if (passed and not self.best_passed) or \
                       (passed == self.best_passed and primary_score > self.best_primary_score):
                        self.best_primary_score = primary_score
                        self.best_iteration = iter_id
                        self.best_passed = passed
                        self.best_metrics = metrics.copy()
                        best_path = os.path.join(self.sandbox_dir, "best_output.npz")
                        src_path = os.path.join(self.sandbox_dir, "output.npz")
                        if os.path.exists(src_path):
                            shutil.copy2(src_path, best_path)
                            self._log(f"  Best result saved (iter {iter_id}, score={primary_score:.4f})")

                    if passed:
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
                # Keep failure_history to maintain context for stuck detection.
                # The STUCK escalation path (line 161) clears it when appropriate.
                self.retry_count += 1

        self._log("FAILED after max retries.")
        self._on_failure()
        return False

    # ---- helpers ----
    def _load_data_shapes(self):
        """Load input/output shapes from sandbox dataset."""
        input_shape = None
        output_shape = None
        data_dir = os.path.join(self.sandbox_dir, "data")
        try:
            # GT files are at sandbox root (not data/) for solver isolation
            gt_npz_path = os.path.join(self.sandbox_dir, "ground_truth.npz")
            gt_npy_path = os.path.join(self.sandbox_dir, "gt_output.npy")
            if os.path.exists(gt_npz_path):
                gt_npz = np.load(gt_npz_path)
                first_key = list(gt_npz.keys())[0]
                output_shape = gt_npz[first_key].shape
                self._log(f"  GT output shape: {output_shape}")
            elif os.path.exists(gt_npy_path):
                gt = np.load(gt_npy_path, allow_pickle=True)
                output_shape = gt.shape
                self._log(f"  GT output shape: {output_shape}, dtype: {gt.dtype}")
            # Scan data/ for input .npz and .npy files
            for fname in sorted(os.listdir(data_dir)):
                fpath = os.path.join(data_dir, fname)
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
        """Scan sandbox data/ and build a human-readable layout string for agent context."""
        data_dir = os.path.join(self.sandbox_dir, "data")
        parts = []

        # Try data_info.json first (generated by setup_task_sandbox)
        data_info_path = os.path.join(data_dir, "data_info.json")
        if os.path.exists(data_info_path):
            with open(data_info_path, "r", encoding="utf-8") as f:
                data_info = json.load(f)
            for fname, info in data_info.items():
                if fname == "gt_output.npy":
                    continue  # Don't expose GT path to solver
                if isinstance(info, dict) and "shape" in info:
                    parts.append(f"data/{fname} shape={info['shape']} dtype={info['dtype']}")
                else:
                    for key, kinfo in info.items():
                        parts.append(f"data/{fname} key='{key}' shape={kinfo['shape']} dtype={kinfo['dtype']}")
        else:
            # Fallback: scan files
            for fname in sorted(os.listdir(data_dir)):
                fpath = os.path.join(data_dir, fname)
                if fname in ("gt_output.npy", "gt_key.txt", "data_info.json", "output_keys.json", "gt_keys.json",
                             "ground_truth.npz", "baseline_reference.npz", "ground_truth.npy"):
                    continue
                if fname.endswith(".npz") and os.path.isfile(fpath):
                    try:
                        npz = np.load(fpath)
                        for k in npz.keys():
                            arr = npz[k]
                            parts.append(f"data/{fname} key='{k}' shape={arr.shape} dtype={arr.dtype}")
                    except Exception:
                        parts.append(f"data/{fname}")
                elif os.path.isfile(fpath):
                    parts.append(f"data/{fname}")

        meta_path = os.path.join(data_dir, "meta_data.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta_summary = ", ".join(f"{k}={v}" for k, v in meta.items()
                                         if k != "description")
                parts.append(f"data/meta_data.json ({meta_summary})")
            except Exception:
                parts.append("data/meta_data.json (physical parameters)")
        if not parts:
            parts.append("No data files found in data/")
        # Append output format from output_keys.json if available
        ok_path = os.path.join(data_dir, "output_keys.json")
        if os.path.exists(ok_path):
            try:
                with open(ok_path, "r", encoding="utf-8") as f:
                    ok_info = json.load(f)
                save_instr = ok_info.get("save_instruction", "np.savez('output.npz', output=result)")
                key_descs = []
                for kn, ki in ok_info.get("keys", {}).items():
                    key_descs.append(f"'{kn}' shape={ki.get('shape','?')} dtype={ki.get('dtype','float64')}")
                keys_str = ", ".join(key_descs)
                layout = "; ".join(parts) + f". OUTPUT FORMAT: {save_instr} — keys: {keys_str}"
            except Exception:
                layout = "; ".join(parts) + ". Save output to output.npz using np.savez"
        else:
            layout = "; ".join(parts) + ". Save output to output.npz using np.savez('output.npz', output=result)"
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

    def _find_best_solver_artifact(self):
        """Find the solver.py artifact from the best iteration in the snapshot dir."""
        if self.best_iteration <= 0:
            return None
        # Artifact naming: {exp_id}_iter_{N}_solver.py
        pattern = f"_iter_{self.best_iteration}_solver.py"
        for fname in os.listdir(self.snapshot_dir):
            if fname.endswith(pattern):
                return os.path.join(self.snapshot_dir, fname)
        return None

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
        """Fix output shapes per-key using output_keys.json or GT reference."""
        try:
            import numpy as np
            out_path = os.path.join(self.sandbox_dir, "output.npz")
            if not os.path.exists(out_path):
                return

            # Load output_keys.json for expected shapes
            ok_path = os.path.join(self.sandbox_dir, "data", "output_keys.json")
            expected_keys = {}
            if os.path.exists(ok_path):
                with open(ok_path, "r", encoding="utf-8") as f:
                    ok_info = json.load(f)
                expected_keys = ok_info.get("keys", {})

            # Load GT reference for shape comparison (GT is at sandbox root, not data/)
            gt_npz_path = os.path.join(self.sandbox_dir, "ground_truth.npz")
            gt_npy_path = os.path.join(self.sandbox_dir, "gt_output.npy")
            gt_arrays = {}
            if os.path.exists(gt_npz_path):
                gt_npz = np.load(gt_npz_path)
                gt_arrays = {k: gt_npz[k] for k in gt_npz.keys()}
            elif os.path.exists(gt_npy_path):
                gt_arrays = {"output": np.load(gt_npy_path, allow_pickle=True)}

            pred_npz = np.load(out_path)
            pred_data = {k: pred_npz[k] for k in pred_npz.keys()}
            changed = False

            for key, arr in pred_data.items():
                # Determine expected shape from output_keys.json or GT
                exp_shape = None
                if key in expected_keys and "shape" in expected_keys[key]:
                    exp_shape = tuple(expected_keys[key]["shape"])
                elif key in gt_arrays:
                    exp_shape = gt_arrays[key].shape

                if exp_shape is None or arr.shape == exp_shape:
                    continue

                self._log(f"  Shape fix [{key}]: pred={arr.shape} vs expected={exp_shape}")
                squeezed = np.squeeze(arr)
                if squeezed.shape == exp_shape:
                    pred_data[key] = squeezed
                    changed = True
                elif arr.size == np.prod(exp_shape):
                    pred_data[key] = arr.reshape(exp_shape)
                    changed = True

            if changed:
                np.savez(out_path, **pred_data)
                self._log("  Shape fix applied to output.npz")
        except Exception as e:
            self._log(f"  Shape fix error: {e}")

    def _format_eval_thresholds(self) -> str:
        """Format eval boundaries for Judge context."""
        eval_boundaries = getattr(self, '_eval_boundaries', {})
        if eval_boundaries:
            LOWER_IS_BETTER = ("nrmse", "error", "mae", "fwhm", "lateral", "axial")
            parts = []
            for bk, bv in eval_boundaries.items():
                mk = bk.replace("_boundary_deg", "_deg").replace("_boundary_nm", "_nm").replace("_boundary", "")
                is_lower = any(kw in bk.lower() for kw in LOWER_IS_BETTER)
                op = "<=" if is_lower else ">="
                parts.append(f"{mk} {op} {bv}")
            return "; ".join(parts)
        return f"NCC >= {self.min_ncc}, NRMSE <= {self.max_nrmse}"

    def _check_threshold(self, metrics: dict) -> bool:
        """Check all evaluation boundaries, not just NCC/NRMSE."""
        LOWER_IS_BETTER = ("nrmse", "error", "mae", "fwhm", "lateral", "axial")

        # Use flexible eval_boundaries if available
        eval_boundaries = getattr(self, '_eval_boundaries', {})
        if eval_boundaries:
            matched_count = 0
            for boundary_key, boundary_val in eval_boundaries.items():
                # Derive metric key from boundary key
                metric_key = boundary_key.replace("_boundary_deg", "_deg") \
                                         .replace("_boundary_nm", "_nm") \
                                         .replace("_boundary", "")
                metric_val = metrics.get(metric_key)

                # Fuzzy fallback: if exact match fails, try substring matching
                if metric_val is None:
                    # Try finding a metric key that contains the derived key or vice versa
                    for mk, mv in metrics.items():
                        mk_norm = mk.lower().replace("_vs_ref", "").replace("_", "")
                        key_norm = metric_key.lower().replace("_", "")
                        if mk_norm == key_norm or key_norm in mk_norm or mk_norm in key_norm:
                            metric_val = mv
                            self._log(f"  [Boundary] Fuzzy matched: '{metric_key}' -> '{mk}'")
                            break

                if metric_val is None:
                    self._log(f"  [Boundary] WARN: metric '{metric_key}' not found in eval output, skipping boundary '{boundary_key}'")
                    continue
                matched_count += 1
                try:
                    metric_val = float(metric_val)
                    boundary_val = float(boundary_val)
                except (TypeError, ValueError):
                    continue
                # NaN/Inf check: NaN or Inf metric values always fail
                import math
                if math.isnan(metric_val) or math.isinf(metric_val):
                    self._log(f"  FAIL: {metric_key}={metric_val} (NaN/Inf is invalid)")
                    return False
                is_lower_better = any(kw in boundary_key.lower() for kw in LOWER_IS_BETTER)
                if is_lower_better:
                    if metric_val > boundary_val:
                        self._log(f"  FAIL: {metric_key}={metric_val:.4f} > {boundary_key}={boundary_val}")
                        return False
                else:
                    if metric_val < boundary_val:
                        self._log(f"  FAIL: {metric_key}={metric_val:.4f} < {boundary_key}={boundary_val}")
                        return False
            self._log(f"  All boundaries passed: {list(eval_boundaries.keys())}")
            return True

        # Fallback to legacy NCC/NRMSE check
        import math
        ncc = metrics.get("ncc", 0)
        nrmse = metrics.get("nrmse", float('inf'))
        try:
            ncc = float(ncc)
            nrmse = float(nrmse)
        except (TypeError, ValueError):
            return False
        if math.isnan(ncc) or math.isnan(nrmse) or math.isinf(ncc) or math.isinf(nrmse):
            self._log(f"  FAIL: NCC={ncc}, NRMSE={nrmse} (NaN/Inf is invalid)")
            return False
        self._log(f"  NCC={ncc:.4f} (min={self.min_ncc}), NRMSE={nrmse:.4f} (max={self.max_nrmse})")
        return ncc >= self.min_ncc and nrmse <= self.max_nrmse

    def _detect_missing_metrics(self, metrics: dict) -> list:
        """Check if any required boundary metrics are missing from eval output.

        Returns a list of missing metric keys. Empty list means all present.
        """
        eval_boundaries = getattr(self, '_eval_boundaries', {})
        if not eval_boundaries:
            return []

        missing = []
        for boundary_key in eval_boundaries:
            metric_key = boundary_key.replace("_boundary_deg", "_deg") \
                                     .replace("_boundary_nm", "_nm") \
                                     .replace("_boundary", "")
            metric_val = metrics.get(metric_key)

            # Fuzzy fallback
            if metric_val is None:
                for mk, mv in metrics.items():
                    mk_norm = mk.lower().replace("_vs_ref", "").replace("_", "")
                    key_norm = metric_key.lower().replace("_", "")
                    if mk_norm == key_norm or key_norm in mk_norm or mk_norm in key_norm:
                        metric_val = mv
                        break

            if metric_val is None:
                missing.append(metric_key)

        return missing

    def _regenerate_eval_script(self, missing_metrics: list, current_metrics: dict) -> bool:
        """Regenerate eval_script.py when required metrics are missing.

        Returns True if regeneration succeeded, False otherwise.
        """
        self._log(f"  [System] Regenerating eval_script.py (missing metrics: {missing_metrics})")

        try:
            from agents.sandbox_agents import EvalGenAgent

            eval_gen_agent = EvalGenAgent(self.client, self.model_name)

            # Build context for regeneration
            import numpy as np
            data_shape_hint = "N/A"
            # GT files are at sandbox root (not data/) for solver isolation
            gt_npz_path = os.path.join(self.sandbox_dir, "ground_truth.npz")
            gt_npy_path = os.path.join(self.sandbox_dir, "gt_output.npy")
            if os.path.exists(gt_npz_path):
                gt_npz = np.load(gt_npz_path)
                first_key = list(gt_npz.keys())[0]
                data_shape_hint = f"shape={gt_npz[first_key].shape}, dtype={gt_npz[first_key].dtype} (key='{first_key}')"
            elif os.path.exists(gt_npy_path):
                gt = np.load(gt_npy_path, allow_pickle=True)
                data_shape_hint = f"shape={gt.shape}, dtype={gt.dtype}"

            # Load meta_data if available
            meta_data = None
            meta_path = os.path.join(self.sandbox_dir, "data", "meta_data.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)

            # Extract eval context from task dir if available
            eval_extra = {}
            if self.task_dir:
                from run_task import _extract_eval_context
                eval_extra = _extract_eval_context(self.task_dir, self.sandbox_dir)

            eval_ctx = {
                "task_desc": self.task_desc,
                "data_shape_hint": data_shape_hint,
                "package_list": self.package_list,
                "meta_data": meta_data,
                "eval_thresholds": {"min_ncc": self.min_ncc, "max_nrmse": self.max_nrmse},
                **eval_extra,
                "feedback": (
                    f"CRITICAL: Your previous eval_script.py produced metrics: {current_metrics}, "
                    f"but the following REQUIRED metrics are MISSING: {missing_metrics}. "
                    f"The pipeline needs ALL of these metric keys in the JSON output. "
                    f"Fix the eval_script to compute and output ALL required metrics."
                ),
            }

            eval_response = eval_gen_agent.generate(eval_ctx)

            # Parse response
            from run_task import _parse_eval_agent_response
            output_keys, eval_code = _parse_eval_agent_response(eval_response)

            if not eval_code or len(eval_code.strip()) < 20:
                self._log("  [System] Eval script regeneration failed: empty code")
                return False

            # Save new eval_script.py
            eval_path = os.path.join(self.sandbox_dir, "eval_script.py")
            with open(eval_path, "w", encoding="utf-8") as f:
                f.write(eval_code)

            # Update output_keys.json if new one was generated
            if output_keys and "keys" in output_keys:
                ok_path = os.path.join(self.sandbox_dir, "data", "output_keys.json")
                with open(ok_path, "w", encoding="utf-8") as f:
                    json.dump(output_keys, f, indent=2)

            self._log("  [System] eval_script.py regenerated successfully")
            return True

        except Exception as e:
            self._log(f"  [System] Eval script regeneration error: {e}")
            return False

    def _get_primary_score(self, metrics: dict) -> float:
        """Derive a single comparable score from metrics for best-result tracking.

        Uses the first boundary key to determine the primary metric.
        Higher score = better (for lower-is-better metrics, returns negative).
        Falls back to NCC if no boundaries are configured.
        Returns -inf for NaN/Inf values so they never count as "best".
        """
        import math
        LOWER_IS_BETTER = ("nrmse", "error", "mae", "fwhm", "lateral", "axial")
        eval_boundaries = getattr(self, '_eval_boundaries', {})
        if eval_boundaries:
            # Use first boundary key as primary
            for boundary_key in eval_boundaries:
                metric_key = boundary_key.replace("_boundary_deg", "_deg") \
                                         .replace("_boundary_nm", "_nm") \
                                         .replace("_boundary", "")
                val = metrics.get(metric_key)
                if val is not None:
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(val) or math.isinf(val):
                        return float('-inf')
                    is_lower_better = any(kw in boundary_key.lower() for kw in LOWER_IS_BETTER)
                    return -val if is_lower_better else val
        # Fallback: use NCC
        fallback = float(metrics.get("ncc", -1))
        if math.isnan(fallback) or math.isinf(fallback):
            return float('-inf')
        return fallback

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
                "eval_thresholds": self._format_eval_thresholds(),
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
        """Write all round trajectories to data/trajectory/<task_name>/<exp_id>.json"""
        paths_cfg = self.config.get("paths", {})
        traj_root = paths_cfg.get("trajectories_dir", "./data/trajectory")
        traj_dir = os.path.join(traj_root, self.task_name)
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

    # ---- Visualization ----
    def _generate_visualization(self):
        """Generate comparison images at end of task.

        Priority: run task-specific visualize_output.py if it exists,
        fallback to generic GT-vs-output comparison.
        """
        viz_dir = os.path.join(self.sandbox_dir, "visualization")
        os.makedirs(viz_dir, exist_ok=True)

        # Try task-specific visualization script first
        viz_script = os.path.join(self.sandbox_dir, "visualize_output.py")
        if os.path.exists(viz_script):
            self._log("  [Viz] Running task-specific visualize_output.py...")
            ok, stdout, stderr = run_cmd(
                self.python_path, self.sandbox_dir, "visualize_output.py",
                args=["output.npz"], timeout=120,
            )
            if ok:
                self._log(f"  [Viz] Task-specific visualization completed.")
                # Copy to snapshot
                for fname in os.listdir(viz_dir):
                    if fname.endswith(".png"):
                        snapshot_viz = os.path.join(self.snapshot_dir, "visualization.png")
                        shutil.copy2(os.path.join(viz_dir, fname), snapshot_viz)
                        break
                return
            else:
                self._log(f"  [Viz] Task-specific script failed: {stderr[:300]}. Falling back to generic.")

        # Fallback: generic visualization
        self._generic_visualization()

    def _generic_visualization(self):
        """Fallback generic GT vs output comparison images."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self._log("  [Viz] matplotlib not available, skipping visualization.")
            return

        output_path = os.path.join(self.sandbox_dir, "output.npz")
        # GT files are at sandbox root (not data/) for solver isolation
        gt_npz_path = os.path.join(self.sandbox_dir, "ground_truth.npz")
        gt_npy_path = os.path.join(self.sandbox_dir, "gt_output.npy")

        if not os.path.exists(output_path):
            self._log("  [Viz] No output.npz found, skipping.")
            return

        viz_dir = os.path.join(self.sandbox_dir, "visualization")
        os.makedirs(viz_dir, exist_ok=True)

        try:
            # Load prediction — use first key for generic visualization
            pred_npz = np.load(output_path)
            pred_keys = list(pred_npz.keys())
            if not pred_keys:
                self._log("  [Viz] output.npz is empty, skipping.")
                return
            pred_data = pred_npz[pred_keys[0]]
            if pred_data.dtype == object:
                pred_data = np.array(pred_data.tolist(), dtype=float)
            pred_data = np.squeeze(pred_data)

            # Load GT reference
            gt_data = None
            if os.path.exists(gt_npz_path):
                gt_npz = np.load(gt_npz_path)
                # Try to match the same key, or use first key
                gt_key = pred_keys[0] if pred_keys[0] in gt_npz else list(gt_npz.keys())[0]
                gt_data = gt_npz[gt_key]
            elif os.path.exists(gt_npy_path):
                gt_data = np.load(gt_npy_path, allow_pickle=True)
            if gt_data is not None:
                if gt_data.dtype == object:
                    gt_data = np.array(gt_data.tolist(), dtype=float)
                gt_data = np.squeeze(gt_data)

            # Compute metrics for annotation
            ncc_val, nrmse_val = None, None
            if gt_data is not None:
                gt_eval = np.abs(gt_data) if np.iscomplexobj(gt_data) else gt_data.copy()
                pred_eval = np.abs(pred_data) if np.iscomplexobj(pred_data) else pred_data.copy()
                gt_f = gt_eval.ravel().astype(np.float64)
                pred_f = pred_eval.ravel().astype(np.float64)
                min_len = min(len(gt_f), len(pred_f))
                gt_f, pred_f = gt_f[:min_len], pred_f[:min_len]
                pred_c = pred_f - pred_f.mean()
                gt_c = gt_f - gt_f.mean()
                ncc_val = float(np.sum(pred_c * gt_c) / (np.linalg.norm(pred_c) * np.linalg.norm(gt_c) + 1e-10))
                rmse = float(np.sqrt(np.mean((pred_f - gt_f) ** 2)))
                val_range = float(gt_f.max() - gt_f.min() + 1e-30)
                nrmse_val = rmse / val_range

            # Get 2D display slices
            def get_display(arr):
                arr = np.squeeze(arr)
                if np.iscomplexobj(arr):
                    arr = np.abs(arr)
                if arr.ndim <= 1:
                    return arr
                if arr.ndim == 2:
                    return arr
                if arr.ndim == 3:
                    if arr.shape[0] <= 20:
                        return arr[arr.shape[0] // 2]
                    elif arr.shape[-1] <= 20:
                        return arr[:, :, arr.shape[-1] // 2]
                    return arr[arr.shape[0] // 2]
                if arr.ndim == 4:
                    arr = arr[0]
                    return arr[arr.shape[0] // 2] if arr.shape[0] <= 20 else arr[:, :, arr.shape[-1] // 2]
                return arr.reshape(-1)

            pred_disp = get_display(pred_data)
            gt_disp = get_display(gt_data) if gt_data is not None else None

            # Create comparison figure
            if pred_disp.ndim == 1 or (gt_disp is not None and gt_disp.ndim == 1):
                fig, ax = plt.subplots(1, 1, figsize=(12, 5))
                if gt_disp is not None:
                    ax.plot(gt_disp.ravel(), label='Ground Truth', alpha=0.8)
                ax.plot(pred_disp.ravel(), label='Pipeline Output', alpha=0.8)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                n_panels = (1 if gt_disp is not None else 0) + 1 + (1 if gt_disp is not None else 0)
                fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
                if n_panels == 1:
                    axes = [axes]
                idx = 0
                vmin, vmax = None, None
                if gt_disp is not None:
                    all_vals = np.concatenate([gt_disp.ravel(), pred_disp.ravel()])
                    vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
                    im = axes[idx].imshow(gt_disp, cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[idx].set_title('Ground Truth')
                    axes[idx].axis('off')
                    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                    idx += 1

                im = axes[idx].imshow(pred_disp, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[idx].set_title('Pipeline Output')
                axes[idx].axis('off')
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                idx += 1

                if gt_disp is not None:
                    gt_r = gt_disp.astype(np.float64)
                    pred_r = pred_disp.astype(np.float64)
                    min_shape = tuple(min(g, p) for g, p in zip(gt_r.shape, pred_r.shape))
                    gt_r = gt_r[tuple(slice(0, s) for s in min_shape)]
                    pred_r = pred_r[tuple(slice(0, s) for s in min_shape)]
                    diff = np.abs(gt_r - pred_r)
                    im = axes[idx].imshow(diff, cmap='hot')
                    axes[idx].set_title('|Difference|')
                    axes[idx].axis('off')
                    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

            metric_str = ""
            if ncc_val is not None:
                metric_str = f"NCC={ncc_val:.4f}, NRMSE={nrmse_val:.4f}"
            if self.best_iteration > 0:
                metric_str += f" (best iter {self.best_iteration})"
            fig.suptitle(f'{self.task_name}\n{metric_str}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(viz_dir, f'{self.task_name}_comparison.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self._log(f"  [Viz] Saved to {save_path}")

            # Also copy to snapshot dir
            snapshot_viz = os.path.join(self.snapshot_dir, "visualization.png")
            shutil.copy2(save_path, snapshot_viz)

        except Exception as e:
            self._log(f"  [Viz] Visualization failed: {e}")

    # ---- Success/Failure handlers ----
    def _on_success(self, metrics, logs):
        self._save_snapshot(self.retry_count + 1, "final_success", {"metrics": metrics})
        self._generate_visualization()
        self._write_trajectories("success")
        self._post_task_skills_analysis("success")
        self.generate_knowledge_report(success=True)

    def _on_failure(self):
        # Finalize last round if still open
        if self._current_round:
            self._finalize_round(success=False)
        # Restore best result if available
        best_path = os.path.join(self.sandbox_dir, "best_output.npz")
        output_path = os.path.join(self.sandbox_dir, "output.npz")
        if os.path.exists(best_path):
            shutil.copy2(best_path, output_path)
            metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                     for k, v in (self.best_metrics or {}).items())
            self._log(f"  Restored best result from iter {self.best_iteration} "
                      f"({metrics_str})")
        self._generate_visualization()
        self._write_trajectories("failure")
        self._post_task_skills_analysis("failure")
        self.failure_history.clear()
        self.generate_knowledge_report(success=False)
