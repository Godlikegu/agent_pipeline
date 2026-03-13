import re
import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class SkillTeacher:
    def __init__(self, client: OpenAI, model_name: str, temperature: float = 0.2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    def analyze_trajectory_layered(self, trajectory: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Analyze a task trajectory and extract layered knowledge (Instances, Experiences).
        NO Core Knowledge extraction here (reserved for offline abstraction).

        Instance extraction is rule-based (direct artifact extraction).
        Experience extraction is LLM-based (pattern distillation).
        """
        results = {
            "instances": [],
            "experiences": []
        }

        steps = trajectory.get('steps', [])
        outcome = trajectory.get('outcome')
        task_desc = trajectory.get('task_desc', '')
        task_name = trajectory.get('task_name', 'Unknown Task')

        # 1. Extract Instances (Rule-based, per Agent)
        # Only if outcome is success (Quality Gate)
        if outcome == 'success':
            # A. Planner Instance — extract final plan
            final_plan = trajectory.get('final_plan')
            if final_plan:
                results['instances'].append({
                    "name": f"Plan for {task_name}",
                    "content": final_plan,
                    "agent_scope": "Planner",
                    "artifact_type": "plan",
                    "description": task_desc
                })

            # B. Architect Instance (Skeleton)
            final_skeleton = trajectory.get('final_skeleton')
            if final_skeleton:
                results['instances'].append({
                    "name": f"Architecture for {task_name}",
                    "content": final_skeleton,
                    "agent_scope": "Architect",
                    "artifact_type": "skeleton",
                    "description": task_desc
                })
            else:
                for step in reversed(steps):
                    step_output = step.get('output', {})
                    if step.get('role') == 'Architect' and isinstance(step_output, dict) and step_output.get('skeleton'):
                        results['instances'].append({
                            "name": f"Architecture for {task_name}",
                            "content": step_output['skeleton'],
                            "agent_scope": "Architect",
                            "artifact_type": "skeleton",
                            "description": task_desc
                        })
                        break

            # C. Coder Instance (Final Code)
            final_code = trajectory.get('final_code')
            if final_code:
                results['instances'].append({
                    "name": f"Solution Code for {task_name}",
                    "content": final_code,
                    "agent_scope": "Coder",
                    "artifact_type": "code",
                    "description": task_desc
                })

            # D. Judge Instance (Feedback Samples)
            judge_count = 0
            for step in reversed(steps):
                step_output = step.get('output', {})
                if step.get('role') == 'Judge' and isinstance(step_output, dict) and step_output.get('full_judgement_analysis'):
                    results['instances'].append({
                        "name": f"Evaluation Logic for {task_name} (Iter {step.get('iteration')})",
                        "content": step_output,
                        "agent_scope": "Judge",
                        "artifact_type": "feedback",
                        "description": task_desc
                    })
                    judge_count += 1
                    if judge_count >= 2:
                        break

        # 2. Extract Experiences (LLM-based)
        components = self._extract_trajectory_components(trajectory)

        roles_to_analyze = ['Planner', 'Architect', 'Coder', 'Judge']
        all_experiences = []

        for role in roles_to_analyze:
            prompt = self._construct_role_specific_prompt(role, components, trajectory)
            if not prompt:
                continue

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": f"You are an expert AI coding tutor specializing in {role} strategies. Analyze the trajectory context to distill reusable Experience Patterns. Output ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )

                content = response.choices[0].message.content
                json_match = re.search(r'(\{[\s\S]*\})', content)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    role_exps = parsed.get('experiences', [])

                    # Enforce agent_scope
                    for exp in role_exps:
                        exp['agent_scope'] = role

                    all_experiences.extend(role_exps)
                else:
                    logger.warning(f"Failed to parse JSON from teacher response for role {role}")

            except Exception as e:
                logger.error(f"Error in layered analysis for role {role}: {e}")

        results['experiences'] = all_experiences

        return results

    def _extract_trajectory_components(self, trajectory: Dict[str, Any]) -> Dict[str, str]:
        """
        Decompose trajectory into structured text blocks for each role.
        Returns a dict with keys: 'planner', 'architect', 'coder', 'judge', 'execution'
        """
        steps = trajectory.get('steps', [])

        components = {
            'planner': "",
            'architect': "",
            'coder': "",
            'judge': "",
            'execution': "",
            'full_history': ""
        }

        if not steps:
            return components

        for step in steps:
            role = step.get('role', 'Unknown')
            iter_id = step.get('iteration', '?')
            output = step.get('output', {})
            if not isinstance(output, dict): output = {}

            entry = f"[Iter {iter_id}] {role}:"

            if role == "Planner":
                plan = output.get('plan', '')
                detail = f"\n  Plan: {plan}"
                entry += detail
                components['planner'] += f"[Iter {iter_id}] {detail}\n"

            elif role == "Architect":
                skel = output.get('skeleton', '')
                thought = output.get('thought', output.get('reasoning', ''))
                detail = ""
                if thought: detail += f"\n  Thought: {thought}"
                detail += f"\n  Skeleton: {str(skel)}"
                entry += detail
                components['architect'] += f"[Iter {iter_id}] {detail}\n"

            elif role == "Coder":
                task_type = output.get('task_type', '')
                target = output.get('target', 'unknown')
                error = output.get('error', '')
                if task_type == 'full_rewrite_syntax':
                    detail = f"\n  Action: Syntax Fix for {str(error)[:200]}..."
                else:
                    detail = f"\n  Action: Implement/Edit {target} ({task_type})"
                entry += detail
                components['coder'] += f"[Iter {iter_id}] {detail}\n"

            elif role == "Execution":
                success = output.get('success', False)
                if success:
                    detail = f"\n  Result: SUCCESS. Metrics: {output.get('metrics')}"
                else:
                    err = output.get('stderr', '')
                    detail = f"\n  Result: FAILED. Error: {str(err)[:500]}..."
                entry += detail
                components['execution'] += f"[Iter {iter_id}] {detail}\n"
                components['coder'] += f"  -> Execution: {detail.strip()}\n"
                components['judge'] += f"[Iter {iter_id}] Execution Observation: {detail.strip()}\n"

            elif role == "Judge":
                analysis = output.get('full_judgement_analysis', output.get('analysis', ''))
                ticket = output.get('ticket')
                detail = f"\n  Analysis: {analysis}\n  Ticket: {ticket}"
                entry += detail
                components['judge'] += f"[Iter {iter_id}] {detail}\n"

            components['full_history'] += entry + "\n"

        return components

    def _construct_role_specific_prompt(self, role: str, components: Dict[str, str], trajectory: Dict[str, Any]) -> str:
        """
        Construct a context-aware prompt for a specific agent role.
        """
        task_desc = trajectory.get('task_desc')
        outcome = trajectory.get('outcome')

        base_prompt = f"""
Analyze the trajectory to extract **{role} Experiences**.
Task: "{task_desc}"
Outcome: {outcome}

"""

        if role == "Planner":
            context = f"""
=== CONTEXT: JUDGE FEEDBACK (Why strategy might need change) ===
{components['judge']}

=== FOCUS: PLANNER HISTORY (Strategy Evolution) ===
{components['planner']}
"""
            instructions = """
TASK: Extract **Planner Experiences** (Strategy & Parameter Tuning).
IMPORTANT: Group related insights into COARSE-GRAINED patterns. Do not list every minor adjustment as a separate experience.
Limit to Top 1-3 most critical strategic insights.

Focus on:
1. Initial strategy selection (Why was this plan chosen for this task?).
2. Strategy pivots (Did the plan change? Why? What was the trigger?).
3. Parameter tuning (e.g., "Adjusting ADMM rho", "Changing wavelet levels").

Structure:
- Condition: The scenario (e.g., "When initial ADMM plan produced artifacts").
- Action: The strategic decision (e.g., "Switch to PnP-ADMM with denoiser").
- Rationale: Why this strategy is better for this condition.
"""
            if outcome != 'success':
                 instructions = """
TASK: Extract **Planner Failure Lessons** (Strategy Flaws).
IMPORTANT: Group related failures into COARSE-GRAINED patterns.
Limit to Top 1-3 most critical failure lessons.

Focus on:
1. Strategy mismatch: Why did the chosen plan fail to solve the task?
2. Parameter sensitivity: Did poor parameter choices lead to divergence or stagnation?
3. Incorrect assumptions: What did the planner assume that turned out to be false?

Structure:
- Condition: The scenario or task requirement.
- Action: The failed strategy or decision.
- Rationale: Why this strategy failed (Root Cause Analysis).
"""

        elif role == "Architect":
            context = f"""
=== CONTEXT: PLANNER OUTPUT (The Requirement) ===
{components['planner']}

=== CONTEXT: IMPLEMENTATION ISSUES (Feedback) ===
{components['coder']}

=== FOCUS: ARCHITECT HISTORY (Design Decisions) ===
{components['architect']}
"""
            instructions = """
TASK: Extract **Architect Experiences** (Interface & Modular Design).
IMPORTANT: Group related insights into COARSE-GRAINED patterns.
Limit to Top 1-3 most critical design insights.

Focus on:
1. Plan-to-Skeleton conversion (How to structure complex math into functions?).
2. Interface design (What arguments/returns were critical?).
3. Helper decomposition (Did extracting a helper function solve a complexity issue?).
4. Handling specific data flows (e.g., "How to handle 3D stacks vs 2D images").

Structure:
- Condition: The design requirement or complexity trigger.
- Action: The architectural pattern or interface choice.
- Rationale: Why this design improves maintainability/correctness.
"""
            if outcome != 'success':
                instructions = """
TASK: Extract **Architect Failure Lessons** (Design Flaws).
IMPORTANT: Group related failures into COARSE-GRAINED patterns.
Limit to Top 1-3 most critical design flaws.

Focus on:
1. Complexity overload: Did the design become too complex to implement or debug?
2. Missing interfaces: Did the skeleton fail to expose necessary parameters to the Coder?
3. Inflexibility: Did the design prevent necessary pivots?

Structure:
- Condition: The design requirement.
- Action: The flawed design choice.
- Rationale: Why this design hindered success.
"""

        elif role == "Coder":
            context = f"""
=== CONTEXT: ARCHITECT SPEC (The Blueprint) ===
{components['architect']}

=== FOCUS: CODER & DEBUGGING TRACE (Implementation Loop) ===
{components['coder']}
"""
            instructions = """
TASK: Extract **Coder Experiences** (Implementation & Debugging).
IMPORTANT: Group related insights into COARSE-GRAINED patterns. Do not list every minor syntax fix.
Combine related fixes (e.g., multiple import fixes) into a single experience.
Limit to Top 1-3 most critical implementation insights.

Focus on:
1. **Error-Fix Patterns**: Look for "FAILED -> Action -> SUCCESS" sequences.
2. API usage nuances (e.g., "PyTorch dimension requirements", "FFT shifting").
3. Numerical stability fixes (e.g., "Adding epsilon", "Clipping values").
4. Performance optimizations (if any).

Structure:
- Condition: The specific error or implementation hurdle (cite the error msg if relevant).
- Action: The code fix or implementation technique.
- Rationale: Why this fixes the error (technical explanation).
"""
            if outcome != 'success':
                instructions = """
TASK: Extract **Coder Failure Lessons** (Implementation Pitfalls).
IMPORTANT: Group related failures into COARSE-GRAINED patterns.
Combine repeated identical errors into ONE experience.
Limit to Top 1-3 most critical failure lessons.

Focus on:
1. **Anti-Patterns**: Repeated mistakes or loops (e.g., "Trying to import a missing library 5 times").
2. Timeout causes: What code structure led to timeouts (e.g., recursive file search)?
3. Debugging failures: Why did the fix attempts fail to resolve the error?

Structure:
- Condition: The error or hurdle.
- Action: The failed fix or problematic pattern.
- Rationale: Why this approach failed and what should be done instead (Avoidance Strategy).
"""

        elif role == "Judge":
            context = f"""
=== CONTEXT: EXECUTION OBSERVATIONS (Metrics & Errors) ===
{components['execution']}

=== FOCUS: JUDGE HISTORY (Diagnosis & Critique) ===
{components['judge']}
"""
            instructions = """
TASK: Extract **Judge Experiences** (Diagnosis & Evaluation).
IMPORTANT: Group related insights into COARSE-GRAINED patterns.
Limit to Top 1-3 most critical diagnostic insights.

Focus on:
1. Metric interpretation (e.g., "High PSNR but low SSIM implies...").
2. Failure diagnosis (How to distinguish code bug vs. parameter issue?).
3. Critique quality (What feedback led to a successful fix?).

Structure:
- Condition: The observed symptom (Metric pattern or visual artifact description).
- Action: The diagnosis or critique strategy.
- Rationale: The logic connecting symptom to root cause.
"""
            if outcome != 'success':
                instructions = """
TASK: Extract **Judge Failure Lessons** (Diagnosis Blind Spots).
IMPORTANT: Group related failures into COARSE-GRAINED patterns.
Limit to Top 1-3 most critical diagnostic failures.

Focus on:
1. Misdiagnosis: Did the Judge misidentify the root cause?
2. Vague feedback: Was the feedback too generic to guide the Coder?
3. Loop detection: Did the Judge fail to stop a repetitive failure loop?

Structure:
- Condition: The observed symptom.
- Action: The ineffective diagnosis or critique.
- Rationale: Why this diagnosis was insufficient or incorrect.
"""
        else:
            return ""

        # Common Output Format
        format_instr = f"""
OUTPUT JSON FORMAT:
{{
  "experiences": [
    {{
      "name": "str (Concise title)",
      "content": {{
        "condition": "str",
        "action": "str",
        "rationale": "str"
      }},
      "agent_scope": "{role}",
      "tags": ["str", "str"]
    }}
  ]
}}
"""
        return base_prompt + context + instructions + format_instr

    def analyze_trajectory(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Backward compatibility wrapper
        results = self.analyze_trajectory_layered(trajectory)
        return []

    def _summarize_steps(self, steps: List[Dict]) -> str:
        # Backward compatibility wrapper
        return self._extract_trajectory_components({'steps': steps})['full_history']

    def _construct_prompt(self, trajectory: Dict[str, Any]) -> str:
        steps_summary = self._summarize_steps(trajectory.get('steps', []))

        return f"""
Analyze the following agent trajectory for the task: "{trajectory.get('task_desc')}"

Outcome: {trajectory.get('outcome')}
Final Reward/Metrics: {trajectory.get('final_reward')}

TRAJECTORY HISTORY (Step-by-Step):
{steps_summary}

========================================
TASK:
You are an expert in Scientific Coding and Inverse Problems.
Extract MULTIPLE reusable skills from this trajectory.
Do NOT limit yourself to just one skill.
The goal is to distill high-quality, task-agnostic wisdom that can be applied to future inverse problems.

STRUCTURE YOUR ANALYSIS AROUND THESE 4 DIMENSIONS (Extract skills for each if relevant):

1. **Data Preprocessing & Loading**:
   - How was data format discovery handled? (e.g., handling unknown .npy structures, missing metadata)
   - What normalization or padding strategies were critical?
   - Any specific handling of complex/real-world data artifacts?

2. **Forward Modelling**:
   - How was the physics model implemented efficiently? (e.g., vectorization, FFT usage, padding for convolution)
   - Were there tricks for boundary conditions or discrete approximations?

3. **Inverse Processing (The Core Algorithm)**:
   - What optimization strategy worked? (e.g., ADMM, Gradient Descent, PnP, Unrolling)
   - Specific tricks for parameter tuning, step-size selection, or initialization?
   - How were gradients or adjoints verified?

4. **Post-processing & Evaluation**:
   - How were results visualized or metrics calculated?
   - Any specific cropping, de-padding, or thresholding steps required for valid evaluation?

CRITICAL INSTRUCTIONS FOR GENERALIZATION:
1. **Abstract away specific task details**: Do NOT use specific dataset names (e.g., "DiffuserCam", "SimDataset"), specific file paths, or specific dimensions (e.g., "64x64") in the skill name or applicability.
2. **Focus on the Pattern**: Identify the underlying pattern.
   - *Bad*: "Fixing the 64x64 array mismatch in DiffuserCam"
   - *Good*: "Handling Shape Mismatches in Convolutional Forward Models"
3. **Granularity**: Mix high-level workflow skills with low-level technical tips.

OUTPUT JSON FORMAT (Strictly enforce this):
{{
  "skills": [
    {{
      "skill_name": "str (Short, task-agnostic title. e.g., 'Robust Data Loading for Blind Formats')",
      "category": "str (Choose one: 'Data Preprocessing', 'Forward Modelling', 'Inverse Processing', 'Post-processing', 'General Debugging')",
      "applicability_criteria": "str (When to apply this skill)",
      "description": "str (Summary of the principle/strategy)",
      "generalizable_steps": ["str", ... (Ordered steps) ],
      "failure_analysis": {{  // Optional, include if this skill relates to fixing an error
        "root_cause": "str",
        "avoidance_strategy": "str"
      }}
    }},
    ...
  ]
}}
"""
