from typing import Any, Dict
from .base import BaseAgent
import subprocess

class DataGenAgent(BaseAgent):
    def _build_system_prompt(self) -> str:
        return """You are an expert Data Engineer for scientific computing.
                Your Goal: Create a robust Python script `data_gen.py` to prepare datasets for an Inverse Problem.
                Output Format: Return ONLY the Python code block. No markdown, no explanation.
                Constraints:
                1. The code must be self-contained or import provided GT code correctly.
                2. It MUST generate three files in `dataset/`:
                - `input.npy` (The measurement/degraded data)
                14. - `gt_output.npy` (The ground truth)
                15. - `baseline.npy` (A simple heuristic result, e.g., identity or simple filter)
                16. 3. **Data Complexity**:
                    - AVOID trivial data (e.g., all zeros, all ones, or simple constants).
                    - Use `numpy.random` to generate complex, non-trivial signals (e.g., mixtures of sinusoids, Gaussian random fields, or realistic synthetic data relevant to the domain).
                    - Ensure the data has sufficient variance and structure to challenge the solver.
                17. 4. Ensure random seeds are fixed for reproducibility.
                18. 5. **Strict Type Safety**: All saved .npy files MUST be standard numeric arrays (e.g., `.astype(np.float32)`). Do NOT save Python object arrays (lists of lists, ragged arrays) which trigger pickle security errors.
                19. """

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        prompt = f"""
        Task Description: {context['task_desc']}

        Reference Ground Truth Code Context:
        {context['gt_code_snippet']}

        Available Packages: {context.get('package_list', 'Standard Python Libraries')[:500]}

        Please write `data_gen.py`.
        IMPORTANT:
        - Use ONLY the packages listed above. Check API signatures carefully.
        - Generate synthetic data using numpy/scipy. Do NOT depend on external files.
        - All saved arrays must be standard numeric types (np.float32 or np.float64).
        """
        if context.get('feedback'):
            prompt += f"\n\nPrevious attempt failed: {context['feedback']}"
        return prompt

class EvalGenAgent(BaseAgent):
    def _build_system_prompt(self) -> str:
        return """You are a QA Engineer for Scientific Computing evaluation.
Your Goal: Create an evaluation script `eval_script.py` that computes NCC and NRMSE.
Output Format: Return ONLY the Python code.

Requirements:
1. Accept one command-line argument: path to the prediction file (.npy).
2. Load `dataset/gt_output.npy` as the reference. Use `np.load(path, allow_pickle=True)`.
3. Calculate NCC (Normalized Cross-Correlation) and NRMSE (Normalized RMSE).

   For COMPLEX data (e.g., ptychography reconstruction where dtype is complex64/complex128):
   - Extract phase and remove global mean:
     gt_phase = np.angle(gt) - np.angle(gt).mean()
     pred_phase = np.angle(pred) - np.angle(pred).mean()
   - Use the phase arrays for NCC and NRMSE computation.

   For REAL data: use the arrays directly.

   NCC computation:
     pred_c = pred_arr - pred_arr.mean()
     gt_c = gt_arr - gt_arr.mean()
     ncc = np.sum(pred_c * gt_c) / (np.linalg.norm(pred_c) * np.linalg.norm(gt_c) + 1e-10)

   NRMSE computation (normalized by L2 norm of ground truth):
     nrmse = np.linalg.norm(pred_arr - gt_arr) / (np.linalg.norm(gt_arr) + 1e-10)

4. Handle minor shape mismatches (squeeze singleton dimensions).
5. Print result strictly as JSON to stdout: {"ncc": <float>, "nrmse": <float>}
   - Print NOTHING ELSE to stdout. No debug messages, no warnings.
6. Load meta_data.json from dataset/ if available, to determine data type and processing needs.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        prompt = f"""
Task: {context['task_desc']}

Data Shape Hint: {context.get('data_shape_hint', 'N/A')}
"""
        if context.get('meta_data'):
            import json as _json
            prompt += f"\nMeta Data (physical parameters, data format info):\n{_json.dumps(context['meta_data'], indent=2, default=str)}\n"

        # Include evaluation thresholds
        eval_thresholds = context.get('eval_thresholds', {})
        if eval_thresholds:
            prompt += f"\nEvaluation Thresholds (from metrics.json):\n"
            prompt += f"  - NCC boundary: {eval_thresholds.get('min_ncc', 'N/A')} (NCC >= this value means PASS)\n"
            prompt += f"  - NRMSE boundary: {eval_thresholds.get('max_nrmse', 'N/A')} (NRMSE <= this value means PASS)\n"

        prompt += """
Please generate `eval_script.py` that computes NCC and NRMSE between `dataset/gt_output.npy` and the provided prediction file.
Output JSON: {"ncc": <float>, "nrmse": <float>}
"""
        if context.get('feedback'):
            prompt += f"\n\nPrevious attempt failed: {context['feedback']}"
        return prompt

def get_installed_libraries(python_path: str) -> str:
    """Detects installed libraries in the environment."""
    try:
        result = subprocess.run(
            [python_path, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            import json
            packages = json.loads(result.stdout)
            return ", ".join([p["name"] for p in packages])
    except Exception:
        pass
    return "Standard Python Libraries"
