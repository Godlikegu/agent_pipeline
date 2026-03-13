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
        return """You are a QA Engineer for Computer Vision/Signal Processing.
Your Goal: Create an evaluation script `eval_script.py`.
Output Format: Return ONLY the Python code.

Requirements:
1. Accept one command-line argument: path to the prediction file (.npy).
2. Load `dataset/gt_output.npy` as the reference. Use `np.load(path, allow_pickle=True)` to avoid security errors with complex data types.
3. Calculate PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity).
   - PREFERRED: Use numpy-based implementations for maximum compatibility:
     ```
     def compute_psnr(pred, gt):
         mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
         if mse == 0: return 100.0
         max_val = max(np.max(np.abs(gt)), 1e-10)
         return float(10 * np.log10(max_val ** 2 / mse))
     ```
   - Only use `skimage.metrics` if you are SURE it is installed. If not, use the numpy fallback.
   - For SSIM, a simple correlation-based approximation is acceptable:
     ```
     def compute_ssim(pred, gt):
         pred_f = pred.flatten().astype(np.float64)
         gt_f = gt.flatten().astype(np.float64)
         if np.std(pred_f) < 1e-10 or np.std(gt_f) < 1e-10: return 0.0
         return float(np.corrcoef(pred_f, gt_f)[0, 1])
     ```
4. **Metric Guardrails**:
   - Handle minor shape mismatches gracefully (e.g., squeeze singleton dimensions).
   - If shapes are fundamentally incompatible, print JSON with shape info.
5. Print the result strictly in JSON format to stdout: {"psnr": <float>, "ssim": <float>}
   - Print NOTHING ELSE to stdout. No debug messages, no warnings.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        return f"""
        Task: {context['task_desc']}

        Data Shape Hint: {context.get('data_shape_hint', 'N/A')}

        Please generate `eval_script.py` that calculates PSNR and SSIM between `dataset/gt_output.npy` and the provided prediction file.
        """

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
