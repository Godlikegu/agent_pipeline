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
                2. It MUST generate three files in `data/`:
                - `input.npy` (The measurement/degraded data)
                - `gt_output.npy` (The ground truth)
                - `baseline.npy` (A simple heuristic result, e.g., identity or simple filter)
                3. **Data Complexity**:
                    - AVOID trivial data (e.g., all zeros, all ones, or simple constants).
                    - Use `numpy.random` to generate complex, non-trivial signals (e.g., mixtures of sinusoids, Gaussian random fields, or realistic synthetic data relevant to the domain).
                    - Ensure the data has sufficient variance and structure to challenge the solver.
                4. Ensure random seeds are fixed for reproducibility.
                5. **Strict Type Safety**: All saved .npy files MUST be standard numeric arrays (e.g., `.astype(np.float32)`). Do NOT save Python object arrays (lists of lists, ragged arrays) which trigger pickle security errors.
                """

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
Your Goal: TWO tasks:
  (A) Determine what output keys the solver should produce in `output.npz`.
  (B) Create an evaluation script `eval_script.py` that computes task-specific metrics.

## OUTPUT FORMAT (you MUST follow this exact structure):

FIRST, output a JSON block specifying the required solver output keys:
```json:output_keys
{
  "keys": {
    "key_name": {"shape": [dim1, dim2, ...], "dtype": "float64",
                 "description": "what this array represents",
                 "eval_against": "ground_truth.npz:key_name_in_gt"}
  },
  "save_instruction": "np.savez('output.npz', key_name=arr, ...)"
}
```

THEN, output the eval_script.py code:
```python
# eval_script.py code here
```

## Rules for determining output keys:
- Analyze the metric boundaries: each metric comparing solver output against GT implies an output key.
- The output key shape/dtype MUST match the GT key it will be compared against.
- If the task EXTRACTS FEATURES (e.g., position angles from images), the output should be the
  extracted features, NOT the raw reconstruction. The solver should do both reconstruction AND
  feature extraction, saving the features as output keys.
- If the task reconstructs a single array (e.g., velocity field, phase image), use ONE key
  matching the GT key name (e.g., "v_true", "object").
- For multi-target tasks (e.g., separate fibers/cysts regions), use separate keys per target.
- For parameter estimation tasks (e.g., lens parameters), use separate keys per parameter.
- For tasks with no GT file, determine output format from metric definitions and README context.
- The eval_script must load prediction from the .npz file using the SAME key names.

## Requirements for eval_script.py:
1. Accept one command-line argument: path to the prediction file (.npz).
2. Load prediction: `pred = np.load(pred_path)`, then access keys as `pred['key_name']`.
   For backward compatibility, if the file is .npy, load as: `arr = np.load(pred_path, allow_pickle=True)`
   and treat as single-key output.
3. Load ground truth reference data:
   - First check for `ground_truth.npz` in the WORKING DIRECTORY (NOT in data/).
     Then check `baseline_reference.npz` in the working directory.
   - IMPORTANT: GT files are at the script's working directory level, NOT inside data/.
     Use `np.load('ground_truth.npz')`, NOT `np.load('data/ground_truth.npz')`.
   - Key structure and shapes are provided in the context (gt_structure).
4. Compute ALL metrics required by the task. The required output keys and boundary definitions
   are listed in the context. Your output JSON MUST contain ALL required metric keys.

   Default NCC computation (unless custom definition is provided):
     pred_c = pred_arr - pred_arr.mean()
     gt_c = gt_arr - gt_arr.mean()
     ncc = np.sum(pred_c * gt_c) / (np.linalg.norm(pred_c) * np.linalg.norm(gt_c) + 1e-10)

   Default NRMSE computation (unless custom definition is provided):
     nrmse = float(np.sqrt(np.mean((pred_arr - gt_arr)**2)) / (gt_arr.max() - gt_arr.min() + 1e-30))

   For COMPLEX data (e.g., ptychography where dtype is complex64/complex128):
   - Extract phase and remove global mean:
     gt_phase = np.angle(gt) - np.angle(gt).mean()
     pred_phase = np.angle(pred) - np.angle(pred).mean()
   - Use the phase arrays for NCC and NRMSE computation.

5. If custom metric definitions (ncc_definition, nrmse_definition) are provided in the context,
   follow them EXACTLY instead of the defaults above.
6. If custom metrics are required (e.g., SNR, CNR, FWHM, sharpness, lateral/axial error,
   position_angle_mae, velocity_nrmse), implement them based on the notebook/src reference code
   provided in the context. Follow the reference implementation EXACTLY — do NOT add extra
   transformations (e.g., circular angle wrapping, log scaling, normalization) unless they appear
   in the reference evaluation code. The reference implementation is the ground truth for how
   metrics should be computed.
7. Handle minor shape mismatches (squeeze singleton dimensions).
   **DTYPE NORMALIZATION**: The solver may save images as float32 in non-[0,1] ranges
   (e.g., uint16 range 0-65535 saved as float). When computing metrics that involve
   pixel-value operations (sharpness via Laplacian, etc.), you MUST normalize images
   to [0,1] before the metric computation. For ANY float image, check its max value:
     if img.max() > 1.0:  img = img / img.max()  # or / 65535.0 if in uint16 range
   This prevents overflow when reference code does `(gray * 255).astype(np.uint8)`.
   The reference code's `_to_gray_f64()` only normalizes uint8 and uint16 dtypes;
   when replicating it for float inputs, add the missing normalization step.
8. Print result strictly as JSON to stdout with ALL required metric keys.
   - Print NOTHING ELSE to stdout. No debug messages, no warnings.
9. Load meta_data.json from data/ if available, to determine data type and processing needs.
10. Only data files under data/ and metrics.json (copied from evaluation/) are available.
    Do NOT reference any other evaluation files. The eval_script must be fully self-contained.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        prompt = f"""
Task: {context['task_desc']}

Data Shape Hint: {context.get('data_shape_hint', 'N/A')}
"""
        if context.get('meta_data'):
            import json as _json
            prompt += f"\nMeta Data (physical parameters, data format info):\n{_json.dumps(context['meta_data'], indent=2, default=str)}\n"

        # Full metrics.json content
        if context.get('metrics_json'):
            import json as _json
            prompt += f"\n### Full metrics.json (ALL metric definitions and baselines):\n```json\n{_json.dumps(context['metrics_json'], indent=2, default=str)}\n```\n"

        # GT structure (all keys, shapes, dtypes)
        if context.get('gt_structure'):
            import json as _json
            prompt += f"\n### Ground Truth Structure (gt_keys.json):\n```json\n{_json.dumps(context['gt_structure'], indent=2)}\n```\n"

        # Boundary definitions and required output keys with explicit mapping
        if context.get('boundary_definitions'):
            import json as _json
            prompt += f"\n### Evaluation Boundaries (your output MUST allow checking these):\n"
            for bk, bv in context['boundary_definitions'].items():
                prompt += f"  - {bk}: {bv}\n"

        if context.get('boundary_to_metric_map'):
            import json as _json
            prompt += f"\n### CRITICAL: Boundary-to-Metric Key Mapping\n"
            prompt += "Your eval_script.py JSON output MUST use EXACTLY these metric key names:\n"
            for bk, mk in context['boundary_to_metric_map'].items():
                prompt += f"  - boundary '{bk}' → output metric key: \"{mk}\"\n"
            prompt += "The pipeline checks thresholds using these exact key names. Do NOT use different names.\n"
        elif context.get('required_output_keys'):
            keys_str = ", ".join(f'"{k}"' for k in context['required_output_keys'])
            prompt += f"\n### REQUIRED OUTPUT KEYS (your JSON output MUST contain ALL of these):\n{{{keys_str}}}\n"

        # Notebook evaluation code (reference implementation)
        if context.get('notebook_eval_code'):
            prompt += f"\n### Reference Evaluation Code (from task notebook — use as implementation guide):\n```python\n{context['notebook_eval_code']}\n```\n"

        # Src metric functions (reference implementation)
        if context.get('src_metric_functions'):
            prompt += f"\n### Reference Metric Functions (from task src/):\n```python\n{context['src_metric_functions']}\n```\n"

        # Legacy eval thresholds
        eval_thresholds = context.get('eval_thresholds', {})
        if eval_thresholds and not context.get('boundary_definitions'):
            prompt += f"\nEvaluation Thresholds:\n"
            prompt += f"  - NCC boundary: {eval_thresholds.get('min_ncc', 'N/A')} (NCC >= this value means PASS)\n"
            prompt += f"  - NRMSE boundary: {eval_thresholds.get('max_nrmse', 'N/A')} (NRMSE <= this value means PASS)\n"

        prompt += """
Please generate BOTH:
1. The ```json:output_keys``` block specifying what the solver should save to output.npz
2. The ```python``` eval_script.py code that evaluates output.npz against ground truth

CRITICAL: Your eval_script must load prediction from .npz using the SAME key names you specified.
CRITICAL: Your output JSON MUST contain ALL required metric keys listed above.
If any custom metric definitions (ncc_definition, nrmse_definition) are provided, follow them EXACTLY.
"""
        if context.get('feedback'):
            prompt += f"\n\nPrevious attempt failed: {context['feedback']}"
        return prompt


class VizGenAgent(BaseAgent):
    def _build_system_prompt(self) -> str:
        return """You are a Scientific Visualization Engineer.
Your Goal: Create a Python script `visualize_output.py` that generates COMPREHENSIVE comparison
figures between the pipeline output and ground truth/baseline reference.

Output Format: Return ONLY the Python code.

CRITICAL: You MUST generate MULTIPLE figures (at least 3-5) for a thorough analysis.
A single plot is NEVER sufficient. Your visualization should be as rich as a research notebook.

Requirements:
1. Accept one command-line argument: path to the prediction file (.npz or .npy).
2. Load prediction from the CLI argument path:
   - If .npz: `pred = np.load(path); pred_key = pred['key_name']`
   - If .npy (legacy): `pred_arr = np.load(path, allow_pickle=True)`
3. Load reference data from `ground_truth.npz` or `baseline_reference.npz` in the
   WORKING DIRECTORY (NOT in data/). Use `np.load('ground_truth.npz')`.
   IMPORTANT: GT files are at the script's working directory level, NOT inside data/.
   Key structure is provided in the context.
4. Load `data/output_keys.json` if available to understand which keys map to which GT keys.
5. You MUST generate ALL of the following visualization types that apply to the data:

   **Always generate:**
   - Figure 1: GT vs Output overlay/comparison (the primary comparison plot)
   - Figure 2: Per-element/per-frame error analysis (error bar chart, residual plot, or difference map)
   - Figure 3: Summary metrics table or bar chart (MAE, RMSE, NCC, NRMSE, etc.)

   **For 2D image data, also generate:**
   - Side-by-side panels: GT | Output | Difference heatmap (with colorbars)
   - Per-channel or per-slice comparison if multi-channel/3D
   - Zoomed insets on regions of interest if applicable

   **For 1D/time-series/parameter data, also generate:**
   - Overlaid line plots with GT and prediction (with error bands if available)
   - Per-frame/per-element scatter plot (predicted vs GT, with y=x reference line)
   - Residual/error plot over time or index

   **For multi-key data (multiple output keys), generate:**
   - Separate comparison plots for EACH output key vs its GT counterpart
   - A combined summary figure showing all keys

   **For complex/phase data:**
   - Separate magnitude and phase visualizations

6. **NOTEBOOK-EXACT REPLICATION (HIGHEST PRIORITY)**:
   Your script MUST reproduce ALL visualizations from the reference notebook.
   The notebook is the specification — your output should be visually identical.
   For EACH visualization cell in the notebook reference code:
   - Replicate the EXACT same plot type, layout, subplot arrangement, axes, labels, and colormap.
   - If the notebook calls functions from src/visualization.py, reproduce those function
     implementations INLINE in your script (the solver sandbox has no src/ access).
     Copy the function logic faithfully — do NOT simplify or rewrite from scratch.
   - Adapt data loading paths (use output.npz keys and ground_truth.npz keys) but keep
     the visualization logic, figure sizes, subplot layouts, and annotation styles identical.
   - Preserve domain-specific conventions: colormaps, units, axis labels, title format, etc.
   - If the notebook generates a text table (e.g., per-frame metrics), generate it as both
     a saved text file AND a matplotlib table figure.
   - Generate ONE figure per notebook visualization section, plus any generic figures from
     rule 5 above that the notebook does NOT already cover.
7. Annotate ALL figures with computed metrics (task-specific metrics, NCC, NRMSE, MAE, etc.).
8. Use matplotlib with Agg backend (non-interactive).
9. Save all figures to `visualization/` directory (create if needed).
   Use descriptive filenames (e.g., `parameter_evolution.png`, `error_analysis.png`,
   `gt_vs_pred_comparison.png`, `per_frame_metrics.png`).
10. Handle edge cases: shape mismatches, complex data, scalar data.
11. If no notebook reference code is provided, fall back to generating rich generic
    visualizations using the data type heuristics in rule 5 above.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        prompt = f"""
Task: {context['task_desc']}

Data Shape Hint: {context.get('data_shape_hint', 'N/A')}
"""
        # GT structure
        if context.get('gt_structure'):
            import json as _json
            prompt += f"\n### Ground Truth Structure:\n```json\n{_json.dumps(context['gt_structure'], indent=2)}\n```\n"

        # Output keys (what the solver produces in output.npz)
        if context.get('output_keys'):
            import json as _json
            prompt += f"\n### Solver Output Keys (output.npz):\n```json\n{_json.dumps(context['output_keys'], indent=2)}\n```\n"

        # Notebook visualization code
        if context.get('notebook_viz_code'):
            prompt += f"\n### Reference Visualization Code (from task notebook — YOU MUST REPRODUCE THESE EXACTLY):\n```python\n{context['notebook_viz_code']}\n```\n"

        # Src visualization.py
        if context.get('src_visualization'):
            prompt += f"\n### src/visualization.py (COPY these function implementations inline into your script):\n```python\n{context['src_visualization']}\n```\n"

        prompt += """
Generate `visualize_output.py`. Save ALL figures to `visualization/` directory.
Use `matplotlib.use('Agg')` at the top.

CRITICAL REQUIREMENTS:
- Your script MUST reproduce EVERY visualization from the notebook reference code above.
- For each src/visualization.py function called by the notebook, copy the implementation
  inline into your script and call it with the appropriate data from output.npz and ground_truth.npz.
- The notebook is the SPECIFICATION. Your figures should be visually identical to what the
  notebook produces, with the only difference being data loading (output.npz instead of
  in-memory variables).
- If NO notebook reference is provided, generate at least 3-5 generic comparison figures.
- Each figure should have a descriptive filename and be saved separately.
- Include numerical annotations (metrics, errors) on each figure.
"""
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
