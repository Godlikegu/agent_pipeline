# Agent Pipeline: Autonomous Solver for Computational Inverse Problems

A multi-agent LLM pipeline that autonomously solves scientific inverse problems in computational imaging, MRI reconstruction, seismic inversion, and more. Given only a **task description** and **input data**, the system plans, designs, implements, executes, and iteratively debugs a Python solver — with no human intervention.

## Key Results

- **51 tasks** spanning CT reconstruction, MRI, ptychography, EHT black hole imaging, seismic FWI, optical diffraction tomography, and more
- **Multi-iteration self-healing**: Planner -> Critic -> Architect -> Coder -> Execution -> Judge loop with automatic bug diagnosis and targeted fix routing
- **Per-task isolated environments**: Each task gets its own conda env with the exact dependencies it needs
- **Skill system**: Optional cross-task knowledge transfer — successful implementations are distilled into reusable skills

---

## Architecture

```
                          Task Description (README.md)
                          + Input Data (dataset/)
                                    |
                                    v
                    +-------------------------------+
                    |         run_task.py            |
                    |  (entry point, sandbox setup)  |
                    +-------------------------------+
                                    |
                    +-------------------------------+
                    |      PipelineWorkflow          |
                    |      (core/workflow.py)         |
                    +-------------------------------+
                                    |
            +-----------+-----------+-----------+-----------+
            |           |           |           |           |
            v           v           v           v           v
        Planner     Critic     Architect     Coder       Judge
       (math plan)  (review)   (skeleton)   (implement)  (diagnose)
            |                       |           |           |
            +--- feedback loop -----+-----------+-----------+
                                    |
                                    v
                            Sandbox Execution
                            (subprocess: solver.py)
                                    |
                                    v
                            Evaluation (NCC/NRMSE)
                                    |
                            Pass? ---> Done + Visualization
                            Fail? ---> Judge routes fix ticket
                                       back to Planner/Architect/Coder
```

### Agent Roles

| Agent | File | Role |
|-------|------|------|
| **Planner** | `agents/planner.py` | Formulates the mathematical model, forward operator, and step-by-step algorithm |
| **Critic** | `agents/planner.py` | Reviews and challenges the Planner's proposal before implementation |
| **Architect** | `agents/architect.py` | Designs the `InverseSolver` class skeleton (interfaces, method signatures) |
| **Coder** | `agents/coder.py` | Implements each function, merges into the full solver via AST-based code editing |
| **Judge** | `agents/judge.py` | Diagnoses failures using a 4-step protocol (syntax -> interface -> fidelity -> algorithm) and routes fix tickets |

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `core/workflow.py` | Main iteration loop, best-result tracking, visualization |
| `core/workflow_base.py` | Base class: state management, structural validation, skills injection, trajectory recording |
| `core/sandbox.py` | Subprocess execution of solver/eval scripts in isolated sandbox |
| `utils/code_editor.py` | AST-based code merger (replace function/imports/main block without breaking surrounding code) |
| `utils/text_utils.py` | Extract JSON/Python from LLM output, format failure histories |
| `utils/reporter.py` | Aggregate multi-task results into execution reports |
| `skills/` | Optional skill system: retrieval, learning, distillation from trajectories |
| `code_cleaner/` | Environment setup: auto-create per-task conda envs from `requirements.txt` |

---

## Project Structure

```
agent_pipeline/
  run_task.py                  # Entry point
  run.sh                       # Batch run script
  create_env.sh                # Environment setup shortcut

  config/
    default.yaml               # Pipeline params, eval thresholds, skills config
    llm.yaml                   # LLM endpoints and API keys
    tasks/
      auto_tasks.yaml          # Task list (name, python_path, task_dir)

  core/
    workflow.py                # Main pipeline loop (Planner->Judge iteration)
    workflow_base.py           # Base class: agents, state, structural validation
    sandbox.py                 # Subprocess runner

  agents/
    base.py                    # BaseAgent: LLM call with auto-continuation
    planner.py                 # PlannerAgent + CriticAgent
    architect.py               # ArchitectAgent
    coder.py                   # CoderAgent with AST merge
    judge.py                   # JudgeAgent with 4-step diagnostic
    sandbox_agents.py          # DataGenAgent, EvalGenAgent
    skills_generator.py        # Post-task skill distillation
    code_diff_analyzer.py      # Reference vs generated code comparison

  utils/
    code_editor.py             # AST-based code merge (replace_function, replace_imports, etc.)
    config_loader.py           # YAML config loading with defaults
    llm_client.py              # OpenAI-compatible client factory
    text_utils.py              # JSON/Python extraction from LLM output
    reporter.py                # Multi-task execution report generator

  skills/
    __init__.py                # Factory: create_skill_manager()
    file_manager.py            # FileSkillManager: retrieval + learning
    file_store.py              # Skill storage with embedding search
    ablation.py                # NoSkillManager (disabled mode)

  code_cleaner/
    environment.py             # CondaEnvManager: auto-create per-task envs
    cli.py                     # CLI for env-setup command

  data/
    tasks/                     # 51 task directories
      <task_name>/
        README.md              # Task description
        requirements.txt       # Python dependencies
        main.py                # Reference implementation (not exposed to agents)
        data/                  # Input data (raw_data.npz, meta_data.json, ground_truth.npz)
        evaluation/
          metrics.json         # Per-task NCC/NRMSE thresholds
          eval_script.py       # Evaluation script

  test_conda_envs/             # Auto-generated per-task conda environments
    task_<name>/python.exe
```

---

## Quick Start

### 1. Install Base Dependencies

```bash
conda create -n agent python=3.10 -y
conda activate agent
pip install openai pyyaml numpy
```

### 2. Configure LLM

Edit `config/llm.yaml`:

```yaml
models:
  "Vendor2/Claude-4.6-opus":
    api_type: "openai"
    base_url: "https://your-api-gateway/v1"
    api_key: "YOUR_API_KEY"
```

### 3. Set Up Task Environments

Each task requires its own Python environment with the correct dependencies:

```bash
# Auto-create all task environments from requirements.txt
python -m code_cleaner env-setup \
  --tasks-dir data/tasks \
  --envs-dir ./test_conda_envs \
  --model Vendor2/Claude-4.6-opus \
  --output-yaml config/tasks/auto_tasks.yaml
```

Or use the shortcut:
```bash
bash create_env.sh
```

### 4. Run

```bash
# Run all tasks
bash run.sh

# Run specific tasks
TASK_FILTER=hessian_sim,lucky_imaging bash run.sh

# Or directly
python -m run_task \
  --task-config config/tasks/auto_tasks.yaml \
  --model Vendor2/Claude-4.6-opus \
  --task-filter SSNP_ODT
```

---

## Configuration

### Pipeline Parameters (`config/default.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline.max_retries` | 5 | Maximum Planner->Judge iterations per task |
| `pipeline.execution_timeout` | 1800 | Solver execution timeout (seconds) |
| `pipeline.syntax_check_timeout` | 30 | Syntax check timeout (seconds) |
| `pipeline.code_size_guard` | 12000 | Max generated code length (chars) |
| `evaluation.min_ncc` | 0.85 | Global NCC pass threshold |
| `evaluation.max_nrmse` | 0.5 | Global NRMSE pass threshold |

Per-task thresholds override globals via `data/tasks/<name>/evaluation/metrics.json`.

### Skills System

```yaml
skills:
  retrieval_enabled: false    # Enable skill retrieval (read-only)
  learning_enabled: false     # Enable skill learning from trajectories
```

When enabled, the system:
1. **Retrieves** relevant skills from past tasks and injects them into Planner/Coder prompts
2. **Distills** new skills from successful/failed trajectories after each task
3. **Promotes** validated skills from draft to permanent status

### Task Configuration (`config/tasks/auto_tasks.yaml`)

```yaml
tasks:
  - name: SSNP_ODT
    python_path: ./test_conda_envs/task_SSNP_ODT/python.exe
    task_description_path: ./data/tasks/SSNP_ODT/README.md
    task_dir: ./data/tasks/SSNP_ODT
```

---

## Pipeline Details

### Iteration Flow

1. **Planner** reads the task description and proposes a mathematical algorithm
2. **Critic** reviews the plan (up to 3 rounds of revision)
3. **Architect** designs a `class InverseSolver` skeleton with method signatures
4. **Coder** implements each function, using AST-based merge to preserve existing code
5. **Structural Validator** checks for nested classes, orphaned methods, duplicate definitions
6. **Syntax Check** with auto-fix (up to 5 attempts)
7. **Execution** in sandbox subprocess with timeout
8. **Evaluation** via `eval_script.py` computing NCC and NRMSE against ground truth
9. **Judge** diagnoses failure and routes a fix ticket:
   - Syntax/structural errors -> **Coder**
   - Interface mismatches -> **Architect**
   - Implementation deviates from plan -> **Coder** (with specific fix target)
   - Algorithm fundamentally wrong -> **Planner** (full replanning)

### Best Result Tracking

The pipeline tracks the best NCC across all iterations. If the final iteration regresses, the best result is automatically restored. Passed iterations are always prioritized over non-passed ones.

### Visualization

At the end of each task (success or failure), the pipeline generates a comparison image:
- Ground Truth | Pipeline Output | |Difference|
- Annotated with NCC, NRMSE, and best iteration number
- Saved to `<sandbox_dir>/visualization/` and the snapshot directory

### Structural Validation

After each code merge, an AST-based validator detects and auto-fixes:
- **Nested class definitions** (class inside class) -> flattened
- **Orphaned methods** (`self` parameter at module scope) -> moved into class
- **Duplicate class definitions** -> merged

These are common failure modes when LLMs generate code segments that get merged into existing files.

---

## Task Structure

Each task under `data/tasks/<name>/` follows this structure:

```
<task_name>/
  README.md              # Task description (provided to agents)
  requirements.txt       # Python dependencies for this task
  main.py                # Reference implementation (NOT exposed to agents during solving)
  src/                   # Additional reference source files
  data/
    raw_data.npz         # Input measurements/observations
    meta_data.json       # Physical parameters (wavelength, pixel size, etc.)
    ground_truth.npz     # Ground truth output for evaluation
  evaluation/
    metrics.json         # {"ncc_boundary": 0.85, "nrmse_boundary": 0.5}
    eval_script.py       # Evaluation script (auto-generated if missing)
```

**Information boundary**: Agents only see `README.md`, `dataset/` contents (data + metadata), and available packages. The reference implementation (`main.py`, `src/`) is never exposed during the solving process. It is only used post-task for optional skill distillation when `learning_enabled=true`.

---

## Outputs

Each task run produces:

```
<sandbox_root>/<model_name>/<task_name>_<timestamp>/
  workflow.log                    # Full agent decision log
  iter_001_plan.md                # Planner output per iteration
  iter_001_skeleton.py            # Architect output
  iter_001_solver.py              # Final solver code
  iter_001_exec_log.txt           # Execution stdout/stderr
  iter_001_judge.json             # Judge diagnosis
  visualization.png               # GT vs output comparison
  final_success.json / ...        # Final snapshot with metrics

<sandbox_dir>/
  solver.py                       # Generated solver
  output.npy                      # Pipeline output (best result)
  best_output.npy                 # Best intermediate result
  visualization/                  # Comparison images
  dataset/                        # Input data (read-only)
```

Aggregate reports are saved to `reports/`.

---

## Covered Task Domains

The 51 tasks span diverse computational inverse problems:

| Domain | Tasks |
|--------|-------|
| **CT Reconstruction** | ct_fan_beam, ct_sparse_view, ct_poisson_lowdose, ct_dual_energy, xray_tooth_gridrec, xray_laminography_tike |
| **MRI** | mri_sense, mri_grappa, mri_tv, mri_l1_wavelet, mri_pnp_admm, mri_varnet, mri_noncartesian_cs, mri_dynamic_dce, mri_t2_mapping, pnp_mri_reconstruction, diffusion_mri_dti |
| **Optical / Microscopy** | SSNP_ODT, reflection_ODT, fourier_ptychography, conventional_ptychography, electron_ptychography, fpm_inr_reconstruction, hessian_sim, microscope_denoising, lensless_imaging, light_field_microscope, single_molecule_light_field |
| **Astronomy** | eht_black_hole_dynamic, eht_black_hole_original, eht_black_hole_tomography, eht_black_hole_feature_extraction_dynamic, eht_black_hole_UQ, exoplanet_imaging, lucky_imaging, shapelet_source_reconstruction |
| **Spectral / Hyperspectral** | spectral_snapshot_compressive_imaging, mcr_hyperspectral, raman_cell_phenotyping |
| **Seismic / Acoustic** | seismic_FWI_original, usct_FWI, ultrasound_sos_tomography, photoacoustic_tomography, plane_wave_ultrasound |
| **Other** | differentiable_deflectometry, confocal-nlos-fk, eit_conductivity_reconstruction, insar_phase_unwrapping, weather_radar_data_assimilation, pet_mlem, xray_ptychography_tike |
