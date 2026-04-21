# Agent Pipeline

Agent Pipeline is a multi-agent coding workflow for inverse problems. Given a task folder with a problem description, inputs, and evaluation metadata, it plans a solver, writes code, runs it in a sandbox, and iterates until it passes or exhausts retries.

## Public Repo Notes

- `config/llm.yaml` is a template only. Replace the example endpoint and `YOUR_API_KEY` locally.
- `config/tasks/example_tasks.yaml` is a non-runnable sample manifest for documentation.
- `config/tasks/generated_tasks.yaml` is the local manifest produced by environment setup. It is intentionally ignored by git.
- `data/tasks/` is kept as an empty placeholder in the public repo. Put your own task folders there locally.
- The `skills/` system is still experimental and immature. It is kept in the codebase, but it is **not used for evaluation or benchmarking** in the public workflow.

## Repo Layout

```text
agent_pipeline/
  agents/                  # Planner / Architect / Coder / Judge agents
  code_cleaner/            # Task environment setup utilities
  config/
    default.yaml           # Pipeline defaults and relative paths
    llm.yaml               # Public-safe LLM config template
    tasks/
      example_tasks.yaml   # Checked-in example manifest
  core/                    # Workflow loop and sandbox execution
  data/
    tasks/                 # Empty placeholder; populate locally
  tests/                   # Utility scripts and local test helpers
  create_env.sh            # Generate per-task envs and manifest
  run.sh                   # Run the main pipeline
  run_task.py              # Python entry point
```

## Task Folder Contract

Place each task under `data/tasks/<task_name>/` with at least:

```text
data/tasks/<task_name>/
  README.md
  requirements.txt
  data/
  evaluation/
```

Typical tasks also include `evaluation/metrics.json`, optional notebooks, and optional reference code that is not exposed to the solving agents.

## Quick Start

### 1. Install the base environment

```bash
conda create -n agent python=3.10 -y
conda activate agent
pip install openai pyyaml numpy
```

### 2. Fill in `config/llm.yaml`

```yaml
models:
  "example/default-model":
    api_type: "openai"
    base_url: "https://api.example.com/v1"
    api_key: "YOUR_API_KEY"
```

All checked-in YAML files are examples or templates. Do not commit your real credentials.

### 3. Add your tasks locally

Create one folder per task under `data/tasks/`.

### 4. Generate environments and the runnable task manifest

```bash
./create_env.sh
```

This reads tasks from `data/tasks/`, creates per-task environments in `test_conda_envs/`, and writes `config/tasks/generated_tasks.yaml`.

### 5. Run the pipeline

```bash
./run.sh
```

To run a subset of tasks:

```bash
TASK_FILTER=example_inverse_problem ./run.sh
```

Direct Python entrypoint:

```bash
python -m run_task \
  --task-config config/tasks/generated_tasks.yaml \
  --llm-config config/llm.yaml \
  --model example/default-model
```

## What Each Step Produces

### Prepare tasks

- Input location: `data/tasks/<task_name>/`
- You provide: task description, data, evaluation metadata, and task dependencies

### Environment setup

- Command: `./create_env.sh`
- Outputs:
  - `test_conda_envs/task_<task_name>/...`
  - `config/tasks/generated_tasks.yaml`

### Pipeline run

- Command: `./run.sh`
- Outputs:
  - Per-run snapshots under `data/end_sandbox/<model>/<task_name>_<timestamp>/`
  - Task sandbox artifacts such as generated solver code, execution logs, evaluation outputs, and visualizations

### Aggregate report

- Produced during `run.sh`
- Output directory: `reports/`
- Typical file: `reports/execution_report_<timestamp>.json`

## Main Runtime Outputs

Each run creates a timestamped snapshot directory like:

```text
data/end_sandbox/<model>/<task_name>_<timestamp>/
  workflow.log
  iter_001_plan.md
  iter_001_skeleton.py
  iter_001_solver.py
  iter_001_exec_log.txt
  iter_001_judge.json
  iter_001_final_success.json
```

The sandbox working directory also contains generated files such as `solver.py`, `output.npy` or `output.npz`, `eval_script.py`, and visualization artifacts.

## Example Task Manifest

`config/tasks/example_tasks.yaml` is intentionally fake and documentation-only:

```yaml
tasks:
  - name: example_inverse_problem
    python_path: ./test_conda_envs/task_example_inverse_problem/<python-binary>
    task_description_path: ./data/tasks/example_inverse_problem/README.md
    task_dir: ./data/tasks/example_inverse_problem
```

The real local manifest is `config/tasks/generated_tasks.yaml`, which is generated from your machine-specific environment paths and should not be committed.

## Configuration Notes

- `config/default.yaml` contains only repo-relative defaults in the public repo.
- Per-task evaluation thresholds can override global defaults through each task's `evaluation/metrics.json`.
- Skills remain disabled by default, and public evaluation should keep them disabled.

## Public Cleanup Intent

This public version is intentionally minimal:

- no hardcoded machine paths
- no real API keys or private endpoints
- no checked-in local run manifests
- no checked-in benchmark logs or sandbox outputs
