#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS_DIR="${TASKS_DIR:-data/tasks}"
ENVS_DIR="${ENVS_DIR:-./test_conda_envs}"
LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-config/llm.yaml}"
MODEL_NAME="${MODEL_NAME:-example/default-model}"
OUTPUT_YAML="${OUTPUT_YAML:-config/tasks/generated_tasks.yaml}"
TASK_FILTER="${TASK_FILTER:-}"

CMD=(
  "${PYTHON_BIN}" -m code_cleaner env-setup
  --tasks-dir "${TASKS_DIR}"
  --envs-dir "${ENVS_DIR}"
  --llm-config "${LLM_CONFIG_PATH}"
  --model "${MODEL_NAME}"
  --output-yaml "${OUTPUT_YAML}"
)

[[ -n "${TASK_FILTER}" ]] && CMD+=(--task-filter "${TASK_FILTER}")

echo "=== create_env.sh ==="
echo "Tasks dir   : ${TASKS_DIR}"
echo "Envs dir    : ${ENVS_DIR}"
echo "Model       : ${MODEL_NAME}"
echo "Output YAML : ${OUTPUT_YAML}"
echo "Task filter : ${TASK_FILTER:-all}"
echo "====================="

"${CMD[@]}"
