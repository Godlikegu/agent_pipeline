#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 4: Run the main agentic pipeline
# Input:  Generated task config YAML + LLM config
# Output: Generated solver code + evaluation results + reports
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-config/llm.yaml}"
MODEL_NAME="${MODEL_NAME:-example/default-model}"
TASK_CONFIG="${TASK_CONFIG:-config/tasks/generated_tasks.yaml}"
TASK_FILTER="${TASK_FILTER:-}"
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-}"

CMD=("${PYTHON_BIN}" -m run_task
  --task-config "${TASK_CONFIG}"
  --llm-config "${LLM_CONFIG_PATH}"
  --model "${MODEL_NAME}"
)

[[ -n "${TASK_FILTER}" ]] && CMD+=(--task-filter "${TASK_FILTER}")
[[ -n "${CONFIG_OVERRIDE}" ]] && CMD+=(--config "${CONFIG_OVERRIDE}")

echo "=== run.sh ==="
echo "Task config : ${TASK_CONFIG}"
echo "Model       : ${MODEL_NAME}"
echo "Task filter : ${TASK_FILTER:-all}"
echo "==============="

"${CMD[@]}"
