#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 4: Run the main agentic pipeline
# Input:  Task config YAML with task descriptions + sandbox dirs
# Output: Generated solver code + evaluation results + reports
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-config/llm.yaml}"
MODEL_NAME="${MODEL_NAME:-Vendor2/Claude-4.6-opus}"
TASK_CONFIG="${TASK_CONFIG:-config/tasks/auto_tasks.yaml}"
TASK_FILTER="${TASK_FILTER:-}"
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-}"

CMD=("${PYTHON_BIN}" -m run_task
  --task-config "${TASK_CONFIG}"
  --llm-config "${LLM_CONFIG_PATH}"
  --model "${MODEL_NAME}"
)

[[ -n "${TASK_FILTER}" ]]     && CMD+=(--task-filter "${TASK_FILTER}")
[[ -n "${CONFIG_OVERRIDE}" ]] && CMD+=(--config "${CONFIG_OVERRIDE}")

echo "=== run.sh ==="
echo "Task config : ${TASK_CONFIG}"
echo "Model       : ${MODEL_NAME}"
echo "Task filter : ${TASK_FILTER:-all}"
echo "==============="

"${CMD[@]}"




python -m run_task --task-config "config/tasks/nonstandard_tasks.yaml" --llm-config "config/llm.yaml" --model "gemini-3.1-pro-preview" --task-filter "seismic_FWI_original"
