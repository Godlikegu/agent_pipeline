#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-config/llm.yaml}"
TASK_CONFIG="${TASK_CONFIG:-config/tasks/generated_tasks.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-plan_test}"
MODEL_FILTER="${MODEL_FILTER:-}"
TASK_FILTER="${TASK_FILTER:-}"

CMD=(
  "${PYTHON_BIN}" -m tests.test_plan_architect
  --llm-config "${LLM_CONFIG_PATH}"
  --task-config "${TASK_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --temperature 0.5
)

[[ -n "${MODEL_FILTER}" ]] && CMD+=(--model-filter "${MODEL_FILTER}")
[[ -n "${TASK_FILTER}" ]] && CMD+=(--task-filter "${TASK_FILTER}")

"${CMD[@]}"
