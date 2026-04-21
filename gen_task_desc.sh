#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 3: Generate task description
# Input:  user prompt (required) + paper/code/readme (optional)
# Output: task_description markdown file
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-config/llm.yaml}"
MODEL_NAME="${MODEL_NAME:-example/default-model}"

USER_PROMPT="${USER_PROMPT:-Generate a detailed task_description for the scientific coding pipeline.}"
TASK_NAME="${TASK_NAME:-example_task}"

PAPER_MD_PATH="${PAPER_MD_PATH:-./data/paper_markdown/example.md}"
CLEANED_CODE_PATH="${CLEANED_CODE_PATH:-}"
GT_CODE_PATH="${GT_CODE_PATH:-}"
README_PATH="${README_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./data/task_descriptions/${TASK_NAME}_description.md}"
TASK_DESC_PATH="${TASK_DESC_PATH:-}"

CMD=("${PYTHON_BIN}" -m gen_task_desc
  --user-prompt "${USER_PROMPT}"
  --task-name "${TASK_NAME}"
  --output-path "${OUTPUT_PATH}"
  --llm-config "${LLM_CONFIG_PATH}"
  --model "${MODEL_NAME}"
)

[[ -n "${PAPER_MD_PATH}" ]] && CMD+=(--paper-markdown-path "${PAPER_MD_PATH}")
[[ -n "${CLEANED_CODE_PATH}" ]] && CMD+=(--cleaned-code-path "${CLEANED_CODE_PATH}")
[[ -n "${GT_CODE_PATH}" ]] && CMD+=(--gt-code-path "${GT_CODE_PATH}")
[[ -n "${README_PATH}" ]] && CMD+=(--readme-path "${README_PATH}")
[[ -n "${TASK_DESC_PATH}" ]] && CMD+=(--task-description-path "${TASK_DESC_PATH}")

echo "=== gen_task_desc.sh ==="
echo "User prompt : ${USER_PROMPT}"
echo "Task name   : ${TASK_NAME}"
echo "Output      : ${OUTPUT_PATH}"
echo "========================"

"${CMD[@]}"
