#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 3: Generate task description
# Input:  user prompt (required) + paper/code/readme (optional)
# Output: task_description markdown file
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-/home/guyuxuan/.conda/envs/agent/bin/python}"
LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-/home/guyuxuan/pipeline/config/llm.yaml}"
MODEL_NAME="${MODEL_NAME:-cds/Claude-4.6-opus}"

USER_PROMPT="${USER_PROMPT:-Generate a detailed task_description for the scientific coding pipeline.}"
TASK_NAME="${TASK_NAME:-sim}"

PAPER_MD_PATH="${PAPER_MD_PATH:-/data/guyuxuan/agent/paper_md/sim.md}"
CLEANED_CODE_PATH="${CLEANED_CODE_PATH:-/home/guyuxuan/pipeline/artifacts/code_cleaner/sparse-deconv-py_20260322_194931/code_cleaned.py}"
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

[[ -n "${PAPER_MD_PATH}" ]]     && CMD+=(--paper-markdown-path "${PAPER_MD_PATH}")
[[ -n "${CLEANED_CODE_PATH}" ]] && CMD+=(--cleaned-code-path "${CLEANED_CODE_PATH}")
[[ -n "${GT_CODE_PATH}" ]]      && CMD+=(--gt-code-path "${GT_CODE_PATH}")
[[ -n "${README_PATH}" ]]       && CMD+=(--readme-path "${README_PATH}")
[[ -n "${TASK_DESC_PATH}" ]]    && CMD+=(--task-description-path "${TASK_DESC_PATH}")

echo "=== gen_task_desc.sh ==="
echo "User prompt : ${USER_PROMPT}"
echo "Task name   : ${TASK_NAME}"
echo "Output      : ${OUTPUT_PATH}"
echo "========================"

"${CMD[@]}"
