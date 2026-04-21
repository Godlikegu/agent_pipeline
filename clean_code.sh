#!/usr/bin/env bash
set -euo pipefail

# Deprecated compatibility wrapper.
# Public workflow now uses `create_env.sh` and `run.sh`.

PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS_DIR="${TASKS_DIR:-data/tasks}"
ENVS_DIR="${ENVS_DIR:-./test_conda_envs}"
LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-config/llm.yaml}"
MODEL_NAME="${MODEL_NAME:-example/default-model}"
OUTPUT_YAML="${OUTPUT_YAML:-config/tasks/generated_tasks.yaml}"

echo "clean_code.sh is a compatibility wrapper for environment setup."
echo "For the public workflow, prefer running: ./create_env.sh"

"${PYTHON_BIN}" -m code_cleaner env-setup \
  --tasks-dir "${TASKS_DIR}" \
  --envs-dir "${ENVS_DIR}" \
  --llm-config "${LLM_CONFIG_PATH}" \
  --model "${MODEL_NAME}" \
  --output-yaml "${OUTPUT_YAML}"
