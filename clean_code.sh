#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/guyuxuan/.conda/envs/agent/bin/python"
REPO_URL="https://github.com/WeisongZhao/sparse-deconv-py"
TASK_FAMILY="computational-imaging"
SANDBOX_ROOT="/data/guyuxuan/agent/end_sandbox/test"
LLM_CONFIG_PATH="/home/guyuxuan/pipeline/config/llm.yaml"
MODEL_NAME="cds/Claude-4.6-opus"

"${PYTHON_BIN}" -m code_cleaner.cli \
  --github-url "${REPO_URL}" \
  --task-family "${TASK_FAMILY}" \
  --sandbox-root "${SANDBOX_ROOT}" \
  --llm-enabled true \
  --llm-required true \
  --force-rebuild-env false \
  --llm-config "${LLM_CONFIG_PATH}" \
  --model "${MODEL_NAME}"
