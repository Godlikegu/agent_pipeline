#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 2: Clean code repository + set up sandbox environment
# Input:  GitHub URL or local repo path
# Output: Cleaned code + sandbox with test data + eval scripts
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-/home/guyuxuan/.conda/envs/agent/bin/python}"
LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-/home/guyuxuan/pipeline/config/llm.yaml}"
MODEL_NAME="${MODEL_NAME:-cds/Claude-4.6-opus}"

REPO_URL="${REPO_URL:-https://github.com/WeisongZhao/sparse-deconv-py}"
LOCAL_REPO="${LOCAL_REPO:-}"
PAPER_MD="${PAPER_MD:-}"
TASK_FAMILY="${TASK_FAMILY:-general}"
SANDBOX_ROOT="${SANDBOX_ROOT:-/data/guyuxuan/agent/end_sandbox}"
LLM_ENABLED="${LLM_ENABLED:-true}"
LLM_REQUIRED="${LLM_REQUIRED:-true}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"

if [[ -z "${REPO_URL}" && -z "${LOCAL_REPO}" ]]; then
  echo "Error: Set either REPO_URL or LOCAL_REPO"
  exit 1
fi

CMD=("${PYTHON_BIN}" -m code_cleaner.cli
  --task-family "${TASK_FAMILY}"
  --sandbox-root "${SANDBOX_ROOT}"
  --llm-enabled "${LLM_ENABLED}"
  --llm-required "${LLM_REQUIRED}"
  --force-rebuild-env "${FORCE_REBUILD}"
  --llm-config "${LLM_CONFIG_PATH}"
  --model "${MODEL_NAME}"
)

[[ -n "${REPO_URL}" ]]   && CMD+=(--github-url "${REPO_URL}")
[[ -n "${LOCAL_REPO}" ]] && CMD+=(--local-repo "${LOCAL_REPO}")
[[ -n "${PAPER_MD}" ]]   && CMD+=(--paper-md "${PAPER_MD}")

echo "=== clean_code.sh ==="
echo "Repo URL    : ${REPO_URL:-N/A}"
echo "Local repo  : ${LOCAL_REPO:-N/A}"
echo "Task family : ${TASK_FAMILY}"
echo "Sandbox root: ${SANDBOX_ROOT}"
echo "======================"

"${CMD[@]}"
