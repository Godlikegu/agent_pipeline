#!/usr/bin/env bash
# Run prompt optimization for the task generator.
# Usage:
#   ./run_optimization.sh [OPTIONS]
#
# Required: --paper-dir, --ground-truth-dir, --models, --optimizer-model
#
# Example:
#   ./run_optimization.sh \
#     --paper-dir ./data/paper_markdown \
#     --ground-truth-dir ./data/ground_truth_descriptions \
#     --models "cds/Claude-4.6-opus" \
#     --optimizer-model "cds/Claude-4.6-opus"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Default paths (override with env or args)
PAPER_DIR="${PAPER_DIR:-/data/guyuxuan/agent/paper_md}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-/data/guyuxuan/agent/gt_task_desc}"
LLM_CONFIG="${LLM_CONFIG:-$PROJECT_ROOT/config/llm.yaml}"
MODELS="${MODELS:-cds/Claude-4.6-opus}"
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-cds/Claude-4.6-opus}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/optimized_prompts}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-20}"

python -m prompt_optimizer.prompt_optimization \
  --paper-dir "$PAPER_DIR" \
  --ground-truth-dir "$GROUND_TRUTH_DIR" \
  --llm-config "$LLM_CONFIG" \
  --models "$MODELS" \
  --optimizer-model "$OPTIMIZER_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  "$@"
