#!/usr/bin/env bash
# Run prompt optimization for the task generator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

PAPER_DIR="${PAPER_DIR:-$PROJECT_ROOT/data/paper_markdown}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-$PROJECT_ROOT/data/ground_truth_descriptions}"
LLM_CONFIG="${LLM_CONFIG:-$PROJECT_ROOT/config/llm.yaml}"
MODELS="${MODELS:-example/default-model}"
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-example/default-model}"
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
