#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 1: Convert PDF papers to markdown
# Input:  PDF directory
# Output: Markdown directory
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-./data/paper_pdf}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/paper_markdown}"

echo "=== pdf2md.sh ==="
echo "Input  : ${INPUT_DIR}"
echo "Output : ${OUTPUT_DIR}"
echo "=================="

"${PYTHON_BIN}" -m utils.pdf_parser \
  --input-path "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}"
