"""Convert local PDFs into Markdown using the PaddleOCR parser."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    pdf_dir = Path(
        os.environ.get("PDF_INPUT_DIR", repo_root / "data" / "paper_pdf")
    ).expanduser().resolve()
    out_dir = Path(
        os.environ.get("MARKDOWN_OUTPUT_DIR", repo_root / "data" / "paper_markdown")
    ).expanduser().resolve()

    if not pdf_dir.exists():
        print(f"[ERROR] PDF directory not found: {pdf_dir}")
        return 2

    try:
        from utils.pdf_parser import PaddleOCRMarkdownParser

        parser = PaddleOCRMarkdownParser()
        md_paths = parser.convert_path(
            input_path=pdf_dir,
            output_dir=out_dir,
            recursive=True,
            force=True,
        )
    except ImportError as e:
        print("[ERROR] paddleocr is not installed, so the PaddleOCR backend is unavailable.")
        print(f"        Import error: {e}")
        print("\nYou can install it in the current environment with:")
        print("  pip install paddleocr")
        return 3

    print(f"[OK] Generated {len(md_paths)} Markdown file(s):")
    for p in md_paths:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
