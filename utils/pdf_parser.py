"""Offline PDF-to-Markdown conversion with PaddleOCR."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


class PaddleOCRMarkdownParser:
    """Convert a PDF file or a directory of PDFs into markdown files."""

    def __init__(self):
        try:
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required for PDF to markdown conversion."
            ) from exc

        self.pipeline = PPStructureV3()

    def convert_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        force: bool = False,
    ) -> Path:
        pdf_file = Path(pdf_path).expanduser().resolve()
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF path not found: {pdf_file}")
        if pdf_file.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {pdf_file}")

        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = target_dir / f"{pdf_file.stem}.md"
        if markdown_path.exists() and not force:
            return markdown_path

        output = self.pipeline.predict(input=str(pdf_file))
        markdown_list = []
        markdown_images = []

        for page in output:
            markdown_info = page.markdown
            markdown_list.append(markdown_info)
            markdown_images.append(markdown_info.get("markdown_images", {}))

        markdown_text = self.pipeline.concatenate_markdown_pages(markdown_list)
        markdown_path.write_text(markdown_text, encoding="utf-8")

        for image_bundle in markdown_images:
            if not image_bundle:
                continue
            for relative_path, image in image_bundle.items():
                image_path = target_dir / relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path)

        return markdown_path

    def convert_path(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        recursive: bool = True,
        force: bool = False,
    ) -> List[Path]:
        source = Path(input_path).expanduser().resolve()
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        if source.is_file():
            return [self.convert_pdf(source, output_root, force=force)]

        if not source.is_dir():
            raise FileNotFoundError(f"Input path not found: {source}")

        pdf_iter = source.rglob("*.pdf") if recursive else source.glob("*.pdf")
        pdf_files = sorted(path for path in pdf_iter if path.is_file())
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found under: {source}")

        markdown_paths: List[Path] = []
        for pdf_file in pdf_files:
            relative_parent = pdf_file.parent.relative_to(source)
            target_dir = output_root / relative_parent
            markdown_paths.append(
                self.convert_pdf(pdf_file, target_dir, force=force)
            )
        return markdown_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PDF files to markdown with PaddleOCR")
    parser.add_argument("--input-path", required=True, help="A PDF file or a directory containing PDF files.")
    parser.add_argument("--output-dir", required=True, help="Directory where markdown files will be written.")
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top level when input-path is a directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate markdown even if the target file already exists.",
    )
    args = parser.parse_args()

    parser_instance = PaddleOCRMarkdownParser()
    markdown_paths = parser_instance.convert_path(
        input_path=args.input_path,
        output_dir=args.output_dir,
        recursive=not args.non_recursive,
        force=args.force,
    )

    for markdown_path in markdown_paths:
        print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
