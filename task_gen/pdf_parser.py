"""PDF to Markdown conversion helpers for paper-driven task generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Optional


@dataclass
class OCRScriptConfig:
    """Configuration for converting PDFs into Markdown."""

    backend: str = "external_script"
    python_path: Optional[str] = None
    script_path: Optional[str] = None
    output_dir: Optional[str] = None
    timeout: int = 1800
    result_prefix: str = "RESULT_PATH:"


class PaperMarkdownParser:
    """Converts a PDF paper into Markdown using a configurable OCR backend."""

    def __init__(self, config: Optional[OCRScriptConfig] = None):
        self.config = config or OCRScriptConfig()

    def convert(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None,
        force: bool = False,
    ) -> Path:
        pdf_file = Path(pdf_path).expanduser().resolve()
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF path not found: {pdf_file}")
        if pdf_file.suffix.lower() == ".md":
            return pdf_file

        markdown_dir = self._resolve_output_dir(pdf_file, output_dir)
        markdown_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_dir / f"{pdf_file.stem}.md"
        if markdown_path.exists() and not force:
            return markdown_path

        if self.config.backend == "external_script":
            return self._convert_with_external_script(pdf_file, markdown_dir)
        if self.config.backend == "direct_paddleocr":
            return self._convert_with_paddleocr(pdf_file, markdown_dir)
        raise ValueError(f"Unsupported OCR backend: {self.config.backend}")

    def _resolve_output_dir(
        self,
        pdf_file: Path,
        output_dir: Optional[str | Path],
    ) -> Path:
        if output_dir is not None:
            return Path(output_dir).expanduser().resolve()
        if self.config.output_dir:
            return Path(self.config.output_dir).expanduser().resolve()
        return pdf_file.parent / "paper_markdown"

    def _convert_with_external_script(self, pdf_file: Path, output_dir: Path) -> Path:
        if not self.config.script_path:
            raise ValueError(
                "OCR script path is required when backend='external_script'."
            )

        python_path = self.config.python_path or sys.executable
        script_path = Path(self.config.script_path).expanduser().resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"OCR script not found: {script_path}")

        result = subprocess.run(
            [
                python_path,
                str(script_path),
                "--pdf",
                str(pdf_file),
                "--output_dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=self.config.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "PDF to Markdown conversion failed.\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        markdown_path = self._extract_result_path(result.stdout, output_dir)
        if not markdown_path.exists():
            raise FileNotFoundError(
                f"OCR script reported Markdown path, but file was not created: {markdown_path}"
            )
        return markdown_path

    def _extract_result_path(self, stdout: str, output_dir: Path) -> Path:
        for raw_line in reversed(stdout.splitlines()):
            line = raw_line.strip()
            if line.startswith(self.config.result_prefix):
                raw_path = line[len(self.config.result_prefix) :].strip()
                path = Path(raw_path)
                if not path.is_absolute():
                    path = output_dir / path
                return path.resolve()
        raise RuntimeError(
            "OCR script finished without emitting a RESULT_PATH line in stdout."
        )

    def _convert_with_paddleocr(self, pdf_file: Path, output_dir: Path) -> Path:
        try:
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required for backend='direct_paddleocr'."
            ) from exc

        pipeline = PPStructureV3()
        output = pipeline.predict(input=str(pdf_file))

        markdown_list = []
        markdown_images = []
        for res in output:
            markdown_info = res.markdown
            markdown_list.append(markdown_info)
            markdown_images.append(markdown_info.get("markdown_images", {}))

        markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
        markdown_path = output_dir / f"{pdf_file.stem}.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_text, encoding="utf-8")

        for image_bundle in markdown_images:
            if not image_bundle:
                continue
            for relative_path, image in image_bundle.items():
                image_path = output_dir / relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path)

        return markdown_path.resolve()
