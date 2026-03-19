"""将 /data/guyuxuan/agent/paper_pdf 下的 PDF 转为 Markdown，输出到 /data/guyuxuan/agent/paper_md。"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    pdf_dir = Path("/data/guyuxuan/agent/paper_pdf").expanduser().resolve()
    out_dir = Path("/data/guyuxuan/agent/paper_md").expanduser().resolve()

    if not pdf_dir.exists():
        print(f"[ERROR] PDF 目录不存在: {pdf_dir}")
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
        print("[ERROR] 未安装 paddleocr，无法使用 PaddleOCR 后端。")
        print(f"        详细错误: {e}")
        print("\n你可以尝试在当前环境安装：")
        print("  pip install paddleocr")
        return 3

    print(f"[OK] 共生成 {len(md_paths)} 个 Markdown 文件：")
    for p in md_paths:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
