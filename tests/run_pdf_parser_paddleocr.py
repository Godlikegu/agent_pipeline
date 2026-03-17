from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    pdf_path = Path("/home/yjh/agentic_reproduce/paper_archive/test.pdf").expanduser().resolve()
    if not pdf_path.exists():
        print(f"[ERROR] PDF 不存在: {pdf_path}")
        return 2

    out_dir = (Path(__file__).resolve().parent / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from task_gen.pdf_parser import OCRScriptConfig, PaperMarkdownParser

        parser = PaperMarkdownParser(
            OCRScriptConfig(
                backend="direct_paddleocr",
                output_dir=str(out_dir),
            )
        )
        md_path = parser.convert(pdf_path, force=True)
    except ImportError as e:
        # direct_paddleocr 依赖 paddleocr（以及其运行时依赖）
        print("[ERROR] 未安装 paddleocr，无法使用 direct_paddleocr 后端。")
        print(f"        详细错误: {e}")
        print("\n你可以尝试在当前环境安装：")
        print("  pip install paddleocr")
        return 3

    print(f"[OK] Markdown 已生成: {md_path}")
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        preview = "\n".join(text.splitlines()[:80])
        print("\n===== Markdown 预览（前 80 行）=====")
        print(preview)
        print("===== 预览结束 =====\n")
    except Exception as e:
        print(f"[WARN] 读取 Markdown 预览失败: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

