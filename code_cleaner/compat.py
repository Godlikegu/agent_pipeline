from __future__ import annotations

from pathlib import Path
from typing import List


CUPY_COMPAT_SHIM = """from __future__ import annotations

import numpy as _np


fft = _np.fft
ndarray = _np.ndarray


def array(value, dtype=None):
    return _np.array(value, dtype=dtype)


def asarray(value, dtype=None):
    return _np.asarray(value, dtype=dtype)


def asnumpy(value):
    return _np.asarray(value)


def clear_memo():
    return None


def maximum(a, b, dtype=None):
    result = _np.maximum(a, b)
    return result.astype(dtype) if dtype is not None else result


def minimum(a, b, dtype=None):
    result = _np.minimum(a, b)
    return result.astype(dtype) if dtype is not None else result


def __getattr__(name):
    return getattr(_np, name)
"""


def ensure_repo_compat_shims(repo_root: Path) -> List[Path]:
    shim_paths: List[Path] = []
    if _repo_imports_module(repo_root, "cupy"):
        shim_path = repo_root / "cupy.py"
        if not shim_path.exists():
            shim_path.write_text(CUPY_COMPAT_SHIM, encoding="utf-8")
            shim_paths.append(shim_path)
    return shim_paths


def _repo_imports_module(repo_root: Path, module_name: str) -> bool:
    needle = f"import {module_name}"
    from_needle = f"from {module_name} import"
    for path in repo_root.rglob("*.py"):
        if ".venv" in path.parts or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        if needle in source or from_needle in source:
            return True
    return False
