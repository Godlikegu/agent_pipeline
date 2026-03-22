"""
core/sandbox.py — lightweight sandbox command runner.

All sandbox setup, test harness generation, and reset logic
has been moved to code_cleaner.test_harness.
This module only provides run_cmd for the pipeline execution loop.
"""
import subprocess
from typing import List, Tuple


def run_cmd(
    python_path: str,
    sandbox_dir: str,
    script_name: str,
    args: List[str] | None = None,
    timeout: int = 600,
    check_syntax_only: bool = False,
    syntax_check_timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Execute a Python script inside the sandbox."""
    if check_syntax_only:
        cmd = [python_path, "-m", "py_compile", script_name]
        timeout = syntax_check_timeout
    else:
        cmd = [python_path, script_name] + (args or [])
    try:
        result = subprocess.run(cmd, cwd=sandbox_dir, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT EXPIRED"
