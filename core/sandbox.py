import os
import shutil
import subprocess
from typing import List, Callable


def setup_sandbox(sandbox_dir: str, gt_code_path: str, log_fn: Callable):
    log_fn(f"\n[System] Initializing Sandbox at {sandbox_dir}...")
    os.makedirs(sandbox_dir, exist_ok=True)

    dest_gt = os.path.join(sandbox_dir, "gt_code")
    if os.path.exists(dest_gt):
        if os.path.isdir(dest_gt): shutil.rmtree(dest_gt)
        else: os.remove(dest_gt)

    if os.path.isdir(gt_code_path):
        shutil.copytree(gt_code_path, dest_gt)
    elif os.path.isfile(gt_code_path):
        os.makedirs(dest_gt, exist_ok=True)
        shutil.copy(gt_code_path, dest_gt)
    else:
        raise FileNotFoundError(f"GT code path not found: {gt_code_path}")

    os.makedirs(os.path.join(sandbox_dir, "dataset"), exist_ok=True)


def run_cmd(python_path: str, sandbox_dir: str, script_name: str, args: List[str] = [], timeout=600, check_syntax_only=False, syntax_check_timeout=30):
    if check_syntax_only:
        cmd = [python_path, "-m", "py_compile", script_name]
        timeout = syntax_check_timeout
    else:
        cmd = [python_path, script_name] + args

    try:
        result = subprocess.run(
            cmd,
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT EXPIRED"


def reset_sandbox_to_phase0_state(sandbox_dir: str, log_fn: Callable):
    preserve_paths = {"dirs": ["dataset", "gt_code"], "files": ["eval_script.py", "data_gen.py"]}
    log_fn(">>> [System] Resetting sandbox to Phase 0 state...")
    for item in os.listdir(sandbox_dir):
        item_path = os.path.join(sandbox_dir, item)
        if (os.path.isdir(item_path) and item in preserve_paths["dirs"]) or \
        (os.path.isfile(item_path) and item in preserve_paths["files"]): continue
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path): os.remove(item_path)
            elif os.path.isdir(item_path): shutil.rmtree(item_path)
        except Exception as e: log_fn(f"  [Warning] Failed to remove {item}: {str(e)}")
    log_fn("  ✅ Sandbox reset complete.")
