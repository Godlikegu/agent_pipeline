"""
code_cleaner/environment.py -- Conda environment setup with LLM-assisted diagnosis.

Provides:
  - CondaEnvManager: subprocess-based conda/pip operations (--prefix mode)
  - EnvSetupAgent(BaseAgent): LLM-assisted failure diagnosis and fix
"""
import os
import sys
import json
import re
import subprocess
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.base import BaseAgent
from utils.text_utils import extract_json


# ---------------------------------------------------------------------------
# Package-name -> import-name mapping for common mismatches
# ---------------------------------------------------------------------------
KNOWN_IMPORT_MAP = {
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "Pillow": "PIL",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pytorch": "torch",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "python-dateutil": "dateutil",
    "jupyter": None,       # meta-package, skip import check
    "ipykernel": "ipykernel",
    "setuptools": "setuptools",
    "ptylab": "PtyLab",
    "pyzmq": "zmq",
    "msgpack-python": "msgpack",
    "attrs": "attr",
}


def parse_requirements(requirements_path: str) -> List[str]:
    """Read requirements.txt and return list of package specifiers."""
    specs = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            specs.append(line)
    return specs


def extract_package_name(specifier: str) -> str:
    """Extract bare package name from a pip specifier.
    e.g. 'numpy>=1.24' -> 'numpy', 'ptylab==0.2.2' -> 'ptylab'
    """
    return re.split(r"[><=!~\[]", specifier)[0].strip()


def _guess_import_name(package_name: str) -> Optional[str]:
    """Guess the import name for a pip package."""
    lower = package_name.lower()
    if lower in KNOWN_IMPORT_MAP:
        return KNOWN_IMPORT_MAP[lower]
    # Also check original case
    if package_name in KNOWN_IMPORT_MAP:
        return KNOWN_IMPORT_MAP[package_name]
    # Default heuristic: lowercase, replace hyphens with underscores
    return lower.replace("-", "_")


def _log(msg: str):
    print(f"[EnvSetup] {msg}")


# ---------------------------------------------------------------------------
# CondaEnvManager — pure subprocess wrapper, no LLM dependency
# ---------------------------------------------------------------------------
class CondaEnvManager:
    """Manages conda environments using --prefix mode in a user-specified directory."""

    def __init__(self, conda_path: Optional[str] = None, envs_dir: str = "./test_conda_envs",
                 python_version: str = "3.10"):
        self.conda_path = conda_path or self._detect_conda()
        self.envs_dir = os.path.abspath(envs_dir)
        self.python_version = python_version
        os.makedirs(self.envs_dir, exist_ok=True)

    @staticmethod
    def _detect_conda() -> str:
        """Auto-detect conda executable."""
        # Try 'conda' on PATH first
        try:
            result = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                return "conda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Common locations
        candidates = []
        if sys.platform == "win32":
            for root in [os.environ.get("CONDA_PREFIX", ""), "D:\\Anaconda3", "C:\\Anaconda3",
                         os.path.expanduser("~/miniconda3"), os.path.expanduser("~/anaconda3")]:
                if root:
                    candidates.append(os.path.join(root, "Scripts", "conda.exe"))
                    candidates.append(os.path.join(root, "condabin", "conda.bat"))
        else:
            for root in [os.environ.get("CONDA_PREFIX", ""),
                         os.path.expanduser("~/miniconda3"), os.path.expanduser("~/anaconda3"),
                         "/opt/conda"]:
                if root:
                    candidates.append(os.path.join(root, "bin", "conda"))

        for cand in candidates:
            if os.path.isfile(cand):
                return cand

        raise FileNotFoundError(
            "conda not found. Please set conda_path in config or install conda."
        )

    def _env_prefix(self, env_name: str) -> str:
        """Return the full prefix path for an environment."""
        return os.path.join(self.envs_dir, env_name)

    def _run_cmd(self, cmd: List[str], timeout: int = 600) -> Tuple[bool, str, str]:
        """Run a subprocess command with timeout."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"TIMEOUT after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def env_exists(self, env_name: str) -> bool:
        """Check if the environment directory exists and contains python."""
        prefix = self._env_prefix(env_name)
        python = self.get_python_path(env_name)
        return os.path.isdir(prefix) and os.path.isfile(python)

    def create_env(self, env_name: str) -> str:
        """Create a conda environment with --prefix."""
        prefix = self._env_prefix(env_name)
        _log(f"Creating conda env at: {prefix}")
        cmd = [
            self.conda_path, "create", "--prefix", prefix,
            f"python={self.python_version}", "-y", "--no-default-packages"
        ]
        ok, stdout, stderr = self._run_cmd(cmd, timeout=600)
        if not ok:
            raise RuntimeError(f"conda create failed:\n{stderr}")
        python_path = self.get_python_path(env_name)
        if not os.path.isfile(python_path):
            raise RuntimeError(
                f"conda create succeeded but python not found at: {python_path}"
            )
        _log(f"Environment created. Python: {python_path}")
        return python_path

    def remove_env(self, env_name: str) -> bool:
        """Remove a conda environment by prefix."""
        prefix = self._env_prefix(env_name)
        if not os.path.isdir(prefix):
            return True
        _log(f"Removing environment: {prefix}")
        cmd = [self.conda_path, "env", "remove", "--prefix", prefix, "-y"]
        ok, stdout, stderr = self._run_cmd(cmd, timeout=300)
        if not ok:
            _log(f"conda env remove failed: {stderr}")
            # Fallback: try shutil
            import shutil
            try:
                shutil.rmtree(prefix)
                ok = True
            except Exception as e:
                _log(f"shutil.rmtree also failed: {e}")
        return ok

    def get_python_path(self, env_name: str) -> str:
        """Get the python executable path for an environment."""
        prefix = self._env_prefix(env_name)
        if sys.platform == "win32":
            return os.path.join(prefix, "python.exe")
        else:
            return os.path.join(prefix, "bin", "python")

    def pip_install_from_file(self, python_path: str, requirements_path: str,
                              timeout: int = 300) -> Tuple[bool, str, str]:
        """Run pip install -r requirements.txt."""
        _log(f"Installing from {requirements_path}")
        cmd = [python_path, "-m", "pip", "install", "-r", requirements_path]
        return self._run_cmd(cmd, timeout=timeout)

    def pip_install(self, python_path: str, packages: List[str],
                    timeout: int = 300) -> Tuple[bool, str, str]:
        """Install specific packages via pip."""
        if not packages:
            return True, "", ""
        _log(f"Installing: {', '.join(packages)}")
        cmd = [python_path, "-m", "pip", "install"] + packages
        return self._run_cmd(cmd, timeout=timeout)

    def pip_install_single(self, python_path: str, package: str,
                           timeout: int = 120) -> Tuple[bool, str, str]:
        """Install a single package via pip. Returns per-package result."""
        cmd = [python_path, "-m", "pip", "install", package]
        return self._run_cmd(cmd, timeout=timeout)

    def verify_imports(self, python_path: str, packages: List[str]) -> Dict[str, Tuple[bool, str]]:
        """Verify each package can be imported. Returns {pkg: (ok, error)}.
        Tries multiple import name candidates for robustness."""
        results = {}
        for pkg in packages:
            pkg_name = extract_package_name(pkg)
            import_name = _guess_import_name(pkg_name)
            if import_name is None:
                # Meta-package, skip
                results[pkg_name] = (True, "")
                continue
            # Try the guessed name first, then the original name as-is
            candidates = [import_name]
            if pkg_name != import_name:
                candidates.append(pkg_name)
            # Also try original case from the specifier
            original_name = re.split(r"[><=!~\[]", pkg)[0].strip()
            if original_name not in candidates:
                candidates.append(original_name)

            ok = False
            last_err = ""
            for candidate in candidates:
                cmd = [python_path, "-c", f"import {candidate}"]
                ok, stdout, stderr = self._run_cmd(cmd, timeout=30)
                if ok:
                    break
                last_err = stderr.strip()
            results[pkg_name] = (ok, last_err if not ok else "")
        return results

    def get_installed_packages(self, python_path: str) -> str:
        """Get installed packages as formatted string."""
        cmd = [python_path, "-m", "pip", "list", "--format=columns"]
        ok, stdout, stderr = self._run_cmd(cmd, timeout=15)
        if ok:
            return stdout
        return ""

    def check_requirements_satisfied(self, python_path: str,
                                     requirements: List[str]) -> bool:
        """Quick check: are all requirements already installed?"""
        cmd = [python_path, "-m", "pip", "check"]
        ok, _, _ = self._run_cmd(cmd, timeout=30)
        if not ok:
            return False
        # Verify imports as well
        results = self.verify_imports(python_path, requirements)
        return all(ok for ok, _ in results.values())


# ---------------------------------------------------------------------------
# EnvSetupAgent — LLM-assisted environment diagnosis
# ---------------------------------------------------------------------------
class EnvSetupAgent(BaseAgent):
    """LLM agent that diagnoses and fixes environment setup failures."""

    def __init__(self, client: Any, model_name: str, conda_manager: CondaEnvManager,
                 config: dict = None, temperature: float = 0.0):
        self.conda_manager = conda_manager
        self._config = config or {}
        cc_cfg = self._config.get("code_cleaner", {})
        env_cfg = cc_cfg.get("env_setup", {})
        self.max_fix_retries = env_cfg.get("max_fix_retries", 3)
        self.pip_timeout = env_cfg.get("pip_install_timeout", 300)
        self.llm_enabled = client is not None
        if self.llm_enabled:
            super().__init__(client, model_name, temperature=temperature)
        else:
            # Skip BaseAgent init (no LLM client)
            self.client = None
            self.model_name = ""
            self.temperature = temperature
            self.system_prompt = ""

    def _build_system_prompt(self) -> str:
        return """You are an expert Python environment configuration specialist.

Your task: Given a list of packages that failed to install or import in a conda
environment, diagnose the root cause and provide concrete fix actions.

Common failure patterns you should handle:
1. Package name mismatch (pip name != import name)
2. Version conflicts between packages
3. Missing system-level dependencies (e.g. HDF5 headers for h5py)
4. Platform-specific issues (Windows vs Linux compilation)
5. Packages requiring specific build tools
6. Deprecated or renamed packages
7. Packages only available from specific indices
8. Case-sensitive package names on PyPI (e.g. "ptylab" might be "PtyLab" on PyPI)
9. Packages installable from GitHub (e.g. pip install git+https://github.com/...)

IMPORTANT: PyPI package names are case-sensitive in some contexts. If a package
is "not found", try alternate capitalizations (e.g. PtyLab, ptyLab) or search
for the correct PyPI name. Prefer "replace_package" with the correct name over "skip".

Output format: Respond with ONLY a JSON object (no markdown, no explanation):
{
  "diagnosis": "Brief explanation of what went wrong",
  "fix_actions": [
    {
      "action_type": "pin_version|replace_package|install_dependency|skip|conda_install",
      "original_package": "the failing specifier",
      "replacement": "the corrected pip specifier",
      "reason": "why this fix works"
    }
  ],
  "pre_install_commands": []
}

Rules:
- action_type "pin_version": change version constraint
- action_type "replace_package": use a different package name
- action_type "install_dependency": install a missing dependency first
- action_type "skip": skip this package (it's optional or a meta-package)
- action_type "conda_install": install via conda instead of pip
- pre_install_commands: conda commands to run before pip install (for system deps)
- Never suggest installing from unverified sources
- Be aware of the platform (Windows/Linux) when suggesting fixes"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        parts = [
            f"Task: {context['task_name']}",
            f"Platform: {context['platform']}",
            f"Python version: {context['python_version']}",
            f"\nOriginal requirements.txt:\n{context['requirements_txt']}",
        ]
        if context.get("installed_packages"):
            parts.append(f"\nCurrently installed packages:\n{context['installed_packages'][:3000]}")
        parts.append("\nFailed packages:")
        for fail in context.get("failed_packages", []):
            error_msg = fail.get("error", "Unknown error")[:500]
            parts.append(f"  - {fail['name']}: {error_msg}")
        if context.get("previous_attempts"):
            parts.append(f"\nPrevious fix attempts that FAILED (do NOT repeat these):\n{context['previous_attempts']}")
        if context.get("reference_info"):
            parts.append(f"\nReference info from task directory:\n{context['reference_info']}")
        parts.append("\nPlease diagnose and provide fix actions.")
        return "\n".join(parts)

    def diagnose_failures(self, task_name: str, requirements_txt: str,
                          failed_packages: List[Dict], installed_packages: str,
                          python_version: str, previous_attempts: str = "",
                          reference_info: str = "") -> Dict:
        """Call LLM to diagnose installation failures. Returns fix plan."""
        context = {
            "task_name": task_name,
            "requirements_txt": requirements_txt,
            "failed_packages": failed_packages,
            "installed_packages": installed_packages,
            "platform": platform.platform(),
            "python_version": python_version,
            "previous_attempts": previous_attempts,
            "reference_info": reference_info,
        }
        raw = self.generate(context)
        try:
            return json.loads(extract_json(raw))
        except (json.JSONDecodeError, Exception) as e:
            _log(f"LLM response parse failed: {e}")
            return {"diagnosis": raw[:500], "fix_actions": [], "pre_install_commands": []}

    def _apply_fix_actions(self, python_path: str, fix_plan: Dict) -> List[str]:
        """Apply the fix actions from LLM diagnosis. Returns list of applied fixes."""
        applied = []

        # Run pre_install_commands (conda commands)
        for cmd_str in fix_plan.get("pre_install_commands", []):
            _log(f"Running pre-install: {cmd_str}")
            parts = cmd_str.split()
            # Replace 'conda' with the actual conda path and inject --prefix
            if parts and parts[0] == "conda":
                parts[0] = self.conda_manager.conda_path
            ok, stdout, stderr = self.conda_manager._run_cmd(parts, timeout=300)
            if ok:
                applied.append(f"pre_install: {cmd_str}")
            else:
                _log(f"Pre-install failed: {stderr[:200]}")

        # Apply individual fix actions
        for action in fix_plan.get("fix_actions", []):
            action_type = action.get("action_type", "")
            original = action.get("original_package", "")
            replacement = action.get("replacement", "")
            reason = action.get("reason", "")

            _log(f"Applying fix: {action_type} | {original} -> {replacement} | {reason}")

            if action_type == "skip":
                applied.append(f"skip: {original}")
                continue

            if action_type == "conda_install" and replacement:
                cmd = [self.conda_manager.conda_path, "install", "--prefix",
                       os.path.dirname(os.path.dirname(python_path) if sys.platform != "win32"
                                       else os.path.dirname(python_path)),
                       replacement, "-y"]
                ok, _, stderr = self.conda_manager._run_cmd(cmd, timeout=300)
                if ok:
                    applied.append(f"conda_install: {replacement}")
                else:
                    _log(f"conda install failed: {stderr[:200]}")
                continue

            if action_type in ("pin_version", "replace_package", "install_dependency",
                               "install_from_url"):
                if replacement:
                    ok, _, stderr = self.conda_manager.pip_install_single(
                        python_path, replacement, timeout=self.pip_timeout
                    )
                    if ok:
                        applied.append(f"{action_type}: {replacement}")
                    else:
                        _log(f"pip install {replacement} failed: {stderr[:200]}")

        return applied

    def setup_single_task(self, task_dir: str, task_name: str,
                          force_rebuild: bool = False) -> Optional[str]:
        """Set up conda environment for a single task. Returns python_path or None."""
        env_cfg = self._config.get("code_cleaner", {}).get("env_setup", {})
        env_prefix = env_cfg.get("env_prefix", "task_")
        env_name = f"{env_prefix}{task_name}"

        req_path = os.path.join(task_dir, "requirements.txt")
        if not os.path.isfile(req_path):
            _log(f"[{task_name}] No requirements.txt found, skipping.")
            return None

        requirements = parse_requirements(req_path)
        if not requirements:
            _log(f"[{task_name}] requirements.txt is empty, skipping.")
            return None

        _log(f"[{task_name}] Requirements: {', '.join(requirements)}")

        # Check if env already exists
        if self.conda_manager.env_exists(env_name) and not force_rebuild:
            python_path = self.conda_manager.get_python_path(env_name)
            _log(f"[{task_name}] Environment exists. Checking dependencies...")
            if self.conda_manager.check_requirements_satisfied(python_path, requirements):
                _log(f"[{task_name}] All dependencies satisfied. Skipping.")
                return python_path
            _log(f"[{task_name}] Dependencies incomplete. Will install missing packages.")
        else:
            if force_rebuild and self.conda_manager.env_exists(env_name):
                self.conda_manager.remove_env(env_name)
            python_path = self.conda_manager.create_env(env_name)

        # Install requirements
        ok, stdout, stderr = self.conda_manager.pip_install_from_file(
            python_path, req_path, timeout=self.pip_timeout
        )
        if not ok:
            _log(f"[{task_name}] Batch install failed. Trying individual packages...")
            # Install one-by-one to isolate failures
            for spec in requirements:
                ok_s, _, err_s = self.conda_manager.pip_install_single(
                    python_path, spec, timeout=self.pip_timeout
                )
                if not ok_s:
                    _log(f"[{task_name}] Failed: {spec} -> {err_s[:200]}")

        # Load reference info (e.g. GitHub URLs) from task directory
        reference_info = ""
        ref_file = os.path.join(task_dir, "reference_website_github.md")
        if os.path.isfile(ref_file):
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_info = f.read().strip()

        # Verify imports
        previous_attempts = []
        for attempt in range(self.max_fix_retries + 1):
            results = self.conda_manager.verify_imports(python_path, requirements)
            failed = {pkg: err for pkg, (ok, err) in results.items() if not ok}

            if not failed:
                _log(f"[{task_name}] All {len(results)} packages verified successfully.")
                return python_path

            _log(f"[{task_name}] Verification failed for {len(failed)} packages: "
                 f"{list(failed.keys())} (attempt {attempt + 1}/{self.max_fix_retries + 1})")

            if attempt >= self.max_fix_retries:
                break

            if not self.llm_enabled:
                _log(f"[{task_name}] LLM disabled. Cannot auto-fix. Stopping retries.")
                break

            # LLM diagnosis
            _log(f"[{task_name}] Calling LLM for diagnosis...")
            try:
                with open(req_path, "r", encoding="utf-8") as f:
                    req_text = f.read()

                failed_list = [{"name": pkg, "error": err} for pkg, err in failed.items()]
                installed = self.conda_manager.get_installed_packages(python_path)

                # Get python version
                ok_v, py_ver, _ = self.conda_manager._run_cmd(
                    [python_path, "--version"], timeout=10
                )
                python_version = py_ver.strip() if ok_v else "unknown"

                prev_str = "\n".join(previous_attempts) if previous_attempts else ""
                fix_plan = self.diagnose_failures(
                    task_name, req_text, failed_list, installed, python_version,
                    previous_attempts=prev_str, reference_info=reference_info,
                )
                _log(f"[{task_name}] LLM diagnosis: {fix_plan.get('diagnosis', 'N/A')}")

                applied = self._apply_fix_actions(python_path, fix_plan)
                _log(f"[{task_name}] Applied {len(applied)} fixes: {applied}")
                # Record this attempt to avoid LLM repeating same fix
                for action in fix_plan.get("fix_actions", []):
                    previous_attempts.append(
                        f"Tried {action.get('action_type')}: {action.get('replacement', '')} -> FAILED"
                    )
            except Exception as e:
                _log(f"[{task_name}] LLM diagnosis failed: {e}. Continuing without fix.")

        # Final status
        final_results = self.conda_manager.verify_imports(python_path, requirements)
        final_failed = {pkg: err for pkg, (ok, err) in final_results.items() if not ok}
        if final_failed:
            _log(f"[{task_name}] WARNING: {len(final_failed)} packages still failing: "
                 f"{list(final_failed.keys())}")
            _log(f"[{task_name}] Returning python_path anyway (partial environment).")
        return python_path

    def setup_all_tasks(self, tasks_dir: str,
                        force_rebuild: bool = False,
                        task_filter: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
        """Set up environments for all tasks. Returns {task_name: python_path}."""
        tasks_dir = os.path.abspath(tasks_dir)
        if not os.path.isdir(tasks_dir):
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

        # Scan for task folders with requirements.txt
        task_folders = []
        for entry in sorted(os.listdir(tasks_dir)):
            full_path = os.path.join(tasks_dir, entry)
            if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "requirements.txt")):
                task_folders.append((entry, full_path))

        if task_filter:
            task_folders = [(name, path) for name, path in task_folders if name in task_filter]

        _log(f"Found {len(task_folders)} tasks: {[n for n, _ in task_folders]}")

        results = {}
        for task_name, task_dir in task_folders:
            _log(f"\n{'='*50}")
            _log(f"Setting up: {task_name}")
            _log(f"{'='*50}")
            try:
                python_path = self.setup_single_task(task_dir, task_name, force_rebuild)
                results[task_name] = python_path
            except Exception as e:
                _log(f"[{task_name}] FATAL ERROR: {e}")
                results[task_name] = None

        # Summary
        _log(f"\n{'='*50}")
        _log("ENVIRONMENT SETUP SUMMARY")
        _log(f"{'='*50}")
        for name, path in results.items():
            status = "OK" if path else "FAILED"
            _log(f"  {name}: [{status}] {path or 'N/A'}")

        return results
