from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from .models import EnvironmentPlan


PACKAGE_NAME_MAP = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "pywt": "PyWavelets",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
}
IMPORT_NAME_MAP = {
    "PyWavelets": "pywt",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
}
BASELINE_RUNTIME_PACKAGES = ("numpy",)
GPU_IMPORTS = {"cupy"}


class EnvironmentResolver:
    def __init__(
        self,
        sandbox_root: Path,
        *,
        backend_preference: str = "auto",
        python_version: str = "3.10",
        gpu_mode: str = "auto",
        force_rebuild: bool = False,
    ) -> None:
        self.sandbox_root = sandbox_root
        self.backend_preference = backend_preference
        self.python_version = python_version
        self.gpu_mode = gpu_mode
        self.force_rebuild = force_rebuild

    def resolve(self, repo_root: Path, repo_name: str) -> EnvironmentPlan:
        env_dir = self.sandbox_root / f"{repo_name}_env"
        conda_manifest = self._first_existing(
            repo_root / "environment.yml",
            repo_root / "environment.yaml",
            repo_root / "conda.yml",
            repo_root / "conda.yaml",
        )
        requirements = self._first_existing(repo_root / "requirements.txt")
        pyproject = self._first_existing(repo_root / "pyproject.toml")
        conda_executable = self._find_conda_executable()
        gpu_info = self._detect_gpu_info()

        inferred_imports = self._infer_external_imports(repo_root)
        repo_requests_gpu = any(module_name in GPU_IMPORTS for module_name in inferred_imports)
        runtime_imports = [module_name for module_name in inferred_imports if module_name not in GPU_IMPORTS]
        inferred_packages = self._resolve_package_names(runtime_imports)
        smoke_imports = self._resolve_import_names(inferred_packages)
        gpu_package = self._select_gpu_package(
            repo_requests_gpu=repo_requests_gpu,
            gpu_available=gpu_info["available"],
            cuda_major=gpu_info["cuda_major"],
        )

        backend = self._select_backend(conda_manifest=conda_manifest, conda_executable=conda_executable)
        if backend == "conda":
            return self._build_conda_plan(
                repo_root=repo_root,
                env_dir=env_dir,
                conda_executable=conda_executable,
                conda_manifest=conda_manifest,
                requirements=requirements,
                pyproject=pyproject,
                inferred_imports=runtime_imports,
                inferred_packages=inferred_packages,
                smoke_imports=smoke_imports,
                repo_requests_gpu=repo_requests_gpu,
                gpu_available=gpu_info["available"],
                gpu_package=gpu_package,
                cuda_version=gpu_info["cuda_version"],
            )

        return self._build_venv_plan(
            repo_root=repo_root,
            env_dir=env_dir,
            requirements=requirements,
            pyproject=pyproject,
            inferred_imports=runtime_imports,
            inferred_packages=inferred_packages,
            smoke_imports=smoke_imports,
            repo_requests_gpu=repo_requests_gpu,
            gpu_available=gpu_info["available"],
            gpu_package=gpu_package,
            conda_missing=(conda_executable is None),
            cuda_version=gpu_info["cuda_version"],
        )

    def provision(self, plan: EnvironmentPlan) -> EnvironmentPlan:
        if self.force_rebuild and plan.env_dir.exists():
            self._clear_environment(plan.env_dir)
            plan.notes.append("Existing environment removed because force_rebuild was requested.")
        elif plan.env_dir.exists():
            if plan.python_executable.exists():
                plan.notes.append(f"Reusing existing environment at {plan.env_dir}")
                plan.gpu_enabled = self._check_gpu_runtime(plan)
                plan.provisioned = True
                return plan
            plan.notes.append(f"Existing environment directory at {plan.env_dir} is incomplete; rebuilding.")
            self._clear_environment(plan.env_dir)

        plan.env_dir.parent.mkdir(parents=True, exist_ok=True)
        for command in plan.install_commands:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Environment provisioning failed for `{plan.backend}` with command {command}: {result.stderr.strip()}"
                )

        for command in plan.optional_install_commands:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                message = f"Optional install failed for command {command}: {result.stderr.strip() or result.stdout.strip()}"
                plan.notes.append(message)
                if plan.gpu_mode == "on":
                    raise RuntimeError(message)

        for command in plan.smoke_test_commands:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Environment smoke test failed with command {command}: {result.stderr.strip()}"
                )

        plan.gpu_enabled = self._check_gpu_runtime(plan)
        plan.provisioned = True
        return plan

    def _environment_healthy(self, plan: EnvironmentPlan) -> bool:
        if not plan.python_executable.exists():
            return False
        basic_checks = [[str(plan.python_executable), "--version"], *plan.smoke_test_commands]
        for command in basic_checks:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return False
        return True

    def _clear_environment(self, env_dir: Path) -> None:
        if env_dir.is_symlink() or env_dir.is_file():
            env_dir.unlink()
            return
        shutil.rmtree(env_dir, ignore_errors=False)

    def _build_conda_plan(
        self,
        *,
        repo_root: Path,
        env_dir: Path,
        conda_executable: str | None,
        conda_manifest: Path | None,
        requirements: Path | None,
        pyproject: Path | None,
        inferred_imports: List[str],
        inferred_packages: List[str],
        smoke_imports: List[str],
        repo_requests_gpu: bool,
        gpu_available: bool,
        gpu_package: str | None,
        cuda_version: str | None,
    ) -> EnvironmentPlan:
        if not conda_executable:
            raise RuntimeError("Conda backend selected but no `conda` executable was found.")

        python_executable = env_dir / ("python.exe" if os.name == "nt" else "bin/python")
        install_commands: List[List[str]] = []
        manifest_paths: List[Path] = []

        if conda_manifest is not None:
            manifest_paths.append(conda_manifest)
            if self._manifest_declares_python(conda_manifest):
                install_commands.append(
                    [
                        conda_executable,
                        "env",
                        "create",
                        "--prefix",
                        str(env_dir),
                        "--file",
                        str(conda_manifest),
                        "--yes",
                    ]
                )
            else:
                install_commands.append(
                    [
                        conda_executable,
                        "create",
                        "--prefix",
                        str(env_dir),
                        f"python={self.python_version}",
                        "--yes",
                    ]
                )
                install_commands.append(
                    [
                        conda_executable,
                        "env",
                        "update",
                        "--prefix",
                        str(env_dir),
                        "--file",
                        str(conda_manifest),
                        "--prune",
                    ]
                )
        else:
            install_commands.append(
                [
                    conda_executable,
                    "create",
                    "--prefix",
                    str(env_dir),
                    f"python={self.python_version}",
                    "--yes",
                ]
            )

        install_commands.append([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])
        if requirements is not None:
            install_commands.append([str(python_executable), "-m", "pip", "install", "-r", str(requirements)])
            manifest_paths.append(requirements)
        elif pyproject is not None:
            install_commands.append([str(python_executable), "-m", "pip", "install", "-e", str(repo_root)])
            manifest_paths.append(pyproject)

        if inferred_packages:
            install_commands.append([str(python_executable), "-m", "pip", "install", *inferred_packages])

        optional_install_commands = self._gpu_install_commands(
            python_executable=python_executable,
            gpu_package=gpu_package,
        )
        smoke_commands = [
            [str(python_executable), "--version"],
            self._import_smoke_command(python_executable, smoke_imports),
        ]

        return EnvironmentPlan(
            backend="conda",
            env_dir=env_dir,
            python_executable=python_executable,
            backend_preference=self.backend_preference,
            python_version=self.python_version,
            manifest_paths=manifest_paths,
            install_commands=install_commands,
            optional_install_commands=optional_install_commands,
            smoke_test_commands=smoke_commands,
            inferred_packages=inferred_packages,
            smoke_imports=smoke_imports,
            gpu_mode=self.gpu_mode,
            repo_requests_gpu=repo_requests_gpu,
            gpu_available=gpu_available,
            gpu_enabled=False,
            gpu_package=gpu_package,
            notes=[
                "Resolved via conda preference.",
                *self._dependency_notes(
                    inferred_imports=inferred_imports,
                    inferred_packages=inferred_packages,
                    repo_requests_gpu=repo_requests_gpu,
                    gpu_available=gpu_available,
                    gpu_package=gpu_package,
                    cuda_version=cuda_version,
                ),
            ],
        )

    def _build_venv_plan(
        self,
        *,
        repo_root: Path,
        env_dir: Path,
        requirements: Path | None,
        pyproject: Path | None,
        inferred_imports: List[str],
        inferred_packages: List[str],
        smoke_imports: List[str],
        repo_requests_gpu: bool,
        gpu_available: bool,
        gpu_package: str | None,
        conda_missing: bool,
        cuda_version: str | None,
    ) -> EnvironmentPlan:
        python_executable = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        install_commands: List[List[str]] = [
            [sys.executable, "-m", "venv", str(env_dir)],
            [str(python_executable), "-m", "pip", "install", "--upgrade", "pip"],
        ]
        manifest_paths: List[Path] = []
        if requirements is not None:
            install_commands.append([str(python_executable), "-m", "pip", "install", "-r", str(requirements)])
            manifest_paths.append(requirements)
        elif pyproject is not None:
            install_commands.append([str(python_executable), "-m", "pip", "install", "-e", str(repo_root)])
            manifest_paths.append(pyproject)

        if inferred_packages:
            install_commands.append([str(python_executable), "-m", "pip", "install", *inferred_packages])

        optional_install_commands = self._gpu_install_commands(
            python_executable=python_executable,
            gpu_package=gpu_package,
        )
        smoke_commands = [
            [str(python_executable), "--version"],
            self._import_smoke_command(python_executable, smoke_imports),
        ]

        notes = ["Resolved via venv fallback."]
        if conda_missing and self.backend_preference == "auto":
            notes.append("Conda was not available, so the resolver fell back to venv.")
        notes.extend(
            self._dependency_notes(
                inferred_imports=inferred_imports,
                inferred_packages=inferred_packages,
                repo_requests_gpu=repo_requests_gpu,
                gpu_available=gpu_available,
                gpu_package=gpu_package,
                cuda_version=cuda_version,
            )
        )

        return EnvironmentPlan(
            backend="venv",
            env_dir=env_dir,
            python_executable=python_executable,
            backend_preference=self.backend_preference,
            python_version=self.python_version,
            manifest_paths=manifest_paths,
            install_commands=install_commands,
            optional_install_commands=optional_install_commands,
            smoke_test_commands=smoke_commands,
            inferred_packages=inferred_packages,
            smoke_imports=smoke_imports,
            gpu_mode=self.gpu_mode,
            repo_requests_gpu=repo_requests_gpu,
            gpu_available=gpu_available,
            gpu_enabled=False,
            gpu_package=gpu_package,
            notes=notes,
        )

    def _select_backend(self, *, conda_manifest: Path | None, conda_executable: str | None) -> str:
        if self.backend_preference == "venv":
            return "venv"
        if self.backend_preference == "conda":
            if conda_executable is None:
                raise RuntimeError("`--env-backend conda` was requested but `conda` was not found on PATH.")
            return "conda"
        if conda_executable is not None:
            return "conda"
        return "venv"

    def _find_conda_executable(self) -> str | None:
        conda_env = os.environ.get("CONDA_EXE")
        if conda_env:
            return conda_env
        return shutil.which("conda")

    def _detect_gpu_info(self) -> dict:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return {"available": False, "cuda_version": None, "cuda_major": None}
        result = subprocess.run([nvidia_smi], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return {"available": False, "cuda_version": None, "cuda_major": None}
        output = result.stdout + "\n" + result.stderr
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", output)
        cuda_version = match.group(1) if match else None
        cuda_major = None
        if cuda_version:
            try:
                cuda_major = int(cuda_version.split(".", 1)[0])
            except ValueError:
                cuda_major = None
        return {"available": True, "cuda_version": cuda_version, "cuda_major": cuda_major}

    def _select_gpu_package(self, *, repo_requests_gpu: bool, gpu_available: bool, cuda_major: int | None) -> str | None:
        if not repo_requests_gpu or self.gpu_mode == "off":
            return None
        if self.gpu_mode == "auto" and not gpu_available:
            return None
        if cuda_major and cuda_major >= 12:
            return "cupy-cuda12x"
        if cuda_major == 11:
            return "cupy-cuda11x"
        return "cupy"

    def _gpu_install_commands(self, *, python_executable: Path, gpu_package: str | None) -> List[List[str]]:
        if not gpu_package:
            return []
        return [[str(python_executable), "-m", "pip", "install", gpu_package]]

    def _check_gpu_runtime(self, plan: EnvironmentPlan) -> bool:
        if not plan.repo_requests_gpu or plan.gpu_mode == "off":
            plan.notes.append("GPU runtime not requested for this repository.")
            return False
        command = [
            str(plan.python_executable),
            "-c",
            (
                "import cupy; "
                "from cupy.cuda import runtime; "
                "count = runtime.getDeviceCount(); "
                "print(count)"
            ),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            device_count = (result.stdout or "").strip() or "unknown"
            plan.notes.append(f"GPU runtime enabled with CuPy. Visible device count: {device_count}")
            return True
        if plan.gpu_mode == "on":
            raise RuntimeError(f"GPU mode was forced but CuPy runtime was unavailable: {result.stderr.strip()}")
        plan.notes.append("GPU runtime unavailable after provisioning; cleaner will fall back to CPU compatibility.")
        return False

    def _infer_external_imports(self, repo_root: Path) -> List[str]:
        python_files = [
            path
            for path in repo_root.rglob("*.py")
            if ".venv" not in path.parts and "__pycache__" not in path.parts and ".git" not in path.parts
        ]
        local_roots = self._local_roots(repo_root, python_files)
        imports = set()
        stdlib_names = set(getattr(sys, "stdlib_module_names", set()))

        for path in python_files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".", 1)[0]
                        if root and root not in local_roots and root not in stdlib_names:
                            imports.add(root)
                elif isinstance(node, ast.ImportFrom):
                    if node.level or not node.module:
                        continue
                    root = node.module.split(".", 1)[0]
                    if root and root not in local_roots and root not in stdlib_names:
                        imports.add(root)

        return sorted(imports)

    def _local_roots(self, repo_root: Path, python_files: Iterable[Path]) -> set[str]:
        roots = {path.stem for path in repo_root.glob("*.py")}
        roots.update(path.name for path in repo_root.iterdir() if path.is_dir() and (path / "__init__.py").exists())
        for path in python_files:
            try:
                first = path.relative_to(repo_root).parts[0]
            except ValueError:
                continue
            if first.endswith(".py"):
                roots.add(Path(first).stem)
            else:
                roots.add(first)
        return roots

    def _resolve_package_names(self, imports: Iterable[str]) -> List[str]:
        packages = list(BASELINE_RUNTIME_PACKAGES)
        for module_name in imports:
            package_name = PACKAGE_NAME_MAP.get(module_name, module_name)
            if package_name not in packages:
                packages.append(package_name)
        return packages

    def _resolve_import_names(self, packages: Iterable[str]) -> List[str]:
        import_names = []
        for package_name in packages:
            module_name = IMPORT_NAME_MAP.get(package_name, package_name)
            if module_name not in import_names:
                import_names.append(module_name)
        return import_names

    def _import_smoke_command(self, python_executable: Path, import_names: Iterable[str]) -> List[str]:
        smoke_line = "; ".join([f"import {name}" for name in import_names]) or "pass"
        return [str(python_executable), "-c", smoke_line]

    def _dependency_notes(
        self,
        *,
        inferred_imports: Iterable[str],
        inferred_packages: Iterable[str],
        repo_requests_gpu: bool,
        gpu_available: bool,
        gpu_package: str | None,
        cuda_version: str | None,
    ) -> List[str]:
        imports = list(inferred_imports)
        packages = list(inferred_packages)
        notes = []
        notes.append(f"Backend preference: {self.backend_preference}")
        notes.append(f"Python version preference: {self.python_version}")
        notes.append(f"GPU mode: {self.gpu_mode}")
        if imports:
            notes.append(f"Inferred runtime imports from source: {', '.join(imports)}")
        if packages:
            notes.append(f"Packages scheduled for install: {', '.join(packages)}")
        if repo_requests_gpu:
            notes.append("Repository imports CuPy or another GPU-specific path.")
            if gpu_available:
                if cuda_version:
                    notes.append(f"Detected NVIDIA GPU with CUDA {cuda_version}.")
                else:
                    notes.append("Detected NVIDIA GPU.")
            else:
                notes.append("No NVIDIA GPU was detected from the host environment.")
            if gpu_package:
                notes.append(f"Selected GPU runtime package: {gpu_package}")
        return notes

    def _manifest_declares_python(self, manifest_path: Path) -> bool:
        text = manifest_path.read_text(encoding="utf-8", errors="ignore")
        return bool(re.search(r"^\s*-\s*python(?:[<>=! ]|$)", text, flags=re.IGNORECASE | re.MULTILINE))

    def _first_existing(self, *paths: Path) -> Path | None:
        for path in paths:
            if path.exists():
                return path.resolve()
        return None
