from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentPlan:
    backend: str
    env_dir: Path
    python_executable: Path
    backend_preference: str = "auto"
    python_version: str = "3.10"
    manifest_paths: List[Path] = field(default_factory=list)
    install_commands: List[List[str]] = field(default_factory=list)
    optional_install_commands: List[List[str]] = field(default_factory=list)
    smoke_test_commands: List[List[str]] = field(default_factory=list)
    inferred_packages: List[str] = field(default_factory=list)
    smoke_imports: List[str] = field(default_factory=list)
    gpu_mode: str = "auto"
    repo_requests_gpu: bool = False
    gpu_available: bool = False
    gpu_enabled: bool = False
    gpu_package: Optional[str] = None
    provisioned: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class RepoDiscovery:
    repo_root: Path
    entry_module: Path
    python_files: List[Path]
    readme_files: List[Path] = field(default_factory=list)
    config_files: List[Path] = field(default_factory=list)
    test_files: List[Path] = field(default_factory=list)
    data_candidates: List[str] = field(default_factory=list)
    main_candidates: List[str] = field(default_factory=list)
    eval_candidates: List[str] = field(default_factory=list)
    class_candidates: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    accepted: bool
    status: str
    comparison: str
    summary: str
    candidate_source: str = "unknown"
    baseline_available: bool = False
    cleaned_metrics: Dict[str, Any] = field(default_factory=dict)
    original_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


@dataclass
class CleaningResult:
    repo_name: str
    source_root: Path
    run_dir: Path
    cleaned_code_path: Path
    environment_plan: EnvironmentPlan
    discovery: RepoDiscovery
    validation: ValidationReport
    artifacts: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
