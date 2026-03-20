from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


def repo_name_from_input(github_url: Optional[str], local_repo: Optional[str]) -> str:
    if github_url:
        return Path(github_url.rstrip("/")).stem.replace(".git", "")
    if local_repo:
        return Path(local_repo).expanduser().resolve().name
    raise ValueError("Either github_url or local_repo must be provided.")


def acquire_repository(
    *,
    github_url: Optional[str],
    local_repo: Optional[str],
    destination_root: Path,
) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    repo_name = repo_name_from_input(github_url, local_repo)
    destination = destination_root / repo_name

    if destination.exists():
        shutil.rmtree(destination)

    if local_repo:
        source = Path(local_repo).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Local repository not found: {source}")
        shutil.copytree(source, destination)
        return destination

    if not github_url:
        raise ValueError("github_url is required when local_repo is not provided.")

    local_candidate = Path(github_url).expanduser()
    if local_candidate.exists():
        shutil.copytree(local_candidate.resolve(), destination)
        return destination

    result = subprocess.run(
        ["git", "clone", github_url, str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return destination
