from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .compat import ensure_repo_compat_shims
from .discovery import discover_repository
from .environment import EnvironmentResolver
from .facade import FacadeSynthesizer
from .llm_synthesizer import LLMCleanerSynthesizer
from .models import CleaningResult, EnvironmentPlan, ValidationReport
from .source import acquire_repository, repo_name_from_input
from .validator import CleanerValidator


class CodeCleaner:
    def __init__(
        self,
        *,
        artifact_root: str,
        config: Optional[dict] = None,
        skill_manager=None,
        provision_environment: bool = True,
        env_backend: str = "auto",
        python_version: str = "3.10",
        gpu_mode: str = "auto",
        llm_client=None,
        llm_model_name: Optional[str] = None,
        llm_required: bool = False,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 12000,
        llm_max_loops: int = 3,
        force_rebuild_env: bool = False,
    ) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.skill_manager = skill_manager
        self.provision_environment = provision_environment
        self.env_backend = env_backend
        self.python_version = python_version
        self.gpu_mode = gpu_mode
        self.llm_client = llm_client
        self.llm_model_name = llm_model_name
        self.llm_required = llm_required
        self.force_rebuild_env = force_rebuild_env
        self.facade_synthesizer = FacadeSynthesizer()
        self.llm_synthesizer = (
            LLMCleanerSynthesizer(
                client=llm_client,
                model_name=llm_model_name,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                max_loops=llm_max_loops,
            )
            if llm_client is not None and llm_model_name
            else None
        )
        self._paper_md_content: Optional[str] = None
        self.validator = CleanerValidator()

    def clean(
        self,
        *,
        github_url: Optional[str] = None,
        local_repo: Optional[str] = None,
        paper_md: Optional[str] = None,
        task_family: Optional[str] = None,
    ) -> CleaningResult:
        repo_name = repo_name_from_input(github_url, local_repo)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.artifact_root / f"{repo_name}_{timestamp}"
        source_root = run_dir / "source_repo"
        run_dir.mkdir(parents=True, exist_ok=True)

        acquired_repo = acquire_repository(
            github_url=github_url,
            local_repo=local_repo,
            destination_root=source_root.parent,
        )
        if acquired_repo != source_root:
            if source_root.exists():
                shutil.rmtree(source_root)
            acquired_repo.rename(source_root)

        if paper_md:
            paper_src = Path(paper_md).expanduser().resolve()
            if paper_src.exists():
                shutil.copy2(paper_src, run_dir / paper_src.name)
                try:
                    self._paper_md_content = paper_src.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    self._paper_md_content = None

        sandbox_root = self._resolve_sandbox_root(repo_name=repo_name, run_dir=run_dir)
        env_resolver = EnvironmentResolver(
            sandbox_root,
            backend_preference=self.env_backend,
            python_version=self.python_version,
            gpu_mode=self.gpu_mode,
            force_rebuild=self.force_rebuild_env,
        )
        environment_plan = env_resolver.resolve(source_root, repo_name)
        try:
            if self.provision_environment:
                environment_plan = env_resolver.provision(environment_plan)
        except Exception as exc:
            environment_plan.notes.append(f"Provisioning skipped after failure: {exc}")

        compat_shims = []
        if not environment_plan.gpu_enabled:
            compat_shims = ensure_repo_compat_shims(source_root)
            if compat_shims:
                environment_plan.notes.append(
                    "Compatibility shims injected: " + ", ".join(str(path.relative_to(source_root)) for path in compat_shims)
                )

        discovery = discover_repository(source_root)
        cleaned_code_path = run_dir / "code_cleaned.py"

        best_report: Optional[ValidationReport] = None
        best_code: Optional[str] = None
        variants = self._build_variants(discovery)
        for index, variant in enumerate(variants, 1):
            cleaned_code_path.write_text(variant["code"], encoding="utf-8")
            metadata_path = run_dir / f"candidate_{index}_metadata.json"
            self._write_json(
                metadata_path,
                {
                    "candidate_index": index,
                    "source": variant["source"],
                    **variant.get("metadata", {}),
                },
            )
            report = self.validator.validate(
                run_dir=run_dir,
                discovery=discovery,
                plan=environment_plan,
                candidate_source=variant["source"],
            )
            report.artifacts["candidate_metadata"] = str(metadata_path)
            self._write_json(run_dir / f"validation_attempt_{index}.json", self._serialize_report(report))
            if best_report is None or (report.accepted and not best_report.accepted):
                best_report = report
                best_code = variant["code"]
            if report.accepted:
                break

        if best_code is not None:
            cleaned_code_path.write_text(best_code, encoding="utf-8")

        if best_report is None:
            best_report = ValidationReport(
                accepted=False,
                status="no_attempts",
                comparison="missing",
                summary="No validation attempt was executed.",
                candidate_source="none",
                baseline_available=False,
            )

        summary = self._build_summary(
            repo_name=repo_name,
            task_family=task_family,
            environment_plan=environment_plan,
            report=best_report,
        )

        result = CleaningResult(
            repo_name=repo_name,
            source_root=source_root,
            run_dir=run_dir,
            cleaned_code_path=cleaned_code_path,
            environment_plan=environment_plan,
            discovery=discovery,
            validation=best_report,
            artifacts={
                "source_root": str(source_root),
                "cleaned_code_path": str(cleaned_code_path),
                "environment_plan": str(run_dir / "environment_plan.json"),
                "discovery": str(run_dir / "discovery.json"),
                "validation": str(run_dir / "validation_summary.json"),
                "sandbox_root": str(sandbox_root),
                **(
                    {"compat_shims": ", ".join(str(path) for path in compat_shims)}
                    if compat_shims
                    else {}
                ),
            },
            summary=summary,
        )

        self._write_json(run_dir / "environment_plan.json", self._serialize_environment(environment_plan))
        self._write_json(run_dir / "discovery.json", self._serialize_discovery(discovery))
        self._write_json(run_dir / "validation_summary.json", self._serialize_report(best_report))
        (run_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")

        if best_report.accepted and self.skill_manager is not None and hasattr(self.skill_manager, "store_code_skill"):
            self._store_code_skill(result)

        return result

    def _store_code_skill(self, result: CleaningResult) -> None:
        cleaned_code = result.cleaned_code_path.read_text(encoding="utf-8")
        snippet = self._extract_code_snippet(cleaned_code)
        instructions = "\n".join([
            "## When to use",
            f"- Adapting validated code from `{result.repo_name}` into a unified wrapper.",
            "",
            "## Guidance",
            "- Prefer this wrapper for a stable single-file interface over heterogeneous research code.",
            "- Keep the adapter thin and validate wrapped behavior before reusing.",
            "",
            "## Constraints",
            "- Revalidate when the upstream entrypoint or environment changes.",
            "- Do not treat adapter equivalence as proof of scientific optimality.",
        ])
        self.skill_manager.store_code_skill(
            title=f"{result.repo_name}-validated-adapter",
            description=f"Validated code skill from cleaned repository `{result.repo_name}`.",
            instructions=instructions,
            code_snippet=snippet,
            repo_name=result.repo_name,
        )

    def _extract_code_snippet(self, cleaned_code: str) -> str:
        markers = ["def load_and_preprocess_data", "def main_process", "def evaluate_results"]
        lines = cleaned_code.splitlines()
        selected: list[str] = []
        capture = False
        for line in lines:
            if any(marker in line for marker in markers):
                capture = True
            if capture:
                selected.append(line)
            if capture and line.startswith("def _main"):
                break
        snippet = "\n".join(selected).strip()
        return snippet or cleaned_code[:1600]

    def _build_summary(
        self,
        *,
        repo_name: str,
        task_family: Optional[str],
        environment_plan: EnvironmentPlan,
        report: ValidationReport,
    ) -> str:
        family_line = f"Task family: `{task_family}`\n" if task_family else ""
        return (
            f"# Code Cleaner Summary: {repo_name}\n\n"
            f"{family_line}"
            f"- Backend: `{environment_plan.backend}`\n"
            f"- Python preference: `{environment_plan.python_version}`\n"
            f"- GPU mode: `{environment_plan.gpu_mode}`\n"
            f"- GPU enabled: `{environment_plan.gpu_enabled}`\n"
            f"- Provisioned: `{environment_plan.provisioned}`\n"
            f"- Candidate source: `{report.candidate_source}`\n"
            f"- Baseline available: `{report.baseline_available}`\n"
            f"- Validation status: `{report.status}`\n"
            f"- Comparison: `{report.comparison}`\n"
            f"- Summary: {report.summary}\n"
        )

    def _resolve_sandbox_root(self, *, repo_name: str, run_dir: Path) -> Path:
        configured = self.config.get("paths", {}).get("sandbox_root")
        if not configured:
            return run_dir / "sandbox"
        return Path(configured).expanduser().resolve() / repo_name

    def _build_variants(self, discovery) -> list[dict]:
        variants: list[dict] = []
        if self.llm_synthesizer is not None:
            if self._paper_md_content:
                self.llm_synthesizer.paper_md = self._paper_md_content
            variants.append(self.llm_synthesizer.generate_variant(discovery))
            static_variants = self.facade_synthesizer.generate_variants(discovery)
            if static_variants:
                variants.append(
                    {
                        "code": static_variants[0],
                        "source": "static_template",
                        "metadata": {
                            "fallback_reason": "diagnostic_static_candidate",
                            "variant_count": len(static_variants),
                        },
                    }
                )
            return variants

        if self.llm_required:
            raise RuntimeError("LLM cleaning is required but no LLM client/model was configured.")

        for index, variant in enumerate(self.facade_synthesizer.generate_variants(discovery), 1):
            variants.append(
                {
                    "code": variant,
                    "source": f"static_template_{index}",
                    "metadata": {"variant_index": index},
                }
            )
        return variants

    def _serialize_environment(self, plan: EnvironmentPlan) -> dict:
        return {
            "backend": plan.backend,
            "env_dir": str(plan.env_dir),
            "python_executable": str(plan.python_executable),
            "backend_preference": plan.backend_preference,
            "python_version": plan.python_version,
            "manifest_paths": [str(path) for path in plan.manifest_paths],
            "install_commands": plan.install_commands,
            "optional_install_commands": plan.optional_install_commands,
            "smoke_test_commands": plan.smoke_test_commands,
            "inferred_packages": plan.inferred_packages,
            "smoke_imports": plan.smoke_imports,
            "gpu_mode": plan.gpu_mode,
            "repo_requests_gpu": plan.repo_requests_gpu,
            "gpu_available": plan.gpu_available,
            "gpu_enabled": plan.gpu_enabled,
            "gpu_package": plan.gpu_package,
            "provisioned": plan.provisioned,
            "notes": plan.notes,
        }

    def _serialize_discovery(self, discovery) -> dict:
        return {
            "repo_root": str(discovery.repo_root),
            "entry_module": str(discovery.entry_module),
            "python_files": [str(path) for path in discovery.python_files],
            "readme_files": [str(path) for path in discovery.readme_files],
            "config_files": [str(path) for path in discovery.config_files],
            "test_files": [str(path) for path in discovery.test_files],
            "data_candidates": discovery.data_candidates,
            "main_candidates": discovery.main_candidates,
            "eval_candidates": discovery.eval_candidates,
            "class_candidates": discovery.class_candidates,
            "notes": discovery.notes,
        }

    def _serialize_report(self, report: ValidationReport) -> dict:
        payload = asdict(report)
        return payload

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_code_cleaner_from_config(
    config: dict,
    *,
    skill_manager=None,
    provision_environment: bool = True,
    env_backend: str = "auto",
    python_version: str = "3.10",
    gpu_mode: str = "auto",
    llm_client=None,
    llm_model_name: Optional[str] = None,
) -> CodeCleaner:
    artifact_root = config.get("paths", {}).get("code_cleaner_workdir", "./artifacts/code_cleaner")
    cleaner_cfg = config.get("code_cleaner", {})
    return CodeCleaner(
        artifact_root=artifact_root,
        config=config,
        skill_manager=skill_manager,
        provision_environment=provision_environment,
        env_backend=env_backend,
        python_version=python_version,
        gpu_mode=gpu_mode,
        llm_client=llm_client,
        llm_model_name=llm_model_name,
        llm_required=cleaner_cfg.get("llm_required", False),
        llm_temperature=cleaner_cfg.get("llm_temperature", 0.0),
        llm_max_tokens=cleaner_cfg.get("llm_max_tokens", 16000),
        llm_max_loops=cleaner_cfg.get("llm_max_loops", 3),
        force_rebuild_env=cleaner_cfg.get("force_rebuild_env", False),
    )
