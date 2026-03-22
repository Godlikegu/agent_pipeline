"""
code_cleaner — environment setup, code cleaning, and test harness generation.

Three-phase pipeline:
  1. Environment configuration  (environment.py)
  2. Code cleaning              (cleaner.py, llm_synthesizer.py, facade.py)
  3. Test script generation      (test_harness.py)
"""
from .cleaner import CodeCleaner, build_code_cleaner_from_config
from .models import CleaningResult, EnvironmentPlan, RepoDiscovery, ValidationReport
from .test_harness import (
    generate_test_harness,
    load_data_shapes,
    reset_sandbox,
    run_cmd,
    setup_sandbox,
)

__all__ = [
    "CodeCleaner",
    "CleaningResult",
    "EnvironmentPlan",
    "RepoDiscovery",
    "ValidationReport",
    "build_code_cleaner_from_config",
    "generate_test_harness",
    "load_data_shapes",
    "reset_sandbox",
    "run_cmd",
    "setup_sandbox",
]
