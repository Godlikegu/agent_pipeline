"""
code_cleaner — environment setup, code cleaning, and test harness generation.

Three-phase pipeline:
  1. Environment configuration  (environment.py)
  2. Code cleaning              (cleaner.py, llm_synthesizer.py, facade.py)
  3. Test script generation      (test_harness.py)
"""
from .environment import CondaEnvManager, EnvSetupAgent