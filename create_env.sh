#!/usr/bin/env bash

python -m code_cleaner env-setup --tasks-dir data/tasks --envs-dir ./test_conda_envs --model Vendor2/Claude-4.6-opus --output-yaml config/tasks/all_tasks.yaml
