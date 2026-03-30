#!/usr/bin/env bash


python -m code_cleaner env-setup \\
        --tasks-dir data/tasks \\
        --envs-dir ./test_conda_envs \\
        --task-filter fourier_ptychography \\
        --model cds/Claude-4.6-opus \\
        --output-yaml config/tasks/auto_tasks.yaml



python -m code_cleaner env-setup --tasks-dir data/tasks --envs-dir ./test_conda_envs --task-filter fourier_ptychography --model cds/Claude-4.6-opus --output-yaml config/tasks/auto_tasks.yaml