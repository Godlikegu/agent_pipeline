#!/bin/bash

# Run all 7 models x all tasks
python -m tests.test_plan_architect --temperature 0.5

# Single model + single task (dry run)
python -m tests.test_plan_architect --task-filter ct_fan_beam --model-filter gemini-3.1-pro-preview --temperature 0.5

# Custom output dir
python -m tests.test_plan_architect --output-dir my_benchmark --temperature 0.5


python -m tests.test_plan_architect --model-filter Vendor2/GLM-5


