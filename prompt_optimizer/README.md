# Prompt Optimizer

This module uses TextGrad to optimize the task-generator system prompt against a set of paper markdown files and ground-truth task descriptions.

## Run

```bash
./prompt_optimizer/run_optimization.sh \
  --paper-dir ./data/paper_markdown \
  --ground-truth-dir ./data/ground_truth_descriptions \
  --models "example/default-model" \
  --optimizer-model "example/default-model"
```

## Environment Variables

| Variable | Default |
| --- | --- |
| `PAPER_DIR` | `./data/paper_markdown` |
| `GROUND_TRUTH_DIR` | `./data/ground_truth_descriptions` |
| `MODELS` | `example/default-model` |
| `OPTIMIZER_MODEL` | `example/default-model` |
| `BATCH_SIZE` | `1` |
| `EPOCHS` | `20` |

## File Matching

- Markdown files are collected recursively from `paper-dir`.
- Ground-truth descriptions are collected recursively from `ground-truth-dir`.
- A ground-truth file is matched to a paper file after stripping one of these suffixes from the ground-truth stem:
  - `_description`
  - `-description`
  - `.description`
- Example:
  - `paper-dir/sim.md` matches `ground-truth-dir/sim_description.md`
  - `paper-dir/sim.md` matches `ground-truth-dir/sim.md`
