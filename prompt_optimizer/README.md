# Prompt Optimizer

使用 TextGrad 优化 task generator 的系统提示词。

## 运行方式

```bash
cd /home/guyuxuan/pipeline
./prompt_optimizer/run_optimization.sh \
  --paper-dir <paper_markdown目录> \
  --ground-truth-dir <ground_truth目录> \
  --models "模型key1,模型key2" \
  --optimizer-model <评估/反向传播用模型key>
```

### 环境变量（可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| PAPER_DIR | ./data/paper_markdown | 论文 markdown 目录 |
| GROUND_TRUTH_DIR | ./data/ground_truth_descriptions | 标准答案 task_description 目录 |
| MODELS | cds/Claude-4.6-opus | 生成用模型（逗号分隔） |
| OPTIMIZER_MODEL | cds/Claude-4.6-opus | 评估/反向传播用模型 |
| BATCH_SIZE | 1 | 批大小 |
| EPOCHS | 5 | 训练轮数 |

### 示例

```bash
./prompt_optimizer/run_optimization.sh \
  --paper-dir ./data/paper_markdown \
  --ground-truth-dir ./data/ground_truth_descriptions \
  --models "cds/Claude-4.6-opus" \
  --optimizer-model "cds/Claude-4.6-opus" \
  --epochs 3
```

---

## ground-truth-dir 与 paper-dir 的文件对应关系

### 匹配规则

1. **paper-dir**：递归收集所有 `.md` 文件，以 **文件名（不含扩展名）** 作为 key 建立映射。
2. **ground-truth-dir**：递归收集所有 `.md` 文件，对每个文件做 `normalize_task_name` 得到 task_name，再用该 task_name 去 paper 映射中查找。

### `normalize_task_name` 规则

从文件名（不含扩展名）中移除以下后缀之一（若存在）：

- `_description`
- `-description`
- `.description`

### 对应示例

| paper-dir 中的文件 | ground-truth-dir 中的文件 | 说明 |
|--------------------|---------------------------|------|
| `sim.md` | `sim_description.md` | gt 的 stem 为 `sim_description`，normalize 后为 `sim`，匹配 `sim.md` |
| `sim.md` | `sim-description.md` | 同上 |
| `sim.md` | `sim.md` | gt 的 stem 为 `sim`，无后缀，直接匹配 |
| `bpm.md` | `bpm_description.md` | 匹配 |
| `foo/bar/sim.md` | `x/y/sim_description.md` | 支持子目录，只看文件名 stem |

### 不匹配情况

- paper-dir 中没有与 normalize 后的 task_name 同名的 `.md` 时，该 ground-truth 文件会被跳过，并打印 warning。
