## 项目简介：Agent 化逆问题求解 Pipeline

本项目是一个面向「物理逆问题 / 科学计算」的自动化求解流水线。  
整体流程由多个智能 Agent（Planner / Architect / Coder / Judge 等）配合完成，从任务描述与 GT 代码出发，自动生成数据、编写求解器、运行评估脚本并给出最终指标。

---

## 环境配置

### 1. 系统与基础依赖

- **操作系统**：Linux（推荐，其他系统需自行适配路径）
- **Python**：推荐 `Python 3.10`（与当前 `agent` 环境一致）
- **Conda**：建议使用 conda 管理环境（`miniconda` / `anaconda` 均可）

### 2. 创建并激活 Conda 环境

```bash
conda create -n agent python=3.10 -y
conda activate agent
```

### 3. 安装 Python 依赖

项目依赖主要包括：

- **openai**：统一的 OpenAI 兼容客户端（可接 Anthropic / DeepSeek / Kimi / Qwen 等）
- **yaml 相关库**：`pyyaml`
- **科学计算**：`numpy`（用于数据加载 / 评估脚本）

如果你已有自己的 `requirements.txt`，可按需调整：

```bash
pip install openai pyyaml numpy scikit-learn pandas tabulate
pip install sentence-transformers scipy scikit-image
```

> 如任务本身（GT 代码 / solver）依赖额外库（如 `torch`、`scipy` 等），请自行在当前环境中额外安装。

### 4. LLM 配置 (`config/llm.yaml`)

`config/llm.yaml` 中配置了可用的大模型列表，每个条目代表一个可选模型，例如：

```yaml
models:
  "cds/Claude-4.6-opus":
    api_type: "openai"
    base_url: "https://ai-gateway-internal.dp.tech/v1"
    api_key: "YOUR_API_KEY"
    temperature: 0.7
```

- **models**：模型字典，Key 为命令行参数 `--model` 的取值，同样也作为model_name存在。
- **api_type**：API 类型（目前代码只用到了 OpenAI 兼容风格）。
- **base_url**：对应模型的 OpenAI 兼容接口地址。
- **api_key**：访问该接口所需密钥。
- **temperature**：采样温度，越大越发散，越小越稳定。

> 注意：仓库中示例 key / url 仅为示例，请根据你实际的网关 / 账号信息进行替换。

### 5. Pipeline 主配置 (`config/default.yaml`)

关键字段说明（常用的几项）：

- **paths**
  - `task_descriptions_dir`：任务描述 markdown 存放目录。
  - `sandbox_root`：所有任务运行时的 sandbox 根目录，workflow 会在此目录下为每个任务创建独立子目录。
- **pipeline**
  - `max_retries`：主循环最大重试次数。
  - `execution_timeout`：`solver.py` 执行超时时间（秒）。
  - `data_gen_timeout`：数据生成脚本 `data_gen.py` 超时。
  - `syntax_check_timeout`：语法检查（`py_compile`）超时。
- **evaluation**
  - `min_guaranteed_psnr`：PSNR 最低保证阈值。
  - `baseline_ratio`：基线 PSNR 的比例系数，最终阈值为 `max(baseline_psnr * ratio, min_guaranteed_psnr)`。
- **skills**
  - `enabled`：是否启用技能系统（知识库）。
  - `mode`：技能注入模式。
  - `retrieval` / `credit` / `embedding`：控制知识检索与打分的详细参数。

---

## 任务配置说明 (`config/tasks/*.yaml`)

项目中的任务列表由 `config/tasks/*.yaml` 描述，例如 `debug_tasks.yaml`：

```yaml
tasks:
  - gt_code_path: /path/to/sim_code.py
    name: sim
    python_path: /path/to/conda/envs/sim/bin/python

  - gt_code_path: /path/to/bpm_code.py
    name: bpm
    python_path: /path/to/conda/envs/ragas/bin/python
```

字段含义：

- **gt_code_path**：该任务的 GT（Ground Truth）代码路径，可为单文件或目录。
- **name**：任务名称，用于筛选与日志标识（如 `--task-filter sim`）。
- **python_path**：执行该任务时使用的 Python 解释器路径（例如另一 conda 环境）。  
  - 如果留空，代码会回退为当前运行 `run_task.py` 的 Python。

`train_tasks.yaml` / `test_tasks.yaml` 的结构与此类似，只是具体任务列表不同。

---

## 运行方式与参数说明

### 1. 直接运行单次调试（推荐）

最常用的入口是：

```bash
cd /your/pipeline/path   # 替换为你的仓库路径

python -m run_task \
  --task-config config/tasks/debug_tasks.yaml \
  --llm-config config/llm.yaml \
  --model "cds/Claude-4.6-opus" \
  --task-filter sim
```

#### `run_task.py` 主要参数

- `**--config**`（可选）  
  - 说明：覆盖默认配置路径（否则使用 `config/default.yaml`）。  
  - 示例：`--config /path/to/custom_config.yaml`。
- `**--task-config**`  
  - 说明：任务列表配置文件，默认 `config/tasks/debug_tasks.yaml`。  
  - 示例：`--task-config config/tasks/train_tasks.yaml`。
- `**--llm-config**`  
  - 说明：LLM 配置文件路径，默认 `config/llm.yaml`。
- `**--model**`  
  - 说明：在 `llm.yaml` 的 `models` 字典中选择一个 key。  
  - 示例：`--model "cds/Claude-4.6-opus"`。
- `**--task-filter**`  
  - 说明：按任务 `name` 进行筛选，逗号分隔。  
  - 示例：`--task-filter sim,bpm` 只运行名为 `sim` 和 `bpm` 的任务。  
  - 若不指定，则运行该 `task-config` 中的全部任务。

### 2. 使用脚本批量运行 (`run.sh`)

仓库根目录提供了一个简单脚本：

```bash
#! /bin/bash
python -m run_task --task-filter sim > reports/log/run_task_sim.log 2>&1 &
```

- 含义：
  - 在后台运行 `python -m run_task --task-filter sim`；
  - 将标准输出和标准错误一起重定向到 `reports/log/run_task_sim.log`；
  - `&` 表示在后台运行，不阻塞当前终端。
- 使用方式：

```bash
chmod +x run.sh        # 首次运行可能需要赋予执行权限
bash run.sh            # 或 ./run.sh
tail -f reports/log/run_task_sim.log   # 实时查看日志
```

> 注意：`run.sh` 默认使用的是当前环境的 `python`，如果你希望指定特定环境，可在脚本中将 `python` 换成完整路径，例如 `/home/xxx/miniconda3/envs/agent/bin/python`。

---

## 日志与结果查看

每个任务在运行时会生成以下几类输出：

- **workflow 日志**  
  - 位置：`${sandbox_root}/${model_name}/${exp_id}/workflow.log`  
  - 内容：Agent 决策过程、阶段状态、执行 / 评估日志（包含 solver 的 stdout / stderr）。
- **阶段快照与中间产物**  
  - 如：`iter_xxx_solver.py`、`iter_xxx_exec_log.txt`、`iter_xxx_*.json` 等。
  - 用于回溯每一步生成的代码和评估结果。
- **整体报告**  
  - 由 `ExecutionReporter` 在 `reports/` 目录下生成汇总报告（任务成功 / 失败统计、指标等）。

---

## 常见问题（FAQ）

- **Q：为什么 `nvidia-smi` 看不到任务？**  
A：当前 pipeline 默认生成的是 CPU 代码（主要依赖 `numpy` 等），只有当 GT / 生成的 `solver.py` 主动使用 `torch.cuda` / `to("cuda")` 等 GPU API 时，进程才会出现在 `nvidia-smi` 中。
- **Q：如何修改为自己的任务？**  
A：在 `config/tasks/*.yaml` 中新增条目，指定自己的 `gt_code_path` 与 `python_path`，并在 `task_descriptions_dir` 下为对应 `name` 提供一份 `<name>_description.md` 即可。
- **Q：如何切换不同大模型？**  
A：在 `config/llm.yaml` 中新增 / 修改某个模型配置，然后在命令行通过 `--model` 选择对应 key。

