# Pipeline 修改建议

**基于**: EVALUATION_REPORT_FULL.md (35.3% pass rate, 18/51 tasks)  
**模型**: Vendor2/Claude-4.6-opus  
**日期**: 2026-04-11

---

## 零、已发现的 Pipeline 设计缺陷（BUG）

### 0.1 STUCK 升级与 max_retries 的边界冲突（高优先级）

**问题来源**: `data/end_sandbox/Vendor2/Claude-4.6-opus/ct_poisson_lowdose_20260411_101929/workflow.log`

**症状**: 任务只运行了 4 次有效迭代就报 FAILED，第 5 次迭代被 STUCK 机制"吞掉"，Planner 升级后根本没机会跑。

**复现日志**:
```
Iteration 1 (Ticket: Planner) → NCC=0.8219 FAIL
Iteration 2 (Ticket: Coder)   → NCC=0.8219 FAIL
Iteration 3 (Ticket: Coder)   → NCC=0.7968 FAIL
Iteration 4 (Ticket: Coder)   → NCC=0.7965 FAIL
Iteration 5 (Ticket: Coder)
  STUCK: Coder 4x. Escalating to Planner.
  Stuck detection overrode Coder → Planner. Skipping Coder this iteration.
FAILED after max retries.   ← Planner 从未执行
```

**根本原因**: `core/workflow.py:149-163` 的 STUCK 升级逻辑：
```python
if (self.failure_history[-1].get("ticket_assigned_to") == "Planner"
        and ticket == "Coder"):
    self._log("  Stuck detection overrode Coder → Planner. Skipping Coder this iteration.")
    self._finalize_round(success=False)
    ticket = "Planner"
    feedback = self.failure_history[-1]
    self._reset_downstream_state("Planner")
    self.failure_history.clear()
    self.retry_count += 1   # ← 问题：消耗了唯一剩余的 retry
    continue                # ← 下一轮 while 条件 5<5 失败，Planner 从未运行
```

**执行序列分析** (max_retries=5):
| 轮次 | retry_count 进入时 | 执行内容 | retry_count 退出时 |
|------|-------------------|---------|-------------------|
| Iter 1 | 0 | Planner→Architect→Coder→Execute→Judge | 1 |
| Iter 2 | 1 | Coder→Execute→Judge | 2 |
| Iter 3 | 2 | Coder→Execute→Judge | 3 |
| Iter 4 | 3 | Coder→Execute→Judge | 4 |
| **Iter 5** | **4** | **STUCK 触发→标记升级到 Planner→retry_count++→continue** | **5** |
| 循环退出 | 5 | `5 < 5 = False`，FAILED | - |

**设计意图 vs 实际效果**:
- 设计意图：Coder 失败 4 次 → 升级到 Planner → Planner 重新规划 → 换算法方案重试
- 实际效果：STUCK 升级动作本身消耗了最后一次 retry 名额，Planner 永远拿不到执行机会，**升级机制失效**

**类似的浪费模式**: `core/workflow.py:204-209` 的语法检查失败升级路径存在同样问题：
```python
if not syntax_ok:
    self._finalize_round(success=False)
    ticket = "Planner"
    self.retry_count += 1  # ← 同样的问题
    self._reset_downstream_state("Planner")
    self.failure_history.clear()
    continue
```

**影响范围**:
- 所有触发 STUCK 升级并在最后一轮发生的任务
- 特别是 `ct_poisson_lowdose`, `xray_tooth_gridrec`, `spectral_snapshot_compressive_imaging` 等"近阈值失败"任务，这些任务的最后一次 Planner 重新规划可能正好能过阈值
- 估计影响 **3-5 个任务** 的最终结果

**修复方案**（按推荐度排序）:

**方案 A（推荐）**: STUCK 升级不消耗 retry 名额
```python
# core/workflow.py:156-163 修改
self._finalize_round(success=False)
ticket = "Planner"
feedback = self.failure_history[-1]
self._reset_downstream_state("Planner")
self.failure_history.clear()
# 删除 self.retry_count += 1
continue
```
**风险**: 理论上可能导致无限 STUCK 循环（但 failure_history 已被清空，实际不会）

**方案 B（稳妥）**: 给 STUCK 升级预留额外 retry
```python
# core/workflow.py:43
while self.retry_count < self.max_retries + (1 if self._stuck_escalated else 0):
```
需要新增 `self._stuck_escalated` 标志位

**方案 C（最小改动）**: 检测到 STUCK 在最后一轮时临时延长 max_retries
```python
# core/workflow.py:152 STUCK 处理内
if self.retry_count + 1 >= self.max_retries:
    self._log("  STUCK on last retry — granting one extra Planner attempt.")
    self.max_retries += 1  # 临时延长
```
**优点**: 改动最小，只影响边界情况
**缺点**: 修改实例状态 `max_retries`，语义不够清晰

**预期收益**: +3-5 个任务通过（pass rate +6-10%）

---

## 一、核心问题分析

### 1.1 失败模式分类

根据 EVALUATION_REPORT_FULL.md 的详细分析，失败任务可分为以下类型：

| 失败类型 | 占比 | 典型任务 | 根本原因 |
|---------|------|---------|---------|
| **算法/物理错误** | ~60% | conventional_ptychography, ct_fan_beam, SSNP_ODT | 模型缺乏领域知识，算法实现不符合物理约束 |
| **参数调优失败** | ~25% | ct_poisson_lowdose, mri_compressed_sensing | 超参数（TV weight, learning rate）选择不当 |
| **代码结构错误** | ~10% | SSNP_ODT (nested class), fourier_ptychography | 代码生成时出现语法/结构性错误 |
| **接口不匹配** | ~5% | 部分任务 | 输入/输出 shape 或数据格式错误 |

### 1.2 成功任务特征

成功的 18 个任务具有以下共性：
- **算法简单直接**：如 FBP、GRAPPA、SENSE 等成熟算法
- **参数鲁棒性高**：对超参数不敏感
- **文档清晰完整**：task_description 中有明确的算法步骤
- **无复杂物理约束**：不涉及波动方程、非线性优化等

---

## 二、短期改进建议（1-2周实施）

### 2.1 增强 Planner Agent 的领域知识注入

**问题**：Planner 生成的计划缺乏领域特定的物理约束和算法细节。

**解决方案**：
```yaml
# config/default.yaml 修改
agents:
  planner:
    temperature: 0.5  # 从 0.7 降低，减少随机性
    max_tokens: 40960  # 从 32768 增加，允许更详细的计划
    max_loops: 5
    # 新增：强制要求 Planner 输出物理约束检查清单
    require_physics_checklist: true
```

**Prompt 增强**（agents/planner.py）：
- 在 system prompt 中添加：
  ```
  For each algorithm step, explicitly state:
  1. Physical constraints (e.g., energy conservation, causality)
  2. Numerical stability requirements (e.g., CFL condition, regularization)
  3. Expected intermediate value ranges
  4. Common failure modes and how to avoid them
  ```

### 2.2 引入参数自适应机制

**问题**：超参数（如 TV weight, learning rate）硬编码，导致收敛失败。

**解决方案**：
1. **在 Architect 阶段生成参数搜索空间**：
   ```python
   # agents/architect.py 新增
   def generate_hyperparameter_grid(self, plan: str) -> dict:
       """Extract hyperparameters from plan and suggest search ranges."""
       # 使用 LLM 识别关键超参数并建议范围
       # 例如：{"tv_weight": [1e-5, 1e-4, 1e-3], "lr": [0.001, 0.01, 0.1]}
   ```

2. **在 Judge 阶段触发参数调优**：
   ```python
   # core/workflow.py 修改
   if metrics["ncc"] < threshold * 0.7:  # 远低于阈值
       if self.retry_count < 3:  # 前3次迭代
           # 触发参数网格搜索
           best_params = self._grid_search_hyperparameters()
           feedback = f"Try these parameters: {best_params}"
   ```

### 2.3 强化 Critic Agent 的物理一致性检查

**问题**：Critic 只做表面检查，未验证物理正确性。

**解决方案**：
```python
# agents/critic.py 新增检查项
PHYSICS_CHECKS = [
    "Does the forward model satisfy energy conservation?",
    "Are boundary conditions properly handled?",
    "Is the adjoint operator correctly implemented?",
    "Are units consistent throughout the computation?",
    "Does the algorithm handle edge cases (zero division, NaN)?",
]

def validate_physics(self, plan: str) -> dict:
    """Run physics-specific validation checks."""
    # 使用 LLM 逐项检查，返回 PASS/FAIL + 建议
```

### 2.4 优化 Code Size Guard

**问题**：`code_size_guard: 12000` 字符限制过于粗糙，导致复杂任务代码被截断。

**解决方案**：
```yaml
# config/default.yaml
pipeline:
  code_size_guard: 20000  # 提高上限
  code_complexity_guard:  # 新增：基于复杂度而非字符数
    max_functions: 15
    max_nesting_depth: 5
    max_lines_per_function: 150
```

---

## 三、中期改进建议（1-2月实施）

### 3.1 引入 Skills 系统（已有框架，需激活）

**当前状态**：
```yaml
skills:
  retrieval_enabled: false  # 关闭
  learning_enabled: false   # 关闭
```

**激活方案**：
1. **Phase 1**：先激活 retrieval，使用预定义 skills
   ```yaml
   skills:
     retrieval_enabled: true
     learning_enabled: false
     retrieval:
       max_token_budget: 15000  # 增加预算
       similarity_threshold: 0.03  # 降低阈值，检索更多
   ```

2. **Phase 2**：运行 10-20 个任务后，激活 learning
   ```yaml
   skills:
     retrieval_enabled: true
     learning_enabled: true
     learning:
       merge_similarity_threshold: 0.50  # 合并相似 skills
       max_skills_per_distillation: 8
   ```

3. **预定义 Skills 库**（手动创建）：
   - `skills/library/drafts/tv_regularization.md`
   - `skills/library/drafts/adjoint_operator_validation.md`
   - `skills/library/drafts/fft_based_propagation.md`
   - `skills/library/drafts/iterative_solver_convergence.md`

### 3.2 增强 Judge Agent 的诊断能力

**问题**：Judge 只报告 NCC/NRMSE，不分析失败原因。

**解决方案**：
```python
# agents/judge.py 新增
def diagnose_failure(self, metrics: dict, output: np.ndarray, gt: np.ndarray) -> str:
    """Perform detailed failure analysis."""
    diagnostics = []
    
    # 1. 检查输出分布
    if np.all(output == 0):
        diagnostics.append("Output is all zeros - likely optimization failed")
    if np.isnan(output).any():
        diagnostics.append("Output contains NaN - numerical instability")
    
    # 2. 频域分析
    output_fft = np.fft.fftn(output)
    gt_fft = np.fft.fftn(gt)
    freq_corr = np.corrcoef(np.abs(output_fft).ravel(), np.abs(gt_fft).ravel())[0,1]
    if freq_corr < 0.3:
        diagnostics.append(f"Frequency domain mismatch (corr={freq_corr:.3f}) - wrong algorithm")
    
    # 3. 动态范围检查
    output_range = output.max() - output.min()
    gt_range = gt.max() - gt.min()
    if abs(output_range / gt_range - 1) > 2:
        diagnostics.append(f"Dynamic range mismatch ({output_range:.2e} vs {gt_range:.2e})")
    
    return " | ".join(diagnostics)
```

### 3.3 实现多阶段验证

**问题**：代码一次性生成并执行，中间步骤无验证。

**解决方案**：
```python
# core/workflow.py 新增
def _validate_intermediate_outputs(self, solver_code: str) -> bool:
    """Insert checkpoints and validate intermediate results."""
    
    # 1. 在 solver_code 中插入 checkpoint
    instrumented_code = self._insert_checkpoints(solver_code)
    
    # 2. 运行并收集中间结果
    checkpoints = self._execute_with_checkpoints(instrumented_code)
    
    # 3. 验证每个 checkpoint
    for name, value in checkpoints.items():
        if not self._validate_checkpoint(name, value):
            return False
    
    return True

def _validate_checkpoint(self, name: str, value: np.ndarray) -> bool:
    """Validate intermediate result."""
    # 检查 NaN, Inf, 动态范围等
    if np.isnan(value).any() or np.isinf(value).any():
        self._log(f"Checkpoint {name} contains NaN/Inf")
        return False
    return True
```

---

## 四、长期改进建议（3-6月实施）

### 4.1 引入 Test-Driven Development (TDD) 模式

**方案**：
1. **在 Planner 阶段生成单元测试**：
   ```python
   # agents/planner.py
   def generate_unit_tests(self, plan: str) -> str:
       """Generate pytest tests for each algorithm component."""
       # 例如：test_forward_operator(), test_adjoint_operator()
   ```

2. **在 Coder 阶段先通过测试**：
   ```python
   # core/workflow.py
   while not all_tests_pass:
       solver_code = self.coder.generate(...)
       test_results = self._run_unit_tests(solver_code)
       if test_results["failed"]:
           feedback = f"Tests failed: {test_results['failed']}"
   ```

### 4.2 实现分层代码生成

**问题**：一次性生成完整代码容易出错。

**方案**：
```
Iteration 1: 生成 forward model + 简单测试
Iteration 2: 生成 adjoint operator + 验证
Iteration 3: 生成 optimization loop
Iteration 4: 添加 regularization
Iteration 5: 参数调优
```

### 4.3 引入外部验证工具

**方案**：
- **静态分析**：使用 `pylint`, `mypy` 检查代码质量
- **性能分析**：使用 `line_profiler` 识别瓶颈
- **数值验证**：使用 `scipy.optimize.check_grad` 验证梯度

---

## 五、配置文件修改建议

### 5.1 立即修改（config/default.yaml）

```yaml
pipeline:
  max_retries: 7  # 从 5 增加到 7
  execution_timeout: 3600  # 从 1800 增加到 3600（1小时）
  code_size_guard: 20000  # 从 12000 增加
  max_history_len: 5  # 从 3 增加到 5

agents:
  planner:
    temperature: 0.5  # 从 0.7 降低
    max_tokens: 40960  # 从 32768 增加
  
  coder:
    temperature: 0.1  # 从 0.2 降低，提高确定性
    max_retries: 5  # 从 3 增加
  
  judge:
    temperature: 0.3  # 从 0.7 降低，提高诊断准确性
    enable_detailed_diagnosis: true  # 新增

skills:
  retrieval_enabled: true  # 激活
  learning_enabled: false  # 暂不激活
  retrieval:
    max_token_budget: 15000
    similarity_threshold: 0.03
    max_items: 15
```

### 5.2 新增配置项

```yaml
# config/default.yaml 新增
validation:
  enable_intermediate_checks: true
  enable_unit_tests: false  # 长期目标
  enable_gradient_checks: true
  
hyperparameter_tuning:
  enable_auto_tuning: true
  max_grid_search_iterations: 3
  common_params:
    tv_weight: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
    num_iterations: [100, 200, 500]
```

---

## 六、预期效果

### 6.1 短期改进（1-2周）
- **Pass rate**: 35% → **45-50%**
- **主要提升领域**：CT, MRI（参数敏感任务）
- **实施成本**：低（主要是配置和 prompt 修改）

### 6.2 中期改进（1-2月）
- **Pass rate**: 50% → **60-65%**
- **主要提升领域**：Ptychography, ODT（需要领域知识）
- **实施成本**：中（需要开发 skills 库和诊断工具）

### 6.3 长期改进（3-6月）
- **Pass rate**: 65% → **75-80%**
- **主要提升领域**：全领域
- **实施成本**：高（需要重构 workflow，引入 TDD）

---

## 七、优先级排序

| 优先级 | 改进项 | 预期提升 | 实施难度 | 建议时间 |
|-------|--------|---------|---------|---------|
| **P0** | 激活 skills retrieval | +10% | 低 | 立即 |
| **P0** | 降低 Planner/Coder temperature | +5% | 低 | 立即 |
| **P0** | 增加 code_size_guard | +3% | 低 | 立即 |
| **P1** | 参数自适应机制 | +8% | 中 | 1周 |
| **P1** | 强化 Critic 物理检查 | +5% | 中 | 1周 |
| **P1** | Judge 详细诊断 | +4% | 中 | 2周 |
| **P2** | 中间结果验证 | +6% | 中 | 1月 |
| **P2** | 预定义 skills 库 | +8% | 高 | 1月 |
| **P3** | TDD 模式 | +10% | 高 | 3月 |
| **P3** | 分层代码生成 | +8% | 高 | 3月 |

---

## 八、立即可执行的修改

### 修改 1: config/default.yaml
```bash
# 备份
cp config/default.yaml config/default.yaml.bak

# 修改关键参数
sed -i 's/max_retries: 5/max_retries: 7/' config/default.yaml
sed -i 's/execution_timeout: 1800/execution_timeout: 3600/' config/default.yaml
sed -i 's/code_size_guard: 12000/code_size_guard: 20000/' config/default.yaml
sed -i 's/temperature: 0.7  # planner/temperature: 0.5  # planner/' config/default.yaml
sed -i 's/temperature: 0.2  # coder/temperature: 0.1  # coder/' config/default.yaml
sed -i 's/retrieval_enabled: false/retrieval_enabled: true/' config/default.yaml
```

### 修改 2: 创建基础 skills
```bash
mkdir -p skills/library/drafts

# 创建 TV regularization skill
cat > skills/library/drafts/tv_regularization.md << 'EOF'
---
name: TV Regularization Best Practices
description: Guidelines for Total Variation regularization in inverse problems
type: feedback
---

When implementing TV regularization:
1. Weight range: Start with 1e-4 to 1e-3 for most imaging tasks
2. Use anisotropic TV for preserving edges: sum(abs(grad_x) + abs(grad_y))
3. Smooth TV (Huber) is more stable: sqrt(grad^2 + epsilon^2) with epsilon=1e-8
4. Scale TV weight by data fidelity term magnitude
5. For 3D: TV weight should be ~10x smaller than 2D

**Why**: Improper TV weight causes either over-smoothing (too large) or no regularization (too small).

**How to apply**: In optimization loops with TV terms, always start with weight=1e-4 and adjust based on residual norm.
EOF
```

---

## 九、监控指标

建议在后续运行中跟踪以下指标：

1. **Per-domain pass rate**：识别哪些领域改进最明显
2. **Average iterations to success**：衡量收敛速度
3. **Failure mode distribution**：跟踪失败类型变化
4. **Code quality metrics**：
   - Average code size
   - Syntax error rate
   - Import error rate
5. **Agent performance**：
   - Planner rejection rate (by Critic)
   - Coder retry count
   - Judge diagnosis accuracy

---

**总结**：当前 pipeline 的主要瓶颈是**缺乏领域知识**和**参数调优能力**。通过激活 skills 系统、降低 temperature、增强诊断能力，预计可在 1-2 周内将 pass rate 从 35% 提升至 45-50%。
