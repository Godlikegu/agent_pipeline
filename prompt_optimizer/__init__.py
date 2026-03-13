"""
prompt_optimizer — Prompt Optimization via textGrad (Reserved for future implementation)

预期功能：以 code_cleaner 从源代码生成的 task_description 为 ground truth，
使用 textGrad 对 task_gen 模块中的 user prompt 进行优化训练。

预期接口：
    class PromptOptimizer:
        def train(self, user_prompts: list[str], gt_descriptions: list[str]) -> str:
            '''使用 textGrad 优化 prompt 模板，返回优化后的模板字符串'''
            ...

        def evaluate(self, prompt_template: str, test_cases: list[dict]) -> dict:
            '''评估 prompt 模板的效果，返回 metrics dict'''
            ...

依赖：textGrad 库
"""
