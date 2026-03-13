"""
task_gen — Task Description Generation Module (Reserved for future implementation)

预期功能：从论文 PDF + 用户 prompt 自动生成 task_description。
生成的 description 作为 core/workflow.py 的输入（替代从文件读取）。

预期接口：
    class TaskDescriptionGenerator:
        def generate(self, paper_path: str, user_prompt: str) -> str:
            '''从论文和用户 prompt 生成结构化的 task description'''
            ...

数据流：
    paper.pdf + user_prompt
        → TaskDescriptionGenerator.generate()
        → task_description (str)
        → InverseProblemWorkflow(task_desc=task_description, ...)
"""
