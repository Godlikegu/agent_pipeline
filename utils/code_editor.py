import ast
import textwrap

class CodeEditor:
    @staticmethod
    def replace_function(source_code: str, func_name: str, new_code: str) -> str:
        """
        在 source_code 中找到名为 func_name 的函数，并将其替换为 new_code。
        
        :param source_code: 原始的完整 Python 代码
        :param func_name: 需要被替换/填充的函数名
        :param new_code: Agent 生成的新函数代码（包含 def func_name...）
        :return: 修改后的完整代码
        """
        try:
            # 1. 解析原始代码为 AST
            tree = ast.parse(source_code)
            
            # 2. 寻找目标函数节点
            target_node = None
            parent_class = None # 记录是否在类内部（用于判断缩进）
            
            # 遍历寻找目标函数
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # 如果在类里，继续找
                    for sub_node in node.body:
                        if isinstance(sub_node, ast.FunctionDef) and sub_node.name == func_name:
                            target_node = sub_node
                            parent_class = node
                            break
                elif isinstance(node, ast.FunctionDef) and node.name == func_name:
                    target_node = node
                    break
            
            if not target_node:
                # 如果找不到函数（可能是 Architect 没生成，或者是全新添加），
                # 策略：简单的 append 到文件末尾，或者报错。
                # 这里为了稳健，如果找不到，我们尝试直接返回 new_code 拼接到最后（视情况而定），
                # 但在你的框架里，Skeleton 应该已经存在，所以报错更安全。
                print(f"[CodeEditor] Warning: Function '{func_name}' not found in source. Appending to end.")
                return source_code + "\n\n" + new_code

            # 3. 获取源代码的行列表
            lines = source_code.splitlines(keepends=True)
            
            # AST 行号从 1 开始，list 索引从 0 开始
            # start_line 是 def xxx(...) 的那一行
            start_index = target_node.lineno - 1
            # end_line 是函数结束的那一行（包括装饰器等，end_lineno 是 Python 3.8+ 特性）
            end_index = target_node.end_lineno
            
            # 4. 缩进处理 (关键步骤)
            # 获取原始代码该函数的缩进级别
            original_indent = ""
            first_line = lines[start_index]
            original_indent = first_line[:len(first_line) - len(first_line.lstrip())]
            
            # 处理新代码的缩进
            # Pre-process: expand tabs to avoid mixed indentation issues
            new_code = new_code.expandtabs(4)
            
            # Validate new_code syntax strictly before insertion
            try:
                # We wrap in a dummy class if it's indented, but usually it's a def.
                # Just parsing it should reveal indentation errors if self-consistent.
                ast.parse(textwrap.dedent(new_code))
            except SyntaxError as e:
                raise ValueError(f"Generated code snippet has SyntaxError: {e.msg}")

            # Agent 生成的代码通常是顶格写的 (no indentation) 或者有它自己的缩进
            # 我们先清除新代码的所有公共缩进，然后加上原始代码的缩进
            dedented_new_code = textwrap.dedent(new_code).strip()
            
            # [FIX] Handle mixed indentation where decorator is unindented but def is indented
            # Example:
            # @staticmethod
            #     def foo():
            lines_new = dedented_new_code.splitlines()
            def_line_idx = -1
            for i, line in enumerate(lines_new):
                if line.lstrip().startswith('def ') or line.lstrip().startswith('async def '):
                    def_line_idx = i
                    break
            
            if def_line_idx > 0:
                def_line = lines_new[def_line_idx]
                if def_line.startswith(' ') or def_line.startswith('\t'):
                    # Check if decorators before it are top-level
                    decorators = lines_new[:def_line_idx]
                    if all(l.strip().startswith('@') and not l.startswith(' ') for l in decorators if l.strip()):
                        # Re-dedent starting from def line
                        body = "\n".join(lines_new[def_line_idx:])
                        body_dedented = textwrap.dedent(body)
                        # Reassemble
                        dedented_new_code = "\n".join(decorators) + "\n" + body_dedented
            
            indented_new_code = textwrap.indent(dedented_new_code, original_indent)
            
            # 5. 替换
            # 我们保留 start_index 之前的内容，插入新代码，保留 end_index 之后的内容
            
            # 构造新内容
            new_lines = [line + "\n" for line in indented_new_code.splitlines()]
            
            # 拼接：前段 + 新段 + 后段
            final_lines = lines[:start_index] + new_lines + lines[end_index:]
            
            result = "".join(final_lines)
            
            # [Validation] Ensure the result is valid Python
            try:
                ast.parse(result)
            except SyntaxError as e:
                # Attempt to provide more detail
                raise ValueError(f"Merged code resulted in SyntaxError at line {e.lineno}: {e.msg}")
                
            return result

        except Exception as e:
            # Re-raise explicit ValueErrors (validation failures) so caller knows it's a code issue
            if isinstance(e, ValueError):
                raise e
            print(f"[CodeEditor] Error modifying code: {e}")
            # For other unexpected errors (e.g. AST parse of source failed), return source
            return source_code
    
    @staticmethod
    def replace_imports(source_code: str, new_imports: str) -> str:
        """
        专门用于替换文件的 Import 部分。
        原理：找到源代码中最后一个 Import 语句的位置，将之前的内容全部替换。
        """
        try:
            tree = ast.parse(source_code)
            last_import_idx = -1
            
            # 找到最后一个 import 或 from ... import ...
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    last_import_idx = i
                elif last_import_idx != -1:
                    # 如果已经找到了 import，且当前节点不再是 import，说明 import 区域结束
                    break
            
            if last_import_idx == -1:
                # 原文件没有 import，直接插在最前面
                return new_imports + "\n\n" + source_code
            
            # 获取最后一个 import 节点的结束行号
            last_node = tree.body[last_import_idx]
            end_lineno = last_node.end_lineno
            
            lines = source_code.splitlines(keepends=True)
            # 替换 0 到 end_lineno 的内容
            # 注意：保留 end_lineno 之后的所有内容
            remaining_code = "".join(lines[end_lineno:])
            
            return new_imports.strip() + "\n\n" + remaining_code.strip()
            
        except Exception as e:
            print(f"[CodeEditor] Import replace failed: {e}")
            return source_code

    @staticmethod
    def replace_main_block(source_code: str, new_main_block: str) -> str:
        """
        替换 if __name__ == "__main__": 模块
        """
        try:
            tree = ast.parse(source_code)
            main_node = None
            
            # 寻找 if __name__ == "__main__":
            for node in tree.body:
                if isinstance(node, ast.If):
                    # 检查条件是否是 __name__ == "__main__"
                    try:
                        if (isinstance(node.test, ast.Compare) and 
                            isinstance(node.test.left, ast.Name) and 
                            node.test.left.id == "__name__" and 
                            isinstance(node.test.comparators[0], ast.Constant) and 
                            node.test.comparators[0].value == "__main__"):
                            main_node = node
                            break
                    except:
                        continue
            
            if not main_node:
                # 如果没有 main block，直接追加到文件末尾
                return source_code.strip() + "\n\n" + new_main_block
            
            # 替换逻辑
            lines = source_code.splitlines(keepends=True)
            start_index = main_node.lineno - 1
            
            # main block 通常是文件的最后部分，直接替换 start_index 之后的所有内容即可
            # 这样比较安全，防止遗漏
            return "".join(lines[:start_index]) + "\n" + new_main_block
            
        except Exception as e:
            print(f"[CodeEditor] Main block replace failed: {e}")
            return source_code

    @staticmethod
    def replace_class(source_code: str, class_name: str, new_code: str) -> str:
        """
        替换整个 Class 定义。
        """
        try:
            tree = ast.parse(source_code)
            target_node = None
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    target_node = node
                    break
            
            if not target_node:
                # 没找到类，插到 import 后面或者文件前面
                # 简单起见，插在 import 之后， solver 之前
                return source_code.replace("class InverseSolver", new_code + "\n\nclass InverseSolver")

            # 替换逻辑完全同 replace_function
            lines = source_code.splitlines(keepends=True)
            start_index = target_node.lineno - 1
            end_index = target_node.end_lineno
            
            new_lines = [line + "\n" for line in new_code.splitlines()]
            final_lines = lines[:start_index] + new_lines + lines[end_index:]
            return "".join(final_lines)
            
        except Exception as e:
            print(f"[CodeEditor] Class replace failed: {e}")
            return source_code