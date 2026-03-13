import abc
from typing import List, Dict, Optional, Any
import time

def is_generation_complete(finish_reason, content):
    # 1. 明确需要继续的唯一情况
    if finish_reason == "length":
        print(f"[DEBUG] finish reason: {finish_reason} -> False (truncated)")
        return False

    # 2. 明确异常中断
    if finish_reason == "content_filter":
        print(f"[ERROR] Generation blocked by content filter.")
        raise RuntimeError("Generation blocked by content filter.")

    if finish_reason is None:
        print(f"[WARN] Finish reason is None (Connection interrupted or timed out). Attempting continuation...")
        # 内容存在时尝试续写，比直接报错更鲁棒
        return False if content and content.strip() else True

    # 3. 正常结束：有实质性内容
    if content and content.strip():
        print(f"[DEBUG] finish reason: {finish_reason} -> True (complete)")
        return True

    # 4. 非 length 但没内容 → 异常情况
    print(f"[WARN] Unexpected finish_reason '{finish_reason}' with empty content. Treating as complete.")
    return True

def get_full_response(client, model, messages, max_loops=5, max_tokens=32768, temperature=0.7, **kwargs):
    """
    通用自动续写函数（非流式）。
    支持 OpenAI/DeepSeek/Writer 等兼容 OpenAI API 的模型。
    """
    full_content = ""
    current_messages = list(messages)
    
    for loop_i in range(max_loops):
        print(f"\n[INFO] Loop {loop_i+1}/{max_loops}: Generating with {model}...")
        
        # 避免无限循环：每次续写减少 max_tokens
        adjusted_max_tokens = max(max_tokens // (loop_i + 1), 512)
        
        try:
            # 构建请求参数
            request_kwargs = {
                "model": model,
                "messages": current_messages,
                "max_tokens": adjusted_max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            # 针对 DeepSeek 启用 thinking（如果未指定）
            if "deepseek" in model.lower() and "extra_body" not in request_kwargs:
                request_kwargs["extra_body"] = {"response_format": {"type": "text"}}
            
            # 调用 API（带简单重试机制）
            for retry in range(3):
                try:
                    response = client.chat.completions.create(**request_kwargs)
                    break
                except Exception as e:
                    if retry == 2:
                        raise
                    print(f"[WARN] Retry {retry+1}/3 after error: {e}")
                    time.sleep(1 << retry)  # 指数退避
            
            # 解析响应
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason
            
            # --- THINKING CAPTURE (NEW) ---
            # Try to capture 'thinking' field from extra_fields if model supports it (e.g. Claude extended)
            # Or if it's mixed in content.
            # Standard OpenAI object might not have it, but let's check 'reasoning_content' (DeepSeek R1)
            # or just rely on explicit prompt instruction for standard models.
            
            if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                print(f"\n[Thinking Process Captured]:\n{choice.message.reasoning_content[:500]}...\n")
            
            # 累积内容
            if content:
                full_content += content
            
            # 检查是否完成
            if is_generation_complete(finish_reason, content):
                print(f"[INFO] Generation complete after {loop_i+1} loop(s). Total chars: {len(full_content)}")
                return full_content
            
            # 准备续写提示
            print(f"[INFO] [Truncated] Output stopped at {len(full_content)} chars. Requesting continuation...")
            current_messages.append({"role": "assistant", "content": content})
            current_messages.append({
                "role": "user",
                "content": (
                    "CONTINUATION REQUEST: Continue EXACTLY from where you left off.\n"
                    "STRICT RULES:\n"
                    "- DO NOT repeat any previous content\n"
                    "- DO NOT add introductions/conclusions\n"
                    "- DO NOT reprint function/class headers\n"
                    "- Output ONLY the continuation of the previous response"
                )
            })
            
        except Exception as e:
            print(f"[ERROR] Error during API call in loop {loop_i+1}: {e}")
            if loop_i == 0:
                raise RuntimeError(f"Initial generation failed: {e}")
            else:
                print(f"[WARN] Continuation failed; returning partial content")
                break
    
    print(f"[WARN] Max continuation loops ({max_loops}) reached. Returning partial content.")
    return full_content

class BaseAgent(abc.ABC):
    """
    所有 Agent 的基类。
    负责管理 System Prompt、LLM 客户端，并定义统一的 interact 接口。
    """
    def __init__(self, client: Any, model_name: str = "gpt-4-turbo", temperature: float = 0.7):
        """
        初始化 Agent
        
        Args:
            client: 兼容 OpenAI API 的客户端实例（如 openai.OpenAI, deepseek.OpenAI 等）
            model_name: 模型名称
            temperature: 生成温度（默认 0.7）
        """
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = self._build_system_prompt()
        self._validate_client()

    def _validate_client(self):
        """验证客户端是否兼容 OpenAI API"""
        if not hasattr(self.client, 'chat') or not hasattr(self.client.chat, 'completions'):
            raise ValueError(
                "Client must be OpenAI-compatible (have .chat.completions.create method). "
                "Examples: openai.OpenAI(), deepseek.OpenAI(), etc."
            )

    @abc.abstractmethod
    def _build_system_prompt(self) -> str:
        """子类必须实现此方法以定义其核心角色"""
        pass

    def call_llm(
        self, 
        user_prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: int = 32768,
        max_loops: int = 5,
        **kwargs
    ) -> str:
        """
        调用 LLM API 并自动处理截断续写
        
        Args:
            user_prompt: 用户输入
            temperature: 生成温度（覆盖实例默认值）
            max_tokens: 单次请求最大 token 数
            max_loops: 最大续写循环次数
            **kwargs: 透传给 API 的其他参数（如 top_p, presence_penalty 等）
        
        Returns:
            完整的 LLM 响应文本
        """
        # 使用实例温度或传入的温度
        temp = temperature if temperature is not None else self.temperature
        
        # 构建消息历史
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[INFO] [{self.__class__.__name__}] Calling LLM ({self.model_name}) with temperature={temp}")
        
        # 调用带自动续写的生成函数
        return get_full_response(
            client=self.client,
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            max_loops=max_loops,
            temperature=temp,
            **kwargs
        )

    def generate(self, context: Dict[str, Any]) -> str:
        """标准调用入口，context 包含任务所需的具体信息"""
        user_prompt = self._build_user_prompt(context)
        return self.call_llm(user_prompt)

    @abc.abstractmethod
    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        """根据上下文构建当次请求的 User Prompt"""
        pass