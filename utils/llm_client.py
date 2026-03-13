"""
utils/llm_client.py — LLM 客户端工厂

统一封装 OpenAI 兼容客户端的创建逻辑，
支持任何与 OpenAI API 兼容的后端（Anthropic、DeepSeek 等）。
"""
from openai import OpenAI


def create_client(llm_config: dict, model_key: str) -> tuple:
    """
    从 LLM 配置文件中创建 OpenAI 兼容客户端。

    Args:
        llm_config: 从 config/llm.yaml 加载的配置字典（含 models 字典）。
        model_key: 模型键名，例如 "cds/Claude-4.6-opus"。

    Returns:
        (client, model_name) 元组：
            client: OpenAI 兼容的客户端实例。
            model_name: 模型名称字符串，用于 API 请求。

    Raises:
        KeyError: model_key 在配置中不存在时。

    Example:
        >>> llm_cfg = load_config_yaml("config/llm.yaml")
        >>> client, model_name = create_client(llm_cfg, "cds/Claude-4.6-opus")
    """
    if model_key not in llm_config.get('models', {}):
        raise KeyError(
            f"Model '{model_key}' not found in LLM config. "
            f"Available: {list(llm_config.get('models', {}).keys())}"
        )

    model_conf = llm_config['models'][model_key]
    # 若配置未显式指定 model_name，则用 model_key 作为 API 请求的模型名
    model_name = model_conf.get('model_name', model_key)
    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url']
    )
    return client, model_name
