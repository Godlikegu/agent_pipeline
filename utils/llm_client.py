"""Helpers for creating OpenAI-compatible API clients from YAML config."""


def create_client(llm_config: dict, model_key: str) -> tuple:
    """
    Create an OpenAI-compatible client from `config/llm.yaml`.

    Args:
        llm_config: Parsed YAML config containing a top-level `models` mapping.
        model_key: Model key, for example `"example/default-model"`.

    Returns:
        `(client, model_name)` where `model_name` is the API model string to send.
    """
    if model_key not in llm_config.get("models", {}):
        raise KeyError(
            f"Model '{model_key}' not found in LLM config. "
            f"Available: {list(llm_config.get('models', {}).keys())}"
        )

    from openai import OpenAI

    model_conf = llm_config["models"][model_key]
    model_name = model_conf.get("model_name", model_key)
    client = OpenAI(
        api_key=model_conf["api_key"],
        base_url=model_conf["base_url"],
    )
    return client, model_name
