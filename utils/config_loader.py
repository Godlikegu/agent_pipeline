"""
utils/config_loader.py — 配置加载工具

加载 config/default.yaml，支持通过 override_path 覆盖部分配置。
提供 cfg() 辅助函数用于点号路径取值。
"""
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def load_config(override_path: str = None) -> dict:
    """
    加载统一配置文件，可选地与覆盖配置深度合并。

    Args:
        override_path: 自定义 YAML 配置路径，其值覆盖 default.yaml 中的对应项。
                       可用于不同环境（开发/生产）的配置切换。

    Returns:
        合并后的配置字典。
    """
    import yaml

    with open(_DEFAULT_CONFIG_PATH, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if override_path:
        with open(override_path, encoding='utf-8') as f:
            custom = yaml.safe_load(f) or {}
        _deep_merge(config, custom)

    return config


def cfg(config: dict, dotted_key: str, default: Any = None) -> Any:
    """
    通过点号路径从配置字典中取值。

    Args:
        config: 配置字典。
        dotted_key: 点号分隔的键路径，例如 'skills.retrieval.max_items'。
        default: 键不存在时返回的默认值。

    Returns:
        对应的配置值，或 default。

    Example:
        >>> config = load_config()
        >>> cfg(config, 'pipeline.max_retries', 5)
        5
    """
    val = config
    for k in dotted_key.split('.'):
        if not isinstance(val, dict):
            return default
        val = val.get(k)
        if val is None:
            return default
    return val


def _deep_merge(base: dict, override: dict) -> dict:
    """递归地将 override 合并入 base（就地修改 base）。"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
