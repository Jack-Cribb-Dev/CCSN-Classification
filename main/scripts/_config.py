from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _expand_env_vars(value: Any) -> Any:
    """
    Expand ${VAR_NAME} occurrences in strings.

    This keeps configs portable across machines (cluster/local).
    """
    if isinstance(value, str):
        # Simple ${VAR} expansion; leave unknown vars as-is.
        if "${" in value:
            for key in set(part[2:-1] for part in value.split("${")[1:]):
                env_val = os.environ.get(key)
                if env_val is not None:
                    value = value.replace("${" + key + "}", env_val)
        return value
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_config(config_path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _expand_env_vars(data)

