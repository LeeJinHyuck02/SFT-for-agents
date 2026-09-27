"""YAML 설정 로드와 경로 해석. 비밀값(HF_TOKEN)은 환경변수에서만 읽는다."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

AGENTS = ("generation", "structure")

try:  # 로컬 편의용. Pod에서는 환경변수로 직접 주입하므로 python-dotenv가 없어도 된다
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass


def root_dir() -> Path:
    """UGRP_ROOT가 없으면 이 파일 기준 저장소 루트(agents/)."""
    env = os.environ.get("UGRP_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """UGRP_DATA_DIR → {root}/data (Pod 번들) → {root}/../data (로컬 작업 트리) 순으로 찾는다."""
    env = os.environ.get("UGRP_DATA_DIR")
    if env:
        return Path(env).resolve()
    bundled = root_dir() / "data"
    return bundled if bundled.is_dir() else (root_dir().parent / "data").resolve()


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(agent: str) -> Dict[str, Any]:
    if agent not in AGENTS:
        raise ValueError(f"agent는 {AGENTS} 중 하나여야 합니다: {agent}")
    config_dir = root_dir() / "configs"
    with (config_dir / "base.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with (config_dir / f"{agent}.yaml").open(encoding="utf-8") as f:
        config = _merge(config, yaml.safe_load(f) or {})
    config["cache_dir"] = str(root_dir() / config["cache_dir"])
    return config


def hf_token() -> Optional[str]:
    """선택 사항. google/gemma-4-31B-it는 토큰 없이 받을 수 있다.

    있으면 Hub의 rate limit이 올라가 대용량(가중치 ~62GB) 다운로드가 안정적이고, --push-to-hub에는 필수다.
    """
    return os.environ.get("HF_TOKEN") or None


def output_dir(agent: str) -> Path:
    return root_dir() / "outputs" / f"{agent}_think"
