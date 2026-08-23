"""Registry-backed common configuration and model construction."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .bit import BitCausalLM, BitConfig
from .common import AmarkenCausalLM, ModelConfig
from .dt import DTCausalLM, DTConfig
from .glimmer import GlimmerCausalLM, GlimmerConfig

MODEL_REGISTRY = {
    DTConfig.model_type: (DTConfig, DTCausalLM),
    BitConfig.model_type: (BitConfig, BitCausalLM),
    GlimmerConfig.model_type: (GlimmerConfig, GlimmerCausalLM),
}


def create_config(model_type: str, **overrides: Any) -> ModelConfig:
    """Build a validated architecture config by stable model identifier."""
    try:
        config_type = MODEL_REGISTRY[model_type][0]
    except KeyError as error:
        raise ValueError(
            f"unknown model_type {model_type!r}; choose {sorted(MODEL_REGISTRY)}"
        ) from error
    return config_type(**overrides)


def create_model(config: ModelConfig) -> AmarkenCausalLM:
    """Build the registered model matching a config; reject accidental duck types."""
    try:
        config_type, model_type = MODEL_REGISTRY[config.model_type]
    except KeyError as error:
        raise ValueError(f"unregistered model_type {config.model_type!r}") from error
    if not isinstance(config, config_type):
        raise TypeError(f"{config.model_type!r} requires {config_type.__name__}")
    return model_type(config)


def save_config(config: ModelConfig, path: str | Path) -> None:
    """Atomically persist constructor fields in a versioned human-readable manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_type": config.model_type,
        "config": asdict(config),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def load_config(path: str | Path) -> ModelConfig:
    """Load and validate a versioned config manifest through the registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported config format")
    return create_config(payload["model_type"], **payload["config"])
