"""
settings.py
===========
Configuration loading for every entry point (pipeline, retraining, API, tools).

* All values live in ``config.yaml`` at the project root; nothing is hard-coded
  in application code.
* Paths in the YAML are interpreted relative to the project root, never to the
  current working directory, so commands work from any directory.
* A small set of environment variables override the YAML for deployment
  (Docker, CI). They are documented in ``.env.example``.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

# env var -> (section, key, caster)
_ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "SVD_DATA_DIR": ("dataset", "local_dir", str),
    "SVD_RATINGS_PATH": ("dataset", "ratings_path", str),
    "SVD_MODEL_DIR": ("registry", "dir", str),
    "SVD_MONITORING_DIR": ("monitoring", "log_dir", str),
    "SVD_LOG_LEVEL": ("output", "log_level", str),
    "SVD_OUTPUT_DIR": ("output", "corrected_dir", str),
    "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri", str),
    "SVD_MLFLOW_ARTIFACTS": ("mlflow", "artifact_location", str),
    "SVD_MLFLOW_ENABLED": ("mlflow", "enabled", lambda s: s.strip().lower() in {"1", "true", "yes"}),
    "SVD_API_MAX_N": ("api", "max_n", int),
    "SVD_RETRAINING_REPORT_DIR": ("retraining", "report_dir", str),
}


class ConfigError(ValueError):
    """Raised when the configuration is missing or invalid."""


def load_config(path: str | Path | None = None, apply_env: bool = True) -> dict[str, Any]:
    """Load ``config.yaml`` (or ``$SVD_CONFIG``) and apply environment overrides.

    Args:
        path: Explicit config path. Defaults to ``$SVD_CONFIG`` or the project
            root ``config.yaml``.
        apply_env: Apply the documented environment-variable overrides.

    Returns:
        A plain nested dict.

    Raises:
        ConfigError: If the file does not exist or is not a YAML mapping.
    """
    cfg_path = Path(path or os.environ.get("SVD_CONFIG", DEFAULT_CONFIG_PATH))
    if not cfg_path.exists():
        raise ConfigError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {cfg_path} must contain a YAML mapping.")
    cfg = copy.deepcopy(data)
    if apply_env:
        for env, (section, key, caster) in _ENV_OVERRIDES.items():
            raw = os.environ.get(env)
            if raw is None or raw == "":
                continue
            try:
                cfg.setdefault(section, {})[key] = caster(raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"Invalid value for {env}={raw!r}: {exc}") from exc
            logger.debug("Config override from %s -> %s.%s", env, section, key)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail fast on values that would silently corrupt an experiment."""
    split = cfg.get("split", {})
    test_frac = float(split.get("test_fraction", 0.2))
    val_frac = float(split.get("validation_fraction", 0.1))
    if not 0.0 < test_frac < 1.0:
        raise ConfigError(f"split.test_fraction must be in (0, 1), got {test_frac}")
    if not 0.0 < val_frac < 1.0:
        raise ConfigError(f"split.validation_fraction must be in (0, 1), got {val_frac}")
    if test_frac + val_frac >= 0.9:
        raise ConfigError("split.test_fraction + split.validation_fraction must be < 0.9")
    ks = cfg.get("model", {}).get("k_values", [])
    if not ks or any(int(k) <= 0 for k in ks):
        raise ConfigError("model.k_values must be a non-empty list of positive integers")
    rank = cfg.get("ranking", {})
    if int(rank.get("k", 10)) <= 0:
        raise ConfigError("ranking.k must be positive")


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths unchanged)."""
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``section.key`` from the nested config with a default."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
