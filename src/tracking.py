"""
tracking.py
===========
Thin MLflow wrapper used by the training pipeline and the retraining workflow.

* Default backend: a local SQLite file (``sqlite:///mlflow.db``) plus a local
  artifact folder; no server and no credentials. ``MLFLOW_TRACKING_URI`` points it
  at a remote tracking server instead.
* If tracking is disabled in config, or MLflow is not installed, every call is a
  no-op and a WARNING is logged once. Nothing fails silently.
* Model registration in the MLflow Model Registry is attempted only for
  database-backed URIs (sqlite/postgres/mysql/http). The serving decision itself
  lives in ``registry.py``; MLflow aliases mirror it for visibility.
"""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from settings import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


def git_commit() -> str:
    """Current git commit (or ``$GIT_COMMIT``, or 'unknown')."""
    env = os.environ.get("GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _normalise_uri(uri: str) -> str:
    """Make relative sqlite / file URIs relative to the project root."""
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        rel = uri[len("sqlite:///"):]
        if not Path(rel).is_absolute():
            return "sqlite:///" + str(resolve_path(rel))
    if "://" not in uri:
        return resolve_path(uri).as_uri()
    return uri


class Tracker:
    """Experiment tracker. Use :meth:`run` as a context manager."""

    def __init__(self, cfg: dict[str, Any]):
        mcfg = cfg.get("mlflow", {})
        self.enabled = bool(mcfg.get("enabled", True))
        self.experiment = str(mcfg.get("experiment", "experiment"))
        self.register_models = bool(mcfg.get("register_models", False))
        self.model_name = str(cfg.get("registry", {}).get("model_name", self.experiment))
        self.uri = _normalise_uri(str(mcfg.get("tracking_uri", "sqlite:///mlflow.db")))
        self.artifact_location = resolve_path(mcfg.get("artifact_location", "mlartifacts"))
        self._mlflow = None
        self.run_id: str | None = None
        if not self.enabled:
            logger.warning("MLflow tracking is DISABLED by configuration; runs will not be recorded.")
            return
        try:
            import mlflow  # noqa: PLC0415
        except ImportError:
            logger.warning("MLflow is not installed; runs will not be recorded. "
                           "Install requirements.txt to enable tracking.")
            self.enabled = False
            return
        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.uri)
        if mlflow.get_experiment_by_name(self.experiment) is None:
            loc = None
            if self.uri.startswith("sqlite:"):
                self.artifact_location.mkdir(parents=True, exist_ok=True)
                loc = self.artifact_location.as_uri()
            mlflow.create_experiment(self.experiment, artifact_location=loc)
        mlflow.set_experiment(self.experiment)
        logger.info("MLflow tracking -> %s (experiment %s)", self.uri, self.experiment)

    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[Tracker]:
        if not self.enabled:
            yield self
            return
        mlflow = self._mlflow
        all_tags = {"git_commit": git_commit()}
        all_tags.update(tags or {})
        with mlflow.start_run(run_name=run_name, tags=all_tags) as active:
            self.run_id = active.info.run_id
            yield self
        logger.info("MLflow run %s finished (%s)", self.run_id, run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            self._mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, directory: str | Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(directory).exists():
            self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def register_model(self, version_tag: str) -> str | None:
        """Register this run's ``model`` artifact in the MLflow Model Registry.

        Returns the MLflow model version, or None when registration is disabled
        or not supported by the backend (a WARNING explains why).
        """
        if not (self.enabled and self.register_models and self.run_id):
            return None
        if self.uri.startswith("file:"):
            logger.warning("MLflow Model Registry needs a database-backed tracking URI; "
                           "skipping registration for %s", self.uri)
            return None
        # MLflow 3's register_model() requires an MLflow "logged model"; our artifact
        # is a plain folder (numpy arrays + JSON), so the version is created directly
        # from the run's artifact URI.
        client = self._mlflow.MlflowClient()
        try:
            client.get_registered_model(self.model_name)
        except self._mlflow.exceptions.MlflowException:
            client.create_registered_model(self.model_name, description="SVD movie recommender")
        source = f"{self._mlflow.get_run(self.run_id).info.artifact_uri}/model"
        mv = client.create_model_version(self.model_name, source=source, run_id=self.run_id,
                                         tags={"registry_version": version_tag})
        logger.info("MLflow registry: %s version %s registered", self.model_name, mv.version)
        return str(mv.version)

    def set_alias(self, alias: str, mlflow_version: str | None) -> None:
        """Point an MLflow alias (e.g. 'champion') at a registered model version."""
        if not (self.enabled and mlflow_version):
            return
        client = self._mlflow.MlflowClient()
        client.set_registered_model_alias(self.model_name, alias, mlflow_version)
        logger.info("MLflow registry: alias '%s' -> %s version %s", alias,
                    self.model_name, mlflow_version)
