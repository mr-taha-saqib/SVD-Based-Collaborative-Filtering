"""
api.py
======
FastAPI inference service for the SVD movie recommender.

Run locally:
    uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000

Swagger UI: http://localhost:8000/docs   OpenAPI: http://localhost:8000/openapi.json

The service loads the APPROVED model from the registry at startup. If none is
approved (or loading fails) it still starts, ``/health`` reports 503 and model
endpoints answer 503, so orchestration can see the problem instead of a crash loop.
"""

from __future__ import annotations

import csv
import hmac
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).parent))

from logging_utils import setup_logging  # noqa: E402
from monitoring import RecommendationMonitor  # noqa: E402
from recommender import ArtifactError, Recommender  # noqa: E402
from registry import ModelRegistry, RegistryError  # noqa: E402
from settings import get, load_config, resolve_path  # noqa: E402

logger = logging.getLogger("api")
API_VERSION = "1.0.0"


# ── schemas ───────────────────────────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"user_id": 1, "n": 5, "exclude_rated": True, "exclude_movie_ids": [356]}})

    user_id: int = Field(..., gt=0, description="MovieLens userId")
    n: int = Field(10, ge=1, le=1000, description="number of recommendations (capped by api.max_n)")
    exclude_rated: bool = Field(True, description="never recommend movies the user already rated")
    exclude_movie_ids: list[int] = Field(default_factory=list, max_length=1000,
                                         description="additional movieIds to exclude")


class RecommendedItem(BaseModel):
    rank: int
    movie_id: int
    title: str | None = Field(None, description="from MovieLens movies.csv; null if unavailable")
    genres: list[str] = Field(default_factory=list)
    score: float = Field(..., description="ranking score (SVD: unclipped predicted rating; "
                                          "fallback: training rating count)")
    predicted_rating: float | None = Field(None, description="clipped predicted rating (SVD only)")


class RecommendationResponse(BaseModel):
    user_id: int
    model_version: str
    strategy: str = Field(..., description="'svd' or 'popularity_fallback' (unknown user)")
    known_user: bool
    cold_start: bool
    n_requested: int
    n_returned: int
    items: list[RecommendedItem]
    unknown_excluded_movie_ids: list[int] = Field(default_factory=list)
    generated_at: str


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"user_id": 1, "movie_id": 50}})
    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)


class PredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float
    method: str = Field(..., description="svd | user_bias_only | item_bias_only | global_mean")
    known_user: bool
    known_movie: bool
    model_version: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"user_id": 1, "movie_id": 50, "rating": 4.5}})
    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)
    rating: float = Field(..., ge=0.5, le=5.0)


class FeedbackResponse(BaseModel):
    accepted: bool
    stored: bool
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    api_version: str = API_VERSION
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    request_id: str | None = None


# ── application state ─────────────────────────────────────────────────────────
class ServiceState:
    def __init__(self, cfg: dict[str, Any], registry_dir: Path, feedback_path: Path | None,
                 monitoring_dir: Path | None):
        self.cfg = cfg
        self.registry = ModelRegistry(registry_dir)
        self.feedback_path = feedback_path
        self.monitoring_dir = monitoring_dir
        self.recommender: Recommender | None = None
        self.monitor: RecommendationMonitor = RecommendationMonitor(None, cfg.get("monitoring", {}), "none")
        self.load_error: str | None = None
        self._fb_lock = threading.Lock()

    def load(self) -> None:
        try:
            rec = Recommender.load(self.registry.approved_dir())
        except (RegistryError, ArtifactError, OSError, ValueError, KeyError) as exc:
            self.recommender = None
            self.load_error = str(exc)
            logger.error("Model not loaded: %s", exc)
            return
        self.recommender = rec
        self.load_error = None
        self.monitor = RecommendationMonitor(rec.reference, self.cfg.get("monitoring", {}),
                                             rec.version, self.monitoring_dir)
        logger.info("Serving model version %s", rec.version)

    def require_model(self) -> Recommender:
        if self.recommender is None:
            raise HTTPException(status_code=503, detail={
                "error": "model_unavailable", "detail": self.load_error or "no model loaded"})
        return self.recommender

    def store_feedback(self, fb: FeedbackRequest) -> bool:
        if self.feedback_path is None:
            return False
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._fb_lock:
            new = not self.feedback_path.exists()
            with open(self.feedback_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["userId", "movieId", "rating", "timestamp"])
                w.writerow([fb.user_id, fb.movie_id, fb.rating, int(time.time())])
        return True


def create_app(cfg: dict[str, Any] | None = None, registry_dir: str | Path | None = None,
               feedback_path: str | Path | None = "default",
               monitoring_dir: str | Path | None = "default") -> FastAPI:
    """Application factory (used by uvicorn and by the tests)."""
    cfg = cfg or load_config()
    reg_dir = Path(registry_dir) if registry_dir else resolve_path(get(cfg, "registry.dir", "models"))
    if feedback_path == "default":
        feedback_path = os.environ.get("SVD_FEEDBACK_PATH") or resolve_path("data/feedback/api_feedback.csv")
    if monitoring_dir == "default":
        monitoring_dir = resolve_path(get(cfg, "monitoring.log_dir", "monitoring"))
    state = ServiceState(cfg, reg_dir, Path(feedback_path) if feedback_path else None,
                         Path(monitoring_dir) if monitoring_dir else None)
    max_n = int(get(cfg, "api.max_n", 100))
    default_n = int(get(cfg, "api.default_n", 10))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.load()
        yield

    app = FastAPI(
        title="SVD Movie Recommendation API",
        version=API_VERSION,
        description=("Top-N movie recommendations from a truncated-SVD collaborative-filtering "
                     "model trained on MovieLens. Unknown users receive a most-popular fallback. "
                     "Predicted ratings are estimates, not guarantees."),
        lifespan=lifespan,
        responses={503: {"model": ErrorResponse, "description": "Model not available"}},
    )
    app.state.service = state

    # ── middleware & error handling ───────────────────────────────────────────
    @app.middleware("http")
    async def timing_and_logging(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error", extra={"request_id": request_id, "path": request.url.path})
            state.monitor.record_error("internal")
            response = JSONResponse(status_code=500, content={
                "error": "internal_error", "detail": "unexpected server error", "request_id": request_id})
        elapsed = time.perf_counter() - t0
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        if endpoint not in ("/metrics",):
            state.monitor.record_request(endpoint, response.status_code, elapsed)
        response.headers["x-request-id"] = request_id
        logger.info("request", extra={"request_id": request_id, "method": request.method,
                                      "endpoint": endpoint, "status": response.status_code,
                                      "latency_ms": round(elapsed * 1000, 3)})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        state.monitor.record_error("validation")
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={
            "error": "validation_error", "detail": errors,
            "request_id": getattr(request.state, "request_id", None)})

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": "http_error", "detail": exc.detail}
        detail.setdefault("request_id", getattr(request.state, "request_id", None))
        return JSONResponse(status_code=exc.status_code, content=detail)

    # ── endpoints ─────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}},
             tags=["operations"])
    def health():
        rec = state.recommender
        if rec is None:
            return JSONResponse(status_code=503, content=HealthResponse(
                status="unavailable", model_loaded=False, detail=state.load_error).model_dump())
        return HealthResponse(status="ok", model_loaded=True, model_version=rec.version)

    @app.get("/model/info", tags=["operations"])
    def model_info() -> dict[str, Any]:
        rec = state.require_model()
        md = rec.metadata
        return {
            "model_version": rec.version,
            "model_name": md.get("model_name"),
            "algorithm": md.get("algorithm"),
            "k": md.get("k"),
            "created_at": md.get("created_at"),
            "status": md.get("status"),
            "n_users": md.get("n_users"), "n_items": md.get("n_items"),
            "serving_policy": md.get("serving_policy"),
            "training_data": md.get("training_data"),
            "metrics": md.get("metrics"),
            "has_movie_metadata": md.get("has_movie_metadata"),
            "mlflow_run_id": md.get("mlflow_run_id"),
            "git_commit": md.get("git_commit"),
            "limits": {"default_n": default_n, "max_n": max_n},
        }

    def _recommend(user_id: int, n: int, exclude_rated: bool, exclude_ids: list[int]) -> RecommendationResponse:
        rec = state.require_model()
        if n > max_n:
            raise HTTPException(status_code=422, detail={
                "error": "validation_error", "detail": f"n must be <= {max_n}"})
        t0 = time.perf_counter()
        res = rec.recommend(user_id, n=n, exclude_rated=exclude_rated, exclude_movie_ids=exclude_ids)
        cols = [rec.item_index[it.movie_id] for it in res.items]
        state.monitor.record_recommendation(user_id, rec.user_activity(user_id), res.strategy, cols, n,
                                            len(res.unknown_excluded_movie_ids), time.perf_counter() - t0)
        return RecommendationResponse(
            user_id=user_id, model_version=rec.version, strategy=res.strategy,
            known_user=res.known_user, cold_start=not res.known_user, n_requested=n,
            n_returned=len(res.items),
            items=[RecommendedItem(rank=i.rank, movie_id=i.movie_id, title=i.title, genres=i.genres,
                                   score=i.score, predicted_rating=i.predicted_rating) for i in res.items],
            unknown_excluded_movie_ids=res.unknown_excluded_movie_ids,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"))

    @app.get("/recommendations/{user_id}", response_model=RecommendationResponse, tags=["recommendations"])
    def recommendations_get(
        user_id: int = PathParam(..., gt=0, description="MovieLens userId"),
        n: int = Query(default_n, ge=1, le=1000, description="number of recommendations"),
        exclude_rated: bool = Query(True),
    ):
        return _recommend(user_id, n, exclude_rated, [])

    @app.post("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
    def recommendations_post(body: RecommendationRequest):
        return _recommend(body.user_id, body.n, body.exclude_rated, body.exclude_movie_ids)

    @app.post("/predict", response_model=PredictionResponse, tags=["recommendations"])
    def predict(body: PredictionRequest):
        rec = state.require_model()
        pred, method = rec.predict_rating(body.user_id, body.movie_id)
        return PredictionResponse(user_id=body.user_id, movie_id=body.movie_id,
                                  predicted_rating=round(pred, 4), method=method,
                                  known_user=rec.is_known_user(body.user_id),
                                  known_movie=rec.is_known_item(body.movie_id), model_version=rec.version)

    @app.post("/feedback", response_model=FeedbackResponse, status_code=202, tags=["monitoring"])
    def feedback(body: FeedbackRequest):
        """Record an observed rating (delayed ground truth) for monitoring and retraining."""
        rec = state.require_model()
        pred, _ = rec.predict_rating(body.user_id, body.movie_id)
        state.monitor.record_feedback(body.user_id, body.movie_id, body.rating, pred)
        stored = state.store_feedback(body)
        return FeedbackResponse(accepted=True, stored=stored, model_version=rec.version)

    @app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
    def metrics():
        state.monitor.summary()   # refresh PSI gauges
        return PlainTextResponse(generate_latest(state.monitor.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/monitoring/summary", tags=["monitoring"])
    def monitoring_summary() -> dict[str, Any]:
        return state.monitor.summary()

    @app.post("/model/reload", tags=["operations"])
    def reload_model(x_admin_token: str | None = Header(default=None)):
        """Reload the approved model (after a promotion). Disabled unless SVD_ADMIN_TOKEN is set."""
        expected = os.environ.get("SVD_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(status_code=404, detail={"error": "not_found",
                                                         "detail": "reload endpoint disabled"})
        if not hmac.compare_digest((x_admin_token or "").encode(), expected.encode()):
            raise HTTPException(status_code=403, detail={"error": "forbidden", "detail": "invalid admin token"})
        state.load()
        if state.recommender is None:
            raise HTTPException(status_code=503, detail={"error": "model_unavailable",
                                                         "detail": state.load_error})
        return {"reloaded": True, "model_version": state.recommender.version}

    return app


def _build_default_app() -> FastAPI:
    cfg = load_config()
    setup_logging(os.environ.get("SVD_LOG_LEVEL", get(cfg, "output.log_level", "INFO")),
                  fmt=os.environ.get("LOG_FORMAT", "json"))
    return create_app(cfg)


# uvicorn api:app --app-dir src
app = _build_default_app() if os.environ.get("SVD_API_NO_AUTOCREATE") != "1" else None

# Silence an unused-import false positive (numpy is used by type checkers in handlers).
_ = np
