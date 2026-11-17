"""FastAPI service: schemas, status codes, validation, cold start, monitoring endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import create_app


@pytest.fixture
def client(trained, cfg, tmp_path):
    app = create_app(cfg, feedback_path=tmp_path / "fb.csv", monitoring_dir=tmp_path / "mon")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def empty_client(cfg, tmp_path):
    app = create_app(cfg, registry_dir=tmp_path / "empty_registry", feedback_path=None, monitoring_dir=None)
    with TestClient(app) as c:
        yield c


def test_health_and_info(client, trained):
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["model_loaded"] is True
    assert h.json()["model_version"] == trained["registry"]["version_id"]
    info = client.get("/model/info").json()
    assert info["k"] == trained["selection"]["selected_k"]
    assert info["metrics"]["validation"]["partition"] == "validation"
    assert info["limits"]["max_n"] == 100


def test_get_recommendations_valid(client, ratings):
    r = client.get("/recommendations/1", params={"n": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "svd" and body["known_user"] and not body["cold_start"]
    assert body["n_returned"] == 5 and len(body["items"]) == 5
    rated = set(ratings.loc[ratings.userId == 1, "movieId"])
    assert not {i["movie_id"] for i in body["items"]} & rated
    assert set(body["items"][0]) == {"rank", "movie_id", "title", "genres", "score", "predicted_rating"}


def test_post_recommendations_with_exclusions(client):
    first = client.get("/recommendations/1", params={"n": 1}).json()["items"][0]["movie_id"]
    r = client.post("/recommendations", json={"user_id": 1, "n": 3, "exclude_movie_ids": [first, 424242]})
    assert r.status_code == 200
    assert first not in [i["movie_id"] for i in r.json()["items"]]
    assert r.json()["unknown_excluded_movie_ids"] == [424242]


def test_unknown_user_cold_start(client):
    r = client.get("/recommendations/987654", params={"n": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "popularity_fallback" and body["cold_start"] and not body["known_user"]
    assert len(body["items"]) == 4


@pytest.mark.parametrize("path,params", [
    ("/recommendations/0", {}), ("/recommendations/-5", {}), ("/recommendations/abc", {}),
    ("/recommendations/1", {"n": 0}), ("/recommendations/1", {"n": "ten"}),
])
def test_invalid_get_requests_422(client, path, params):
    r = client.get(path, params=params)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_n_above_configured_max_rejected(client):
    r = client.get("/recommendations/1", params={"n": 101})
    assert r.status_code == 422


@pytest.mark.parametrize("payload", [
    {}, {"n": 5}, {"user_id": "x"}, {"user_id": 1, "n": -1},
    {"user_id": 1, "unexpected": True}, {"user_id": 1, "exclude_movie_ids": "nope"},
])
def test_invalid_post_bodies_422(client, payload):
    assert client.post("/recommendations", json=payload).status_code == 422


def test_malformed_json_422(client):
    r = client.post("/recommendations", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_predict_known_and_unknown_movie(client):
    ok = client.post("/predict", json={"user_id": 1, "movie_id": 100}).json()
    assert 0.5 <= ok["predicted_rating"] <= 5.0
    unk = client.post("/predict", json={"user_id": 1, "movie_id": 99999999}).json()
    assert unk["known_movie"] is False and unk["method"] == "user_bias_only"


def test_feedback_stored_and_monitored(client, tmp_path):
    r = client.post("/feedback", json={"user_id": 1, "movie_id": 100, "rating": 4.5})
    assert r.status_code == 202 and r.json()["stored"]
    assert (tmp_path / "fb.csv").read_text().splitlines()[0] == "userId,movieId,rating,timestamp"
    assert client.post("/feedback", json={"user_id": 1, "movie_id": 100, "rating": 7}).status_code == 422
    assert client.get("/monitoring/summary").json()["totals"]["feedback"] == 1


def test_metrics_and_summary(client):
    client.get("/recommendations/1")
    client.get("/recommendations/987654")
    client.get("/recommendations/0")
    m = client.get("/metrics")
    assert m.status_code == 200 and "recsys_requests_total" in m.text
    assert 'recsys_recommendations_total{strategy="popularity_fallback"} 1.0' in m.text
    s = client.get("/monitoring/summary").json()
    assert s["rates"]["cold_start_fallback"] == 0.5
    assert s["data_quality"]["validation_errors"] == 1
    assert s["observed_performance"]["status"] == "insufficient_labels"


def test_model_unavailable_returns_503(empty_client):
    h = empty_client.get("/health")
    assert h.status_code == 503 and h.json()["model_loaded"] is False
    for r in (empty_client.get("/recommendations/1"), empty_client.get("/model/info"),
              empty_client.post("/predict", json={"user_id": 1, "movie_id": 1})):
        assert r.status_code == 503 and r.json()["error"] == "model_unavailable"


def test_reload_endpoint_protection(client, monkeypatch):
    assert client.post("/model/reload").status_code == 404
    monkeypatch.setenv("SVD_ADMIN_TOKEN", "s3cret")
    assert client.post("/model/reload", headers={"x-admin-token": "wrong"}).status_code == 403
    assert client.post("/model/reload").status_code == 403                      # header missing
    assert client.post("/model/reload", headers={"x-admin-token": "s3cret-"}).status_code == 403
    ok = client.post("/model/reload", headers={"x-admin-token": "s3cret"})
    assert ok.status_code == 200 and ok.json()["reloaded"]


def test_openapi_schema_documents_endpoints(client):
    spec = client.get("/openapi.json").json()
    for p in ("/health", "/model/info", "/recommendations/{user_id}", "/recommendations",
              "/metrics", "/monitoring/summary", "/predict", "/feedback"):
        assert p in spec["paths"]


def test_request_id_header(client):
    r = client.get("/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
