"""Model registry: registration, explicit approval, rejection, rollback."""

from __future__ import annotations

import json

import pytest

from registry import ModelRegistry, RegistryError


def _model_dir(tmp_path, name, content="x"):
    d = tmp_path / f"src_{name}"
    d.mkdir()
    (d / "model.npz").write_text(content)
    return d


def test_register_does_not_approve(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    reg.register(_model_dir(tmp_path, "a"), "v1", {"content_sha256": "a"})
    assert reg.exists("v1") and reg.approved_version() is None
    with pytest.raises(RegistryError):
        reg.approved_dir()


def test_approve_requires_reason_and_known_version(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    reg.register(_model_dir(tmp_path, "a"), "v1", {"content_sha256": "a"})
    with pytest.raises(RegistryError, match="reason"):
        reg.approve("v1", {})
    with pytest.raises(RegistryError, match="unknown"):
        reg.approve("v9", {"reason": "x"})


def test_promotion_keeps_previous_and_rollback_restores(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    reg.register(_model_dir(tmp_path, "a"), "v1", {"content_sha256": "a"})
    reg.approve("v1", {"reason": "initial"})
    reg.register(_model_dir(tmp_path, "b", "y"), "v2", {"content_sha256": "b"})
    reg.approve("v2", {"reason": "better"})
    assert reg.approved_version() == "v2"
    assert reg.exists("v1") and reg.metadata("v1")["status"] == "superseded"
    assert reg.rollback("bad") == "v1"
    assert reg.approved_version() == "v1"
    assert reg.metadata("v2")["status"] == "rolled_back"
    events = [e["event"] for e in reg.state()["events"]]
    assert events == ["registered", "approved", "registered", "approved", "rollback"]
    with pytest.raises(RegistryError):
        reg.rollback()


def test_rejection_recorded_and_not_served(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    reg.register(_model_dir(tmp_path, "a"), "v1", {"content_sha256": "a"})
    reg.approve("v1", {"reason": "initial"})
    reg.register(_model_dir(tmp_path, "b", "y"), "v2", {"content_sha256": "b"})
    reg.record_rejection("v2", {"reason": "gate failed"})
    assert reg.approved_version() == "v1"
    assert reg.metadata("v2")["status"] == "rejected"


def test_register_is_idempotent_for_identical_content(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    reg.register(_model_dir(tmp_path, "a"), "v1", {"content_sha256": "a"})
    reg.register(_model_dir(tmp_path, "a2"), "v1", {"content_sha256": "a"})
    with pytest.raises(RegistryError, match="different content"):
        reg.register(_model_dir(tmp_path, "a3"), "v1", {"content_sha256": "zzz"})
    assert reg.list_versions() == ["v1"]


@pytest.mark.parametrize("bad", ["../x", "a/b", ".hidden", ""])
def test_invalid_version_ids_rejected(tmp_path, bad):
    with pytest.raises(RegistryError):
        ModelRegistry(tmp_path).version_dir(bad)


def test_registry_file_is_valid_json_after_writes(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    reg.register(_model_dir(tmp_path, "a"), "v1", {"content_sha256": "a"})
    reg.approve("v1", {"reason": "initial"})
    data = json.loads((tmp_path / "reg" / "registry.json").read_text())
    assert data["approved"] == "v1"
    assert not (tmp_path / "reg" / ".registry.lock").exists()
