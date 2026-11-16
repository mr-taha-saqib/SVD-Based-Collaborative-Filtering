"""Data loading, validation, fingerprinting, configuration and the 3-way split."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from conftest import make_ratings

from data_loader import (
    DataValidationError,
    _safe_extract,
    dataset_fingerprint,
    ensure_dataset,
    load_movies,
    sha256_file,
    train_val_test_split,
    validate_ratings,
    verify_no_leakage,
)
from settings import ConfigError, load_config, validate_config


class TestValidation:
    def test_missing_column_raises(self):
        with pytest.raises(DataValidationError, match="missing required columns"):
            validate_ratings(pd.DataFrame({"userId": [1], "movieId": [2]}))

    def test_missing_values_raise(self):
        df = pd.DataFrame({"userId": [1, None], "movieId": [2, 3], "rating": [4.0, 3.0]})
        with pytest.raises(DataValidationError) as exc:
            validate_ratings(df)
        assert exc.value.report.n_missing_values == 1

    def test_out_of_range_and_bad_ids_dropped_when_requested(self):
        df = pd.DataFrame({"userId": [1, 2, -3, 4, 5.5], "movieId": [1, 2, 3, 4, 5],
                           "rating": [4.0, 9.0, 3.0, 0.0, 3.0]})
        clean, rep = validate_ratings(df, drop_invalid=True)
        assert clean["userId"].tolist() == [1]
        assert rep.n_out_of_range == 2 and rep.n_non_integer_ids == 2
        assert rep.passed and rep.warnings

    def test_duplicates_keep_latest(self):
        df = pd.DataFrame({"userId": [1, 1, 2], "movieId": [5, 5, 5], "rating": [2.0, 4.5, 3.0],
                           "timestamp": [10, 20, 5]})
        clean, rep = validate_ratings(df)
        assert rep.n_duplicate_pairs == 1
        assert clean.loc[(clean.userId == 1) & (clean.movieId == 5), "rating"].item() == 4.5
        assert len(clean) == 2

    def test_types_normalised(self):
        df = pd.DataFrame({"userId": ["1"], "movieId": ["7"], "rating": ["3.5"]})
        clean, _ = validate_ratings(df)
        assert clean.dtypes["userId"] == np.int64 and clean.dtypes["rating"] == np.float64

    def test_off_step_rating_is_a_warning_not_an_error(self):
        clean, rep = validate_ratings(pd.DataFrame({"userId": [1], "movieId": [1], "rating": [3.3]}))
        assert rep.n_off_scale_step == 1 and len(clean) == 1


class TestFingerprint:
    def test_order_independent_and_content_sensitive(self, ratings):
        a = dataset_fingerprint(ratings)
        assert a == dataset_fingerprint(ratings.sample(frac=1, random_state=3))
        changed = ratings.copy()
        changed.loc[0, "rating"] = 5.0 if changed.loc[0, "rating"] != 5.0 else 4.0
        assert dataset_fingerprint(changed) != a


class TestThreeWaySplit:
    def test_partitions_disjoint_and_complete(self, ratings):
        sp = train_val_test_split(ratings, 0.1, 0.2, 42)
        assert sp.manifest["leakage_checks"]["passed"]
        assert len(sp.train) + len(sp.validation) + len(sp.test) == len(ratings)
        allpairs = pd.concat([sp.train, sp.validation, sp.test])[["userId", "movieId"]]
        assert not allpairs.duplicated().any()

    def test_every_user_in_all_partitions(self, ratings):
        sp = train_val_test_split(ratings, 0.1, 0.2, 42)
        users = set(ratings.userId)
        assert set(sp.train.userId) == set(sp.validation.userId) == set(sp.test.userId) == users

    def test_fractions_respected(self):
        df = make_ratings(n_users=60, n_items=200, density=0.4)
        sp = train_val_test_split(df, 0.1, 0.2, 1)
        assert abs(len(sp.test) / len(df) - 0.2) < 0.02
        assert abs(len(sp.validation) / len(df) - 0.1) < 0.02

    def test_deterministic_and_row_order_invariant(self, ratings):
        a = train_val_test_split(ratings, 0.1, 0.2, 7)
        b = train_val_test_split(ratings.sample(frac=1, random_state=11), 0.1, 0.2, 7)
        assert a.manifest["split_sha256"] == b.manifest["split_sha256"]
        c = train_val_test_split(ratings, 0.1, 0.2, 8)
        assert a.manifest["split_sha256"] != c.manifest["split_sha256"]

    def test_tiny_users_stay_in_train(self):
        df = pd.DataFrame({"userId": [1, 1, 2, 2, 2, 2, 2], "movieId": [1, 2, 1, 2, 3, 4, 5],
                           "rating": [3.0] * 7})
        sp = train_val_test_split(df, 0.2, 0.2, 0)
        assert set(sp.train.loc[sp.train.userId == 1, "movieId"]) == {1, 2}
        assert sp.manifest["users_kept_in_train_only"] == 1

    def test_rejects_duplicates_and_bad_fractions(self, ratings):
        with pytest.raises(ValueError, match="duplicate"):
            train_val_test_split(pd.concat([ratings, ratings.head(1)]), 0.1, 0.2)
        with pytest.raises(ValueError):
            train_val_test_split(ratings, 0.5, 0.5)

    def test_leakage_detector_detects_injected_overlap(self, ratings):
        sp = train_val_test_split(ratings, 0.1, 0.2, 42)
        leaky_test = pd.concat([sp.test, sp.train.head(3)])
        res = verify_no_leakage(sp.train, sp.validation, leaky_test)
        assert res["train_test_shared_pairs"] == 3 and not res["passed"]


class TestDatasetFiles:
    def _zip_bytes(self, members: dict[str, str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, text in members.items():
                z.writestr(name, text)
        return buf.getvalue()

    def test_ensure_dataset_uses_local_archive_and_checks_checksum(self, tmp_path, cfg):
        zip_path = tmp_path / "data" / "ml.zip"
        zip_path.parent.mkdir()
        zip_path.write_bytes(self._zip_bytes({"ml/ratings.csv": "userId,movieId,rating\n1,1,4.0\n"}))
        cfg["dataset"].update({"local_dir": str(tmp_path / "data"), "zip_filename": "ml.zip",
                               "ratings_path": str(tmp_path / "data" / "ml" / "ratings.csv"),
                               "zip_sha256": sha256_file(zip_path), "url": "https://invalid.example/none"})
        assert ensure_dataset(cfg).exists()
        cfg["dataset"]["ratings_path"] = str(tmp_path / "data" / "ml" / "other.csv")
        cfg["dataset"]["zip_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="checksum"):
            ensure_dataset(cfg)

    def test_safe_extract_rejects_path_traversal(self, tmp_path):
        buf = io.BytesIO(self._zip_bytes({"../evil.txt": "x"}))
        with zipfile.ZipFile(buf) as z, pytest.raises(ValueError, match="Unsafe"):
            _safe_extract(z, tmp_path / "dest")

    def test_load_movies_missing_file_is_empty(self, tmp_path):
        assert load_movies(tmp_path / "nope.csv").empty


class TestConfig:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SVD_MODEL_DIR", "/tmp/elsewhere")
        monkeypatch.setenv("SVD_MLFLOW_ENABLED", "false")
        cfg = load_config()
        assert cfg["registry"]["dir"] == "/tmp/elsewhere" and cfg["mlflow"]["enabled"] is False

    def test_invalid_values_rejected(self, cfg):
        bad = dict(cfg, split={"test_fraction": 0.6, "validation_fraction": 0.4})
        with pytest.raises(ConfigError):
            validate_config(bad)
        with pytest.raises(ConfigError):
            validate_config(dict(cfg, model={"k_values": [0, 5]}))

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "missing.yaml")
