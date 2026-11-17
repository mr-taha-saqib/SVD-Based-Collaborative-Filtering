"""
movietweetings.py
=================
Loader for the MovieTweetings dataset (Dooms et al., RecSys/CrowdRec 2013), used as a
SEPARATE, alternative data source next to MovieLens. Nothing here changes the MovieLens
pipeline or its results.

Source and licence (verified 2026-11-17)
  * Official source: https://github.com/sidooms/MovieTweetings, files under ``latest/`` at
    the pinned commit below (the dataset's last update, 2021-11-24).
  * Licence: MIT (repository LICENSE file, "Copyright (c) 2013 Simon Dooms"); the copyright
    and permission notice must accompany copies. The MIT grant covers the dataset author's
    rights only. The ratings were extracted from public tweets (X/Twitter) and movie metadata
    refers to IMDb, so third-party terms and data-protection law still need a legal review
    before commercial use (see docs/DATASET_LICENSING.md).

Privacy
  ``users.dat`` maps dataset user ids to real Twitter account ids. It is deliberately NEVER
  downloaded or read; only ``ratings.dat`` (dataset user id, IMDb id, rating, timestamp) and
  ``movies.dat`` (IMDb id, title, genres) are used.

Preparation (fixed before any model was fitted; not tuned on validation or test)
  * ratings of 0 are dropped (the documented scale is 1-10; 0 is outside it);
  * ratings are divided by 2, so 1..10 maps exactly onto 0.5..5.0 in 0.5 steps and the
    existing validation, clipping, API schema and relevance threshold (4.0 = 8/10) apply
    unchanged; reported errors are therefore on the 0.5-5 scale (multiply by 2 for the
    10-point scale);
  * iterative 20-core filter (every user and every movie keeps >= 20 ratings), matching the
    MovieLens convention of >= 20 ratings per user and keeping the dense SVD within memory.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from data_loader import download_archive, sha256_file

logger = logging.getLogger(__name__)

REPOSITORY = "https://github.com/sidooms/MovieTweetings"
COMMIT = "44c525d0c766944910686c60697203cda39305d6"
RAW_BASE = f"https://raw.githubusercontent.com/sidooms/MovieTweetings/{COMMIT}"
# SHA-256 of the files at COMMIT (verified 2026-11-17). A mismatch aborts the run.
FILES = {
    "ratings.dat": ("latest/ratings.dat", "f8a46fe159d4bea12d5ea38922807ce1d8e7c2563785ebc84c479357d80feb13"),
    "movies.dat": ("latest/movies.dat", "0403d193fe09b41b2f1c5451e526d74c8c4d8629adb66646be51ba785018001f"),
    "LICENSE": ("LICENSE", "ae985d01e5fe4f32d14937099a4e871dbdc2d93d8ccc2f7a141ec4ecddcab41c"),
}
NEVER_USED = ("users.dat",)  # Twitter id mapping: personal data, not downloaded


@dataclass
class PreparationReport:
    source: str = REPOSITORY
    commit: str = COMMIT
    file_sha256: dict = field(default_factory=dict)
    raw_ratings: int = 0
    raw_users: int = 0
    raw_items: int = 0
    dropped_zero_ratings: int = 0
    dropped_duplicate_pairs: int = 0
    min_user_ratings: int = 20
    min_item_ratings: int = 20
    core_iterations: int = 0
    ratings: int = 0
    users: int = 0
    items: int = 0
    rating_transform: str = "rating_10pt / 2 -> 0.5..5.0"

    def to_dict(self) -> dict:
        return asdict(self)


def download_movietweetings(dest: Path, force: bool = False) -> dict[str, str]:
    """Download the pinned files into ``dest`` and verify their checksums."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, (rel, sha) in FILES.items():
        path = dest / name
        if force or not path.exists():
            download_archive(f"{RAW_BASE}/{rel}", path, expected_sha256=sha)
        digest = sha256_file(path)
        if digest != sha:
            raise ValueError(f"{path} does not match the pinned checksum ({digest} != {sha})")
        digests[name] = digest
    return digests


def read_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="::", engine="python", header=None,
                     names=["userId", "movieId", "rating", "timestamp"],
                     dtype={"userId": "int64", "movieId": "int64", "rating": "int64", "timestamp": "int64"})
    return df


def read_movies(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 2 or not parts[0].strip().isdigit():
                continue
            rows.append((int(parts[0]), parts[1], parts[2] if len(parts) > 2 else ""))
    movies = pd.DataFrame(rows, columns=["movieId", "title", "genres"])
    movies["genres"] = movies["genres"].replace("", "(no genres listed)")
    return movies.drop_duplicates("movieId", keep="first").reset_index(drop=True)


def k_core(df: pd.DataFrame, min_user: int, min_item: int) -> tuple[pd.DataFrame, int]:
    """Iteratively drop users/items with fewer than the minimum number of ratings."""
    it = 0
    while True:
        it += 1
        n = len(df)
        uc = df["userId"].value_counts()
        df = df[df["userId"].isin(uc[uc >= min_user].index)]
        ic = df["movieId"].value_counts()
        df = df[df["movieId"].isin(ic[ic >= min_item].index)]
        if len(df) == n:
            return df.reset_index(drop=True), it


def prepare(raw: pd.DataFrame, min_user: int = 20, min_item: int = 20) -> tuple[pd.DataFrame, PreparationReport]:
    """Apply the fixed preparation described in the module docstring."""
    rep = PreparationReport(min_user_ratings=min_user, min_item_ratings=min_item)
    rep.raw_ratings, rep.raw_users, rep.raw_items = len(raw), raw["userId"].nunique(), raw["movieId"].nunique()
    if ((raw["rating"] < 0) | (raw["rating"] > 10)).any():
        raise ValueError("MovieTweetings ratings must lie in 0..10")
    df = raw[raw["rating"] > 0]
    rep.dropped_zero_ratings = int(len(raw) - len(df))
    before = len(df)
    df = df.sort_values("timestamp", kind="mergesort").drop_duplicates(["userId", "movieId"], keep="last")
    rep.dropped_duplicate_pairs = int(before - len(df))
    df = df.assign(rating=df["rating"].astype("float64") / 2.0)
    df, rep.core_iterations = k_core(df, min_user, min_item)
    rep.ratings, rep.users, rep.items = len(df), df["userId"].nunique(), df["movieId"].nunique()
    return df[["userId", "movieId", "rating", "timestamp"]], rep


def load_movietweetings(data_dir: Path, min_user: int = 20, min_item: int = 20,
                        download: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, PreparationReport]:
    data_dir = Path(data_dir)
    digests = download_movietweetings(data_dir) if download else {
        n: sha256_file(data_dir / n) for n in FILES if (data_dir / n).exists()}
    for name in NEVER_USED:
        if (data_dir / name).exists():
            logger.warning("%s is present in %s but is never read (personal data).", name, data_dir)
    ratings, rep = prepare(read_ratings(data_dir / "ratings.dat"), min_user, min_item)
    rep.file_sha256 = digests
    movies = read_movies(data_dir / "movies.dat")
    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())].reset_index(drop=True)
    logger.info("MovieTweetings: %d raw -> %d ratings | %d users | %d movies after %d-core",
                rep.raw_ratings, rep.ratings, rep.users, rep.items, min_user)
    return ratings, movies, rep
