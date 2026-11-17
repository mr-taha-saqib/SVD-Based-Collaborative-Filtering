# Contributing

1. Create a branch from `main`; keep commits focused, with descriptive messages.
2. `pip install -r requirements-dev.txt`
3. Before pushing: `ruff check src tests scripts` and `pytest -q` (no network needed).
4. Rules that protect the methodology:
   * never pass the test partition to model selection or retraining code;
   * keep historical results in `outputs/` unchanged; new results go to `outputs/corrected/`;
   * new metrics must state which partition they come from;
   * no secrets in code, config or tests (`.env` is git-ignored).
5. Add or update tests for any behaviour change; never weaken an assertion to make a test pass.
6. Update `CHANGELOG.md` and the relevant document (README, API_DOCUMENTATION, ...).
7. Open a pull request; CI (lint, tests on 3.12/3.13, Docker smoke test) must pass.
