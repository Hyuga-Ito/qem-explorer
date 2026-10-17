# Contributing to QEM Explorer

## Set up

```bash
python -m venv .venv && source .venv/bin/activate        # Windows (Anaconda Prompt): conda create -n qemx python=3.12
pip install -r requirements.txt -r requirements-dev.txt
streamlit run app.py
```

Supported platforms are Linux, macOS and Windows with Python 3.10–3.13. `requirements-lock.txt` pins the versions the release was tested with.

PNG/SVG figure export needs Kaleido ≥ 1 and a Chrome or Chromium browser. Without them, exports still contain every table and each figure's Plotly JSON.

## Tests

```bash
python -m pytest -q                     # everything (≈ 25 min on 2 cores)
python -m pytest -q -m "not slow"       # without the end-to-end AppTest pages
python scripts/validate_learning_moments.py --out validation_learning.json   # statistical check (≈ 2 min)
```

Rules that keep the suite meaningful:

- A bug fix comes with a test that fails without the fix.
- Do not loosen a tolerance to make a test pass. Derive the tolerance from round-off or Monte Carlo error, and say how in a comment (see `tests/test_nil.py`, the objective check).
- Numbers quoted in the Learn chapters are recomputed by tests (`tests/test_learn_v142.py`); change the text and the test together.
- Statistical tests state their pass criterion before running: an interval from the declared Monte Carlo error, never "close enough".

## Code conventions

- **Spelling and units:** documentation and UI text use American English. Angles are in radians. Bitstrings are written c(n−1)…c0 with little-endian integers (q0 = 1s bit).
- **Safety:** no `eval`/`exec` on user input. Parsers are whitelists. New state or experiment fields go into `qemx/state_schema.py` or `qemx/experiment.py`, with validation and a migration note.
- **Statistics:** results of methods with finite training follow the contract in `qemx/stats.py`. `mean`/`var1` describe the protocol, not the exact-training reference.
- **Distributions:** distances between distributions go through `qemx/discrepancy.py`. Never clip a quasi-probability silently.
- **Errors:** a long or heavy computation is wrapped so that memory errors and timeouts become messages.
- **Claims:** a method name or citation must match what the code does. Mark deviations from a paper as deviations (see `docs/literature_matrix.md`).

## Data and manifests

Pre-registered manifests are never edited in place; write a new version. Record schemas carry a version. A run directory refuses to mix records of different statistical semantics.
