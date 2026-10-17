# Migrating from 1.4.2 to 1.5.0

1.5.0 changes some reported numbers, because they now describe the protocol that was actually run. It renames one method id, and it adds an experiment file format.

**Existing data are never rewritten:**

- Old run directories stay as they are.
- A run directory written by 1.4.x refuses to be finished by 1.5, because its records would mix statistical meanings.
- Old manifests and state files still load. Every migration is reported.

## 1. Learning methods (CDR, FR, NIL): what `estimate` and `RMSE` mean

**1.4.2.** A learning method's row showed two different things:

- the **exact-training reference fit**: the regression fitted to exact training expectation values, an oracle;
- the spread of **finite-shot refits**.

So bias and RMSE described a protocol nobody runs. In the QA report's diagnostic cases, the reported and the empirical RMSE differed considerably:

| case | 1.4.2 reported | 1.4.2 empirical | 1.5 reported | 1.5 empirical (6 000 independent experiments) |
|---|---|---|---|---|
| CDR-OLS | 0.0880 | 0.1354 | 0.1328 | 0.1354 (z = −0.90) |
| FR-Ridge | 0.2704 | 0.3973 | 0.3986 | 0.3973 (z = 0.47) |
| NIL-ℓ1 | 0.1754 | 0.1818 | 0.1825 | 0.1818 (z = 1.44) |

Reproduce the comparison with `python scripts/validate_learning_moments.py --out validation_learning.json`. The pass criterion is |z| ≤ 4 against the declared Monte Carlo errors.

**1.5.** The fields are now separate (`qemx/stats.py`):

| field | meaning |
|---|---|
| `estimate_kind` | `exact` (no finite training), `protocol` (expected over the finite training data), or `measured` (one hardware experiment) |
| `mean`, `var1` | the protocol's moments: μ = E_D[m(D)] and Var = Var_D[m(D)] + E_D[v(D)]/N_t |
| `reference_mean`, `reference_var1` | the exact-training fit, kept for comparison (the table column *exact-fit value*) |
| `moments` | the Monte Carlo record behind `mean`/`var1`: repeats, finite draws, MC standard errors, failure rate, tolerance |
| `flags` | e.g. `conditional-on-success`, which means some refits failed and the moments are conditional on success; `calibration-mean-first-order` |

**What to do.**

- Recompute any 1.4.x Playground, Batch or export numbers for learning methods that you want to quote.
- Phase-study records written by 1.4 used their own, already protocol-level, ledger. They are still readable. They use record schema `qemx.phase.cell/2`; 1.5 writes `/3`.
- To re-run a study, use a new `--out` directory. `run` refuses to mix the two schemas unless `--force` is given.

## 2. Method id `features: nil` → `features: fr`

The learning estimator's feature-regression variant (FR: the noisy ⟨O⟩, ⟨Z_i⟩ and ⟨Z_iZ_j⟩) was selected with `features: "nil"`. That name reads as neighbor-informed learning (NIL), which is a different method: `estimator: nil`.

| where | 1.4.x | 1.5 |
|---|---|---|
| recipe parameter | `{"estimator": "learning", "features": "nil"}` | `{"estimator": "learning", "features": "fr"}` |
| presets | "FR · Ridge", "FR · Lasso" (already so named) | unchanged, now `features: fr` |
| manifests | method keys such as `NIL-L`, `NIL-R` with `features: nil` | still valid; the key names are kept so cell ids do not change, and a note is printed: `learning: features='nil' is the legacy name of features='fr' (migrated)` |

The legacy value is translated when a recipe is built (`Recipe.make`). The manifest file itself is never rewritten. New manifests (`direction_A_v2_*.yaml`) use the keys `FR-R` and `FR-L` for feature regression and `NIL-G2` for neighbor-informed learning.

## 3. Circuit input and output

**1.4.2.** The QFT templates had a parameter labeled **"output s"**. Setting it changed the *input* preparation: Hadamard and phase gates that prepare F†|s⟩. Choosing output 0 inserted |+⟩ preparation.

**1.5.**

- The parameter is **"input index s"**.
- A new parameter, **"input preparation"**, takes one of two values:
  - `fourier`: the known-answer benchmark F†|s⟩, whose ideal output is |s⟩;
  - `basis`: |s⟩ prepared with X gates, whose ideal output is uniform.
- The Playground shows a read-only **Output** panel: the ideal prediction derived from the input, the circuit and the measurement. The noisy prediction and the sampled outcomes stay in *Mitigate & compare*.

**Compatibility.**

- A parameter set without `prep` means `fourier`, which is exactly the 1.4.2 circuit (tested).
- State files store the circuit's operations explicitly, so a saved 1.4.2 state reproduces the same circuit.
- The phase-study QFT family is unchanged.

## 4. Readout mitigation

`Context.readout_gamma` now returns the inverse 1-norm product ∏_i (1 + |e_i − h_i|) / |1 − e_i − h_i|. For two bits with e = h = 0.6 it gives **25**; 1.4.2 displayed 1e24. TREX uses the same formula with e = h = p̄.

Two consequences:

- An assignment matrix with a negative determinant (a readout that mostly flips) is invertible and accepted.
- A singular matrix (P(1|0) + P(0|1) = 1) is an error with an explanation. It is no longer clamped.

## 5. Budget preflight

The Playground checks every contender's fixed cost (training and calibration shots) before running anything. A contender that cannot run at the total budget is shown as **"insufficient_budget: needs at least N shots"**: it is not run, and it does not count as 120 failed repeats.

Studies (`compare(..., check_budget=False)`, the default) still run every recipe, because their cost curves are evaluated at many budgets.

A budget-scaled teaching preset, **"NIL · small budget"** (T = 6, 3 neighbors + the bare circuit, 128 shots each, so 3 072 training shots), runs within 4 096 shots. It is not the configuration of the NIL paper.

## 6. Guides (Learn page) and contenders

- **Saved custom recipes are now separate from active contenders.** Older states without `custom_active` treat every saved recipe as active, as before.
- **Applying a guide** *replaces* the active contenders by default, so every guide starts from its own experiment. Saved recipes are never deleted. The toggle *Keep my current contenders* merges them instead.
- **Setups are atomic.** Every field is validated first. If anything fails, nothing changes.

## 7. Exports

PNG and SVG rendering runs in a separate, time-limited process. A missing Chrome, a crashed or a hung renderer returns the ZIP promptly, with every table, the state and each figure's Plotly JSON, plus any images finished before the failure. The README inside the ZIP says what is missing. Turning off the sidebar toggle *PNG/SVG images in ZIP downloads* skips rendering.

## 8. One experiment file for the app, the CLI and YAML

Experiment documents use `schema: qemx.experiment/1` and are validated by the same schema as the app's state files (`qemx/state_schema.py`).

- **Getting one:** the Playground's **Export → Download experiment (.yaml)** writes the current experiment. **Load a state or experiment** accepts `.json` and `.yaml`.
- **CLI:**

  ```bash
  python -m qemx.experiment validate|plan|run FILE [--out DIR]
  python -m qemx.experiment from-state qem_explorer_state.json --out experiment.yaml   # 1.4 state → YAML
  ```

- **Guarantees:**
  - Errors name the field, for example `methods[1].params: …`.
  - Credentials anywhere in a document are refused.
  - YAML is read with `safe_load`.
  - The configuration hash identifies the resolved experiment. `run` never overwrites a result of another configuration.

## 9. Divergence tables

`divergence.csv` gains the columns `p_value` (two-sided normal), `p_holm` and `q_bh`. The latter two are adjusted within each predictor × method × source family. The meaning of other columns has changed:

- `significant` is now **Holm-adjusted p < 0.05**. In 1.4 it was the unadjusted |z| > 3.
- The unadjusted flag is kept as `abs_z_gt_3`.
- `structures.csv` adds the columns `holm_worse` and `holm_better`.

## 10. Hardware submission

`scripts/dirA_divergence.py run-ibm` now requires a campaign ledger (`python -m qemx.campaign init`). It admits jobs one by one against the shared 600 s ceiling, defaults to **job mode** (`--mode batch` is still available), writes `jobs.json` after every job, and sets each job's `max_execution_time` from its own high usage estimate.

The IBM page uses the ledger when `QEMX_CAMPAIGN_LEDGER` points to one. See `docs/hardware_preflight.md`.

## 11. Navigation

| 1.4.2 sidebar | 1.5 |
|---|---|
| Cost vs precision, Bias–variance bullseye, Head-to-head sweeps | drill-downs of **QEM Performance & Selection** |
| Model mismatch | drill-down of **Noise Models & Discrepancy** |

Page addresses, Learn setups and bookmarks are unchanged.

## 12. Open decisions for the project owner

- **License.** The archive has no `LICENSE` file. Choose one before calling the project open source. `CITATION.cff` has a placeholder author and no license field until then.
- **Security contact.** `SECURITY.md` asks for a private contact address.
