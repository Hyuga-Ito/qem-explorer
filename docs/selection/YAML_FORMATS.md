# YAML files of QEM Explorer and how each is run (1.9.0)

Every YAML file is recognized by its content and validated by **its own parser**:

```powershell
python -m qemx.selection check-config FILE [FILE ...]     # exit 0 valid, 2 invalid; prints format, parser, command
```

| format (identifying key) | shipped examples | parser | run it with |
|---|---|---|---|
| selection study configuration (`schema: qemx.selection.config/1`) | `configs/selection/*18.yaml`, `*19.yaml`, the 1.6/1.7 ones | `design.load` (JSON Schema, key by key: unknown keys refused, also inside `analysis`, `mismatch`, `regimes`, `limits`) + `design.check` (semantic) | `python -m qemx.selection plan|label|analyze|freeze|… FILE --out DIR` |
| study pipeline (`schema: qemx.selection.pipeline/1`) | `configs/pipelines/*.yaml` | `pipeline.load` (keys, stages, files resolved next to the pipeline file, then the repository) | `python -m qemx.selection pipeline FILE --out DIR` |
| plotting configuration (`schema: qemx.selection.figures/1`) | `configs/selection/figures*.yaml` | `plots.load_config` (unknown keys refused; `release`, `description` descriptive) | `… figures CONFIG --out DIR --plot-config FILE` |
| figure/table specification (`schema: qemx.selection.figure-spec/1`) | `configs/selection/figure_spec19.yaml` | key check; generators tested | `python -m qemx.selection.tables19` |
| experiment document (`schema: qemx.experiment/1`) | `examples/experiments/*.yaml` | `experiment.load` + `resolve` | `python -m qemx.experiment validate|plan|run FILE`; `python -m qemx.selection recommend --experiment FILE --budget B` |
| divergence manifest (`manifest_version` + `kind: divergence`) | `manifests/direction_A_divergence_*.yaml` | `divergence.suite_from_manifest` | `python scripts/dirA_divergence.py plan FILE --out DIR` |
| phase-study manifest (`manifest_version`, no `kind`) | `manifests/direction_A_*.yaml`, `manifests/smoke.yaml`, `data/pilot_v2/manifest.yaml` | `phase.manifest.load` + `expand` | `python -m qemx.phase.run plan|run FILE` |
| hardware section (`backend_name`, `blocks`, `budget`) | `configs/selection/hardware17.yaml` | key check; `selection.hardware.plan` | `python -m qemx.selection hardware plan --hardware-config FILE --bundle B` (plans only; never submits) |
| GitHub Actions workflow | `.github/workflows/ci.yml` | (GitHub) | runs on GitHub after a push |

Parsing is not execution: a valid configuration of a large study has not been run.

## Selection configurations: what is checked before anything is labeled

* **Schema:** budgets and protocols unique; every list (widths, inputs, observables, layouts, devices) non-empty and
  unique; `master_seed ≥ 0`; `analysis` keys (`splits`, `split_options` per split with only its own options,
  `feature_sets`, `model_families`, `extras`, …), `mismatch` keys and `regimes` keys are each a closed set — a typo such
  as `feature_set` is refused instead of silently falling back to the default.
* **Semantics (`design.check`, run by `plan`, `label`, `check-config`):** unknown feature sets or scenarios; more fresh
  designs per half than trials per half; mode-specific `learning_designs` keys (`count` with `fresh`, `per_half` with
  `fixed-shared`); no development tasks; a held-out device, noise setting, family or width that leaves a side empty;
  fewer development base circuits than folds (warning: k is reduced); descriptive keys that the code does not read
  (`mismatch.robust`, `regimes.*`: warnings).
* **Resources (`resources.preflight`):** the known estimate against `limits.cpu_hours`, `disk_gb`, `min_free_disk_gb`
  (refused unless `--override-limits "reason"`); unknown estimates reported as unknown.

## How a task list expands (and how to see it)

```powershell
python -m qemx.selection plan CONFIG --tasks-csv tasks.csv     # every task: identities, split, block, family, width,
                                                              # parameters, input, observable, layout, device, noise
                                                              # modifiers, how it was selected
```

The historical semantics are kept so that 1.6–1.8 task identities do not change (pinned in
`tests/data/expansion_ids_v180.json`):

1. **Default variant.** Each base circuit of a block gets its *default variant* — default input, default observable,
   `best` layout, the block's first device — even when the block's lists name only other values, unless
   `include_default: false`. `plan` reports these as "inserted default variants".
2. **Sampling vs full grid.** `variants_per_base: k` samples k further variants without replacement from the block's
   full grid (seed `master_seed + 101`, per block); `variants_per_base: all` takes the full grid.
3. **Family-defined inputs.** `qft`, `bv`, `grover` and `ghz` always use their own input; listed inputs do not apply to
   them (reported).
4. **Duplicates across blocks** are kept once (the first block).

For new studies set `expansion_policy: explicit`: every block must state `include_default` and `variants_per_base`, and
listing inputs for a family that forces its own is an error. All 1.9.0 studies (`smoke19`, `pilot19`, `fulldev19`,
`conf19`) use it.

## Device and noise specifications

`fake_<device>` or `fake_<device>@k=v,…` with modifiers `s` (overall error scale), `g1`, `g2` (one-/two-qubit gate
errors), `t` (relaxation rates), `ro` (readout), `chi` (coherent fraction of each gate's error; uses the exact engine),
`hom` (homogeneous). They are calibration-supplied: the snapshot the recommender's features read changes with them.
Hidden execution-law perturbations (`truth`, the mismatch scenarios) are a different mechanism and are never features.
