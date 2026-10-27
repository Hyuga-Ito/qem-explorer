# Corrected regenerations of 1.8.0 outputs (1.9.0)

Files here are regenerated **from the unchanged 1.8.0 study** (`reports/selection_v180`) with the corrected 1.9.0
generators. The historical files in `reports/selection_v180` stay as they were released; see
`docs/selection/ERRATA_v180.md` for what changed and why.

* `figures/supp_deployment_policy.*` — the deployment-policy figure with the primary (D41: frozen simulator-feature
  configuration, allocation-level preflight) and secondary (default deployable bundle) roles corrected, the exact metric
  and the fresh-design risk target stated. Command:
  `python -m qemx.selection figures configs/selection/dev18.yaml --out reports/selection_v180 --plot-config configs/selection/figures18.yaml --only supp_deployment_policy --figures-out reports/corrections_v190/figures`
