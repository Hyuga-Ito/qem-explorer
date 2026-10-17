"""qemx.phase — Direction A of the thesis plan: equal-budget phase diagrams of quantum error mitigation (v1.3).

Backend only (no Streamlit): a cell runner that measures, for every method on one grid coordinate, the ledger
primitives (b, v, w, u, …), an exact integer budget ledger, Monte-Carlo validation, a manifest-driven batch runner
and a versioned record schema. See docs/v13_direction_A_design.md.
"""
from .ledger import Cost, Policy, allocate, rmse, rmse_curve, budget_for, winners, crossovers   # noqa: F401
from .lightcone import calculate_lambda_lc, LambdaReport                                 # noqa: F401
from .scaling import scale_noise, calibrated_noise, calibration_from_model, device_at     # noqa: F401
from .cell import CellSpec, MethodSpec, cell_runner                                      # noqa: F401
