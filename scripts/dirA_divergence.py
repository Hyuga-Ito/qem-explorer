#!/usr/bin/env python3
r"""Hardware against fake-device divergence: where the device underperforms the noise model built from its own
component-level metrics (Chen et al., Quantum 8, 1516 (2024), Fig. 11; Lubinski et al., IEEE TQE 4, 3100332 (2023)).

    # 1. plan — compile every benchmark for the device (offline on the fake twin, or on today's calibration)
    python scripts/dirA_divergence.py plan manifests/direction_A_divergence_v1.yaml --out data/dirA_div_v1
    python scripts/dirA_divergence.py plan manifests/direction_A_divergence_v1.yaml --out data/dirA_div_v1 --backend ibm_fez
    # 2a. hardware (the token comes from $QISKIT_IBM_TOKEN, a saved account, or a hidden prompt: --ask-token)
    python scripts/dirA_divergence.py run-ibm     data/dirA_div_v1 [--channel ibm_quantum_platform] [--instance CRN]
    python scripts/dirA_divergence.py collect-ibm data/dirA_div_v1 [--wait]
    # 2b. without hardware
    python scripts/dirA_divergence.py run-local data/dirA_div_v1                 # Sampler local mode on the fake twin
    python scripts/dirA_divergence.py emulate   data/dirA_div_v1 --truth depolarizing --inject crosstalk=0.05
    # 3. predictions, comparison, tables and figures (redraw later from the table, e.g. on Lubinski's depth axis)
    python scripts/dirA_divergence.py analyze data/dirA_div_v1 --tag ibm --out figs_div --workers 8
    python scripts/dirA_divergence.py figures figs_div/tables/divergence.csv --out figs_div --x norm_depth

A run directory holds ``plan.json`` (the as-executed circuits: compact specs, physical qubits, measurement maps, shots;
SHA-256 digests of every circuit) and one folder per observation set, ``obs/<tag>/``: ``counts.json``,
``calibration.json`` (the device's calibration on each instance's qubits: at execution time when the service keeps
the history, otherwise at submission), ``meta.json`` and, for hardware, ``jobs.json`` (job ids only). ``analyze``
adds ``predictions.npz`` (cached) and writes tables and figures to ``--out``.

Credentials. The IBM token is read from an environment variable (default QISKIT_IBM_TOKEN), from a saved account
(``QiskitRuntimeService.save_account``) or from a hidden prompt (``--ask-token``, getpass). It is never a command-line
argument (shell history, process lists), never written to any file of the run directory, never printed; connection
errors are reported with the token masked. A token is a password: rotate it if it was ever pasted into a notebook or
a file.

Tables (``--out``/tables): ``divergence`` (one row per instance × method × predictor: observed and predicted Fc and
normalized fidelity, Δ, z), ``structures`` (per family, method and predictor: mean and largest Δ, share of |z| > 3,
rank correlations of Δ with width and two-qubit count), ``top`` (the most divergent circuits), ``rates`` (effective
two-qubit error rate fitted to observed and predicted Fc, as Chen et al.'s Fig. 9 lines), ``errors``.
Figures: the difference volumetric plot (``dirA_figures.difference_volumetric``) per method, all methods side by side,
and predicted against observed Fc (``dirA_figures.predicted_vs_observed``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import json
import math
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dirA_figures as F  # noqa: E402
from dirA_io import HAVE_QEMX, DirectoryFigureSink, DirectoryReportSink, Report  # noqa: E402,F401

PRIMARY_PREDICTOR = "depolarizing"
Z_SIG = 3.0                 # descriptive threshold of one unadjusted |z| (kept for continuity with 1.4 tables)
ALPHA = 0.05                # family-wise error rate of the Holm-adjusted "significant" flag (1.5)


# ------------------------------------------------------------------ computations (objects in, tables out)
def divergence_table(rows: list) -> pd.DataFrame:
    """The comparison rows of :func:`qemx.divergence.compare` as a table (one row per instance × method × predictor)."""
    D = pd.DataFrame(rows)
    if D.empty:
        return D
    for c in ("fc_obs", "fc_pred", "fc_pred_sd", "delta", "z", "nf_obs", "nf_pred", "delta_nf", "z_nf"):
        if c in D.columns:
            D[c] = pd.to_numeric(D[c], errors="coerce")
    if "z" not in D.columns:
        D["significant"] = False
        return D
    D["abs_z_gt_3"] = D["z"].abs() > Z_SIG                  # descriptive only: one unadjusted flag per row
    # Inference across many circuits (1.5): two-sided normal p-values of z, adjusted within each family of tests
    # (predictor × method × source): Holm for the family-wise error rate, Benjamini–Hochberg q-values for the
    # false-discovery rate. "significant" is Holm at ALPHA. z assumes the replicate spread is the sampling spread of
    # the observation under the model (qemx.divergence.zscore); scripts/rehearse_divergence.py checks the resulting
    # false-positive rate on emulated data from the model itself.
    from qemx import discrepancy as DS
    D["p_value"] = DS.two_sided_p(D["z"].to_numpy(float))
    D["p_holm"], D["q_bh"] = np.nan, np.nan
    keys = [k for k in ("predictor", "method", "source") if k in D.columns]
    for _, idx in (D.groupby(keys).groups.items() if keys else [(None, D.index)]):
        pv = D.loc[idx, "p_value"].to_numpy(float)
        D.loc[idx, "p_holm"] = DS.adjust_pvalues(pv, "holm")
        D.loc[idx, "q_bh"] = DS.adjust_pvalues(pv, "bh")
    D["significant"] = D["p_holm"] < ALPHA
    return D


def _spearman(x, y) -> float:
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return math.nan
    return float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])


def structure_summary(D: pd.DataFrame) -> pd.DataFrame:
    """Per predictor, method and family: circuits, mean and largest Δ, share with |z| > 3 (device worse: z > 3), and the
    rank correlation of Δ with the width and with the two-qubit gate count — which algorithmic structures diverge,
    and whether the divergence grows with scale."""
    d = D[D.get("status", "ok") == "ok"] if "status" in D else D
    if d.empty:
        return pd.DataFrame()
    out = []
    for (pr, m, fam), g in d.groupby(["predictor", "method", "family"]):
        out.append(dict(predictor=pr, method=m, family=fam, circuits=len(g), mean_delta=float(g["delta"].mean()),
                        max_delta=float(g["delta"].max()), worst=g.loc[g["delta"].idxmax(), "label"],
                        share_worse=float((g["z"] > Z_SIG).mean()), share_better=float((g["z"] < -Z_SIG).mean()),
                        holm_worse=int(((g["p_holm"] < ALPHA) & (g["z"] > 0)).sum()) if "p_holm" in g else 0,
                        holm_better=int(((g["p_holm"] < ALPHA) & (g["z"] < 0)).sum()) if "p_holm" in g else 0,
                        rho_width=_spearman(g["width"], g["delta"]), rho_two_qubit=_spearman(g["two_qubit"],
                                                                                              g["delta"])))
    return pd.DataFrame(out).sort_values(["predictor", "method", "mean_delta"], ascending=[True, True, False])


def top_divergent(D: pd.DataFrame, k: int = 10, predictor: str = PRIMARY_PREDICTOR) -> pd.DataFrame:
    """The k (instance, method) pairs with the largest Δ among those with z > 3 under ``predictor``."""
    d = D[(D["predictor"] == predictor) & (D.get("status", "ok") == "ok")]
    d = d[d["z"] > Z_SIG].sort_values("delta", ascending=False)
    keep = ["label", "family", "width", "two_qubit", "depth", "method", "fc_pred", "fc_obs", "delta", "z", "phys"]
    return d[[c for c in keep if c in d.columns]].head(k).reset_index(drop=True)


def rates_table(D: pd.DataFrame) -> pd.DataFrame:
    """Effective two-qubit error rate ε_eff per method: log Fc = a + N₂·log(1 − ε) fitted to the observed values and to
    each predictor's (Chen et al. Fig. 9). ratio = ε_eff(observed) / ε_eff(predicted). Unmitigated and readout-
    mitigated estimators only: an extrapolated (ZNE) distribution does not decay exponentially in N₂ — its Fc is
    limited by the extrapolation's shot noise — so a rate fitted to it means nothing."""
    from qemx.divergence import effective_rate
    d = D[D.get("status", "ok") == "ok"] if "status" in D else D
    if "kind" in d.columns:
        d = d[d["kind"] != "zne"]
    out = []
    for m, g in d.groupby("method"):
        obs = g.drop_duplicates("instance")
        e_o, a_o, k_o = effective_rate(obs["two_qubit"], obs["fc_obs"])
        for pr, gp in g.groupby("predictor"):
            e_p, a_p, k_p = effective_rate(gp["two_qubit"], gp["fc_pred"])
            out.append(dict(method=m, predictor=pr, eps_observed=e_o, eps_predicted=e_p,
                            ratio=e_o / e_p if (math.isfinite(e_o) and math.isfinite(e_p) and e_p > 0) else math.nan,
                            circuits_observed=k_o, circuits_predicted=k_p))
    return pd.DataFrame(out)


def error_summary(E: pd.DataFrame) -> pd.DataFrame:
    """The plan's and the run's errors grouped: one row per (method, reason) with the number of benchmarks (variants
    of one benchmark count once) and a few of them — e.g. fractional ZNE skipped on circuits with too few two-qubit
    gates to fold a third of them."""
    if E.empty:
        return E
    e = E.copy()
    if "method" not in e.columns:
        e["method"] = ""
    e["method"] = e["method"].fillna("")
    msg = e["error"].astype(str)
    e["reason"] = np.where(msg.str.contains("fold to the same λ"),
                           "the scales fold to the same λ (too few two-qubit gates for fractional steps)",
                           msg.str.replace(r" on .*? · v\d+:", ":", regex=True).str.slice(0, 90))
    e["bench"] = e["instance"].astype(str).str.replace(r"#v\d+$", "", regex=True)
    rows = []
    for (m, why), g in e.groupby(["method", "reason"], sort=False):
        benches = list(dict.fromkeys(g["bench"]))
        names = [b.split("|")[0] + "-" + b.split("|")[1] if b.count("|") >= 2 else b for b in benches]
        rows.append(dict(method=m or "(all)", reason=why, benchmarks=len(benches),
                         examples=", ".join(list(dict.fromkeys(names))[:5]) + (" …" if len(set(names)) > 5 else "")))
    return pd.DataFrame(rows)


def feature_columns(D: pd.DataFrame) -> list:
    """The context-feature columns of a divergence table (qemx.context_noise.FEATURES, prefixed f_)."""
    from qemx.context_noise import FEATURES
    return ["f_" + k for k in FEATURES if "f_" + k in D.columns]


def response_se(d: pd.DataFrame, response: str = "ex_ln_nf") -> np.ndarray:
    """Standard error of each circuit's excess log-infidelity: delta method, SE(ln F_obs) = SE(F_obs) / F_obs, with
    SE(F_obs) pooled from the observation's bootstrap and the model's replicate SD (two estimates of the same shot
    noise, as in the z-scores; the replicate mean itself is exact by comparison); floored at 1e-3."""
    nf = response.endswith("_nf") or response == "ex_nf_per_2q"
    f, se, sd = ("nf_obs", "nf_obs_se", "nf_pred_sd") if nf else ("fc_obs", "fc_obs_se", "fc_pred_sd")
    col = lambda c: pd.to_numeric(d[c], errors="coerce").to_numpy() if c in d else np.full(len(d), np.nan)  # noqa
    x, e, s_ = col(f), col(se), col(sd)
    pooled = np.where(np.isfinite(s_), np.sqrt((e ** 2 + np.nan_to_num(s_) ** 2) / 2), e)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(pooled / x)
    return np.maximum(out, 1e-3)


def with_priced(d: pd.DataFrame, response: str = "ex_ln_nf") -> pd.DataFrame:
    """d with the model feature f_priced = −ln F_pred (qemx.context_noise.MODEL_FEATURES; NaN below the floor)."""
    from qemx.divergence import EXCESS_FLOOR
    col = "nf_pred" if (response.endswith("_nf") or response == "ex_nf_per_2q") else "fc_pred"
    x = pd.to_numeric(d[col], errors="coerce") if col in d else pd.Series(np.nan, index=d.index)
    return d.assign(f_priced=np.where(x >= EXCESS_FLOOR, -np.log(x.clip(lower=EXCESS_FLOOR)), np.nan))


def budget_method(D: pd.DataFrame, plan=None) -> str:
    """The method the budget is read from: readout-mitigated (RO) when the suite has it — readout then drops out of
    both sides and −ln F_pred prices the gate errors alone — else RAW."""
    have = set(D["method"]) if "method" in D else set()
    if plan is None:
        return "RO" if "RO" in have else ("RAW" if "RAW" in have else "")
    for kind in ("ro", "raw"):
        k = next((m.key for m in plan.suite.methods if m.kind == kind and m.key in have), None)
        if k:
            return k
    return ""


LADDER = (("none", ()), ("calibration (priced)", ("priced",)), ("context", None),
          ("calibration + context", "full"))


def model_ladder(D: pd.DataFrame, predictor: str = PRIMARY_PREDICTOR, method: str = "RO",
                 response: str = "ex_ln_nf", alpha: float = 0.01) -> tuple:
    """Which explanation of the divergence do the data need? (table, verdict). Weighted non-negative fits of every
    circuit's excess log-infidelity y_c (weights 1/SE_c, :func:`response_se`) by nested models:

    none                   y = 0: the model is right (χ² = Σ (y/SE)²)
    calibration (priced)   y = c·(−ln F_pred): every calibrated error (1 + c)× too low — no context
    context                y = Σ β_k·feature_k over the context mechanisms (qemx.context_noise.CONTEXT)
    calibration + context  both

    With correct SEs a model that suffices has χ²/dof ≈ 1 (p = the χ² tail). Verdict (:func:`ladder_verdict`):
    "as modeled" (none suffices); "rescaled calibration" when context adds nothing significant (Δχ² test) *or*
    nothing material — the calibration alone takes ≥ CAL_SHARE of the χ² that all mechanisms together remove (with
    many precise circuits a first-order model is always rejected somewhere: the rescaling is linear in −ln F_pred only
    to first order); "context" otherwise; plus "beyond the linear budget" when even the full model leaves χ²/dof > 2
    (coherent or non-additive mechanisms)."""
    from scipy.stats import chi2 as CHI2
    from qemx.context_noise import CONTEXT, nnls_attribution
    d = with_priced(D[(D["predictor"] == predictor) & (D["method"] == method) & (D["status"] == "ok")], response)
    if "variant" in d.columns:
        d = d[d["variant"] >= 0]
    if d.empty:
        return pd.DataFrame(), ""
    y = pd.to_numeric(d[response], errors="coerce").to_numpy()
    w = 1 / response_se(d, response)
    ctx = [k for k in CONTEXT if "f_" + k in d.columns and (pd.to_numeric(d["f_" + k], errors="coerce") != 0).any()]
    ok = np.isfinite(y) & np.isfinite(w) & np.isfinite(pd.to_numeric(d["f_priced"], errors="coerce").to_numpy())
    for k in ctx:
        ok &= np.isfinite(pd.to_numeric(d["f_" + k], errors="coerce").to_numpy())
    y, w, d = y[ok], w[ok], d[ok]
    n = len(y)
    rows = []
    for name, cols in LADDER:
        cols = ctx if cols is None else (["priced"] + ctx if cols == "full" else list(cols))
        if not cols:
            c2, dof, coef = float(np.sum((w * y) ** 2)), n, {}
        else:
            r = nnls_attribution(d[["f_" + k for k in cols]].astype(float).to_numpy(), y, cols, boot=0, weights=w)
            c2, dof = r["chi2"], r["dof"]
            coef = {k: float(v) for k, v in zip(cols, r["coef"]) if np.isfinite(v)}
        rows.append(dict(model=name, chi2=c2, dof=dof, chi2_per_dof=c2 / max(dof, 1), p=float(CHI2.sf(c2, max(dof, 1))),
                         circuits=n, priced=coef.get("priced", math.nan),
                         coefficients=", ".join("%s=%.3g" % kv for kv in coef.items() if kv[1] > 0)))
    L = pd.DataFrame(rows)
    d_chi = L["chi2"].iloc[1] - L["chi2"].iloc[3]
    k_ctx = max(int(L["dof"].iloc[1] - L["dof"].iloc[3]), 1)
    L["p_context_added"] = [math.nan, math.nan, math.nan, float(CHI2.sf(d_chi, k_ctx)) if d_chi > 0 else 1.0]
    L["calibration_share"] = [math.nan, calibration_share(L), math.nan, math.nan]
    L["context_features"] = len(ctx)
    return L, ladder_verdict(L, alpha)


CAL_SHARE = 0.95          # calibration share of the removable χ² above which context is immaterial
NONLINEAR_CHI2 = 2.0      # χ²/dof of the full model above which the linear budget is declared insufficient


def calibration_share(L: pd.DataFrame) -> float:
    """(χ²_none − χ²_calibration) / (χ²_none − χ²_full): how much of what every mechanism together explains a
    rescaled calibration explains alone (1: context adds nothing)."""
    tot = float(L["chi2"].iloc[0] - L["chi2"].iloc[3])
    return float((L["chi2"].iloc[0] - L["chi2"].iloc[1]) / tot) if tot > 0 else math.nan


def ladder_verdict(L: pd.DataFrame, alpha: float = 0.01) -> str:
    """The verdict of a :func:`model_ladder` table (also from its saved CSV)."""
    if L is None or len(L) < 4:
        return ""
    if not np.isfinite(L["chi2"].astype(float)).all():
        return "too few circuits to fit the context mechanisms (%d)" % int(L["circuits"].iloc[0])
    if "context_features" in L.columns and int(L["context_features"].iloc[0]) == 0:
        c = float(np.nan_to_num(L["priced"].iloc[1]))
        return ("no context features in this plan (made before 1.4.1: run refresh-features), so only a rescaled "
                "calibration is tested: ×%.2f, χ²/dof %.1f → %.1f" % (1 + c, L["chi2_per_dof"].iloc[0],
                                                                   L["chi2_per_dof"].iloc[1]))
    p_none, p_full = float(L["p"].iloc[0]), float(L["p"].iloc[3])
    d_chi = float(L["chi2"].iloc[1] - L["chi2"].iloc[3])
    p_add = float(L["p_context_added"].iloc[3])
    share = calibration_share(L)
    if p_none > alpha:
        v = "as modeled: no divergence beyond shot noise"
    elif p_add > alpha or (np.isfinite(share) and share >= CAL_SHARE):
        v = ("rescaled calibration: the calibrated errors are ×%.2f too low; context adds %s (Δχ² p = %.2g, "
             "calibration share %.0f %%)" % (1 + float(np.nan_to_num(L["priced"].iloc[1])),
                                              "nothing" if p_add > alpha else "little", p_add, 100 * share))
    else:
        v = ("context: the context mechanisms reduce χ² by %.0f beyond a rescaled calibration (p = %.2g; "
             "calibration share %.0f %%)" % (d_chi, p_add, 100 * share))
    if p_none <= alpha and p_full <= alpha and L["chi2_per_dof"].iloc[3] > NONLINEAR_CHI2:
        v += "; beyond the linear budget: χ²/dof = %.1f with every mechanism (coherent or non-additive errors)" % (
            L["chi2_per_dof"].iloc[3])
    return v


def attribution(D: pd.DataFrame, predictor: str = PRIMARY_PREDICTOR, method: str = "RO",
                response: str = "ex_ln_nf", boot: int = 500, seed: int = 0, mechanisms=None,
                weighted: bool = True) -> tuple:
    """The divergence budget: (coefficients, family budget, meta). Every circuit's unmodeled log-infidelity
    (``response``; default ln F_pred − ln F_obs of the normalized fidelity) is split by non-negative weighted least
    squares (weights 1 / SE, :func:`response_se`) into the mechanisms its as-executed form exposes
    (``mechanisms``, default qemx.context_noise.BUDGET: the calibrated gate and readout error the model already
    prices, −ln F_pred — a positive coefficient means the calibration under-reports it — spectators, parallel
    adjacent gates, idle time; with RAW also the calibrated readout error). Coefficients are excess log-infidelity per unit (per unit of reported error, per spectator exposure,
    per µs of idle time, …) with bootstrap 95 % intervals over circuits; VIF > 10 flags a mechanism these circuits
    cannot tell apart from the others. The family budget is each mechanism's mean contribution per circuit, plus
    what the mechanisms leave unexplained.

    How far the budget can be trusted is checked before any hardware run by injecting each mechanism into an
    emulated device (``emulate --inject``) and asking whether the budget names it (see the release notes'
    recovery table): a regression over application circuits separates only mechanisms the circuits do not vary
    together; the context probes are the designed experiment for the spectator mechanisms."""
    from qemx.context_noise import BUDGET, FEATURES, MODEL_FEATURES, nnls_attribution
    d = with_priced(D[(D["predictor"] == predictor) & (D["method"] == method) & (D["status"] == "ok")], response)
    if "variant" in d.columns:
        d = d[d["variant"] >= 0]
    want = list(mechanisms or (BUDGET + (("readout_error",) if method == "RAW" else ())))
    FEATURES = dict(FEATURES, **MODEL_FEATURES)
    # a mechanism is used when the plan has it; circuits where one is undefined (−ln F_pred below the fidelity
    # floor) are dropped by the fit, not the mechanism
    cols = [c for c in ("f_" + k for k in want) if c in d.columns and np.isfinite(
        pd.to_numeric(d[c], errors="coerce")).any()]
    if d.empty or not cols:
        return pd.DataFrame(), pd.DataFrame(), {}
    X = d[cols].astype(float).to_numpy()
    y = pd.to_numeric(d[response], errors="coerce").to_numpy()
    names = [c[2:] for c in cols]
    w = 1 / response_se(d, response) if weighted else None
    r = nnls_attribution(X, y, names, boot=boot, seed=seed, weights=w)
    coef = pd.DataFrame(dict(mechanism=names, unit=[FEATURES[k][0] for k in names], coef=r["coef"], lo=r["lo"],
                             hi=r["hi"], share=r["share"], vif=r["vif"], description=[FEATURES[k][1] for k in names]))
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if w is not None:
        ok &= np.isfinite(w) & (w > 0)
    dd = d[ok].assign(_fit=r["fitted"], _y=y[ok])
    beta = np.nan_to_num(np.asarray(r["coef"], float))
    rows = []
    for fam, g in dd.groupby("family"):
        Xg = g[cols].astype(float).to_numpy()
        for k, nm in enumerate(names):
            rows.append(dict(family=fam, mechanism=nm, contribution=float(np.mean(Xg[:, k] * beta[k])),
                             circuits=len(g)))
        rows.append(dict(family=fam, mechanism="unexplained", contribution=float(np.mean(g["_y"] - g["_fit"])),
                         circuits=len(g)))
    meta = dict(predictor=predictor, method=method, response=response, r2=r["r2"], cond=r["cond"],
                circuits=r["circuits"], weighted=bool(weighted), mechanisms=names)
    return coef, pd.DataFrame(rows), meta


def qubit_table(D: pd.DataFrame, plan, predictor: str = PRIMARY_PREDICTOR, method: str = "RAW",
                response: str = "ex_ln_nf", boot: int = 300) -> pd.DataFrame:
    """Unmodeled log-infidelity per two-qubit-gate participation of every physical qubit (qemx.context_noise.
    qubit_attribution): where on the chip the excess lives. Needs circuits on different qubits (layout variants)."""
    from qemx.context_noise import qubit_attribution
    d = D[(D["predictor"] == predictor) & (D["method"] == method) & (D["status"] == "ok")]
    if "variant" in d.columns:
        d = d[d["variant"] >= 0]
    d = d[np.isfinite(pd.to_numeric(d[response], errors="coerce"))]
    if d.empty:
        return pd.DataFrame()
    loads = []
    for key in d["instance"]:
        inst = plan.instance(key)
        L = {}
        for o in inst.spec.ops:
            if o.kind == "gate" and len(o.qubits) == 2:
                for q in o.qubits:
                    L[int(inst.phys[q])] = L.get(int(inst.phys[q]), 0) + 1
        loads.append(L)
    res = qubit_attribution(loads, d[response].astype(float).to_numpy(), boot=boot,
                            weights=1 / response_se(d, response))
    return pd.DataFrame([dict(qubit=q, coef=v[0], lo=v[1], hi=v[2], circuits=v[3]) for q, v in sorted(res.items())])


def fingerprint_table(plan, counts: dict, predicted: dict, bootstrap=None) -> pd.DataFrame:
    """The folding fingerprint (qemx.divergence.fold_response) as a table, one row per (circuit, ZNE method,
    predictor), and the global/local pairs side by side (``pairs``)."""
    from qemx.divergence import fold_response
    return pd.DataFrame(fold_response(plan, counts, predicted, bootstrap=bootstrap))


def fingerprint_pairs(FP: pd.DataFrame, global_key: str = "ZNE-G", local_key: str = "ZNE-L") -> pd.DataFrame:
    """A_global and A_local of every circuit and predictor in one row (the two foldings with the same scales)."""
    if FP.empty:
        return pd.DataFrame()
    keep = ["instance", "group", "family", "label", "width", "two_qubit", "predictor"]
    per = ["A", "A_lo", "A_hi", "kappa_obs", "kappa_obs_se", "kappa_pred", "z_kappa"]
    g = FP[FP["method"] == global_key][keep + [c for c in per if c in FP.columns]]
    loc = FP[FP["method"] == local_key][["instance", "predictor"] + [c for c in per if c in FP.columns]]
    return g.merge(loc, on=["instance", "predictor"], suffixes=("_global", "_local"))


def fingerprint_summary(pairs: pd.DataFrame, predictor: str = PRIMARY_PREDICTOR) -> pd.DataFrame:
    """Per family (and all circuits): the pooled anomalies of global and local folding with their SEs and the χ²/dof
    of the common-A fit (qemx.divergence.pooled_anomaly), the circuits whose κ differs significantly from the model's (|z_κ| > Z_SIG),
    and the reading (:func:`fingerprint_reading` at Z_SIG standard errors: one test per family)."""
    from qemx.divergence import pooled_anomaly
    p = pairs[pairs["predictor"] == predictor] if len(pairs) else pairs
    if p.empty:
        return pd.DataFrame()
    out = []
    for fam, g in list(p.groupby("family")) + [("all", p)]:
        ag, sg, cg = pooled_anomaly(g["kappa_obs_global"], g["kappa_pred_global"], g.get("kappa_obs_se_global"))
        al, sl, cl = pooled_anomaly(g["kappa_obs_local"], g["kappa_pred_local"], g.get("kappa_obs_se_local"))
        out.append(dict(family=fam, circuits=len(g), A_global=ag, A_global_se=sg, A_local=al, A_local_se=sl,
                        chi2_global=cg, chi2_local=cl,
                        sig_global=int((g["z_kappa_global"].abs() > Z_SIG).sum()),
                        sig_local=int((g["z_kappa_local"].abs() > Z_SIG).sum()),
                        reading=fingerprint_reading(ag, al, sg, sl, k=Z_SIG)))   # many families: 3 SE
    return pd.DataFrame(out)


def fingerprint_reading(a_global: float, a_local: float, se_global: float = 0.0, se_local: float = 0.0,
                        k: float = 2.0) -> str:
    """The mechanism class the two pooled anomalies point to, at k standard errors:

    as modeled           neither differs from 1
    gate-attached        both above 1, not distinguishable from each other
    time-attached        A_global − A_local > k·SE (global folding also stretches idle windows)
    repetition-coherent  A_local − A_global > k·SE (repeating the same gate builds up a coherent error)
    over-modeled         both below 1 (the model is too pessimistic)
    inconclusive         anything else (too few shots or circuits to tell)"""
    if not (math.isfinite(a_global) and math.isfinite(a_local)):
        return ""
    sg = se_global if (se_global is not None and math.isfinite(se_global)) else 0.0
    sl = se_local if (se_local is not None and math.isfinite(se_local)) else 0.0
    sd = math.hypot(sg, sl)
    up_g, up_l = a_global - 1 > k * sg, a_local - 1 > k * sl
    dn_g, dn_l = 1 - a_global > k * sg, 1 - a_local > k * sl
    if not (up_g or up_l or dn_g or dn_l):
        return "as modeled"
    if a_global - a_local > k * sd and up_g:
        return "time-attached"
    if a_local - a_global > k * sd and up_l:
        return "repetition-coherent"
    if up_g and up_l:
        return "gate-attached"
    if dn_g and dn_l:
        return "over-modeled"
    return "inconclusive"


def context_table(D: pd.DataFrame, predictor: str = PRIMARY_PREDICTOR, method: str = "RAW") -> pd.DataFrame:
    """The spectator-context probes (family ``context``): observed and predicted fidelity per layer count and
    spectator mode, and the probe contrasts against the idle-spectator (``zero``) probe of the same core — the
    component model predicts almost no contrast, so an observed one is spectator or drive crosstalk."""
    d = D[(D["family"] == "context") & (D["predictor"] == predictor) & (D["method"] == method)
          & (D["status"] == "ok")]
    if d.empty:
        return pd.DataFrame()
    P = d["params"].map(json.loads)
    d = d.assign(layers=P.map(lambda p: p.get("layers")), spectators=P.map(lambda p: p.get("spectators")),
                 core_seed=P.map(lambda p: p.get("seed", 0)))
    key = ["layers", "core_seed"] + (["variant"] if "variant" in d.columns else [])
    zero = d[d["spectators"] == "zero"].set_index(key)[["fc_obs", "fc_pred", "fc_obs_se"]]
    d = d.join(zero, on=key, rsuffix="_zero")
    d["contrast_obs"] = d["fc_obs_zero"] - d["fc_obs"]
    d["contrast_pred"] = d["fc_pred_zero"] - d["fc_pred"]
    d["contrast_excess"] = d["contrast_obs"] - d["contrast_pred"]
    # shot noise of the observed contrast (the predicted one is a replicate mean: exact by comparison)
    d["contrast_excess_se"] = np.sqrt(d["fc_obs_se"].astype(float) ** 2 + d["fc_obs_se_zero"].astype(float) ** 2)
    d["contrast_z"] = d["contrast_excess"] / d["contrast_excess_se"].where(d["contrast_excess_se"] > 0)
    keep = ["label", "variant", "layers", "spectators", "fc_obs", "fc_pred", "delta", "z", "contrast_obs",
            "contrast_pred", "contrast_excess", "contrast_excess_se", "contrast_z", "f_parallel_adjacent",
            "f_active_spectators"]
    return d[[c for c in keep if c in d.columns]].sort_values(["layers", "spectators"]).reset_index(drop=True)


def analyse(plan, counts: dict, predicted: dict, *, errors=(), replicates=None, bootstrap=None, seed: int = 0,
            source: str = "observed", progress=None) -> Report:
    """Compare and summarize (everything in memory). ``predicted``: {predictor: {run key: distribution}}."""
    from qemx.divergence import compare
    rep = Report()
    rep.section("DIVERGENCE · %s against component-model predictions (%d circuits, methods %s, budget %d shots each)"
                % (source, len(plan.instances), ", ".join(m.key for m in plan.suite.methods), plan.suite.budget))
    D = divergence_table(compare(plan, counts, predicted, replicates=replicates, bootstrap=bootstrap, seed=seed,
                                 source=source, progress=progress))
    rep.table("divergence", D)
    E = pd.DataFrame(list(errors) + list(plan.errors))
    if len(E):
        rep.table("errors", E)
        rep.text("skipped or failed (full list: errors.csv):")
        rep.frame(error_summary(E))
    if D.empty or "delta" not in D.columns:
        rep.verdict("divergence", None, "no comparable rows (missing counts or predictions)")
        return rep
    ok = D[D["status"] == "ok"]
    rep.text("Δ = predicted − observed Fc at the same shots (> 0: the device is worse than its model); z = Δ / pooled "
             "SD (the model's replicates and the observation's bootstrap); |z| > %g is beyond the shot noise of the "
             "budget." % Z_SIG)
    g = ok.groupby(["predictor", "method"]).agg(circuits=("instance", "nunique"), mean_delta=("delta", "mean"),
                                                median_z=("z", "median"),
                                                worse=("z", lambda z: float((z > Z_SIG).mean())),
                                                better=("z", lambda z: float((z < -Z_SIG).mean()))).reset_index()
    rep.table("overview", g, show=True)
    S = structure_summary(ok)
    rep.table("structures", S)
    rep.section("STRUCTURES · families ordered by mean Δ under the %s model" % PRIMARY_PREDICTOR)
    if len(S):
        rep.frame(S[S["predictor"] == PRIMARY_PREDICTOR].round(4), max_rows=60)
    Tt = top_divergent(ok)
    rep.table("top", Tt)
    if len(Tt):
        rep.text("most divergent circuits (z > %g):" % Z_SIG)
        rep.frame(Tt.round(4))
    R = rates_table(ok)
    rep.table("rates", R)
    if len(R):
        rep.section("EFFECTIVE TWO-QUBIT ERROR RATE · fitted to Fc against the post-compilation two-qubit count")
        rep.frame(R.round(5))
    # ---- context: what the per-gate model cannot see, and where it comes from
    C = context_table(ok)
    rep.table("context", C)
    if len(C):
        rep.section("CONTEXT PROBES · the same core with idle, superposed or driven spectators (%s model, RAW)"
                    % PRIMARY_PREDICTOR)
        rep.text("contrast = Fc(zero) − Fc(mode); excess = observed contrast − predicted contrast (> 0: crosstalk the "
                 "component model does not have)")
        rep.frame(C.round(4))
    raw_key = next((m.key for m in plan.suite.methods if m.kind == "raw"), None)
    bkey = budget_method(ok, plan)
    if bkey:
        L, verdict = model_ladder(ok, method=bkey)
        rep.table("model_ladder", L)
        if len(L):
            rep.section("WHAT THE DIVERGENCE NEEDS · nested explanations of the excess log-infidelity, %s, %s model"
                        % (bkey, PRIMARY_PREDICTOR))
            rep.text("weighted fits (1/SE); a model that suffices has χ²/dof ≈ 1. Verdict: " + verdict)
            rep.frame(L.round(4))
            rep.verdict("divergence-explanation", None, verdict)
        coef, budget, meta = attribution(ok, method=bkey)
        rep.table("budget_coefficients", coef)
        rep.table("budget_families", budget)
        if len(coef):
            rep.section("DIVERGENCE BUDGET · excess log-infidelity (normalized fidelity) by mechanism, %s, %s model"
                        % (bkey, PRIMARY_PREDICTOR))
            rep.text("non-negative %s least squares over %d circuits, R² = %.3f, condition number %.1f; coefficients "
                     "per unit with bootstrap 95 %% intervals; VIF > 10: not separable from the other mechanisms "
                     "with these circuits" % ("weighted (1/SE)" if meta.get("weighted") else "", meta["circuits"],
                                              meta["r2"], meta["cond"]))
            rep.frame(coef.drop(columns=["description"]).round(5))
        Q = qubit_table(ok, plan, method=bkey)
        rep.table("qubits", Q)
    agg = ok[ok["method"].astype(str).str.contains("·")] if "method" in ok else ok.head(0)
    if len(agg):
        rep.section("VARIANTS · simple aggregation and plurality vote (Chen et al.), heterogeneity across variants")
        cols = ["label", "method", "predictor", "fc_obs", "fc_pred", "delta", "z", "het_Q", "het_p", "het_I2"]
        rep.frame(agg[agg["predictor"] == PRIMARY_PREDICTOR][[c for c in cols if c in agg.columns]].round(4))
        rep.text("het_p < 0.01: the variants' divergences differ by more than shot noise — the unmodeled noise depends "
                 "on where (layouts) or how (twirls, seeds) the circuit runs")
    FP = fingerprint_table(plan, counts, predicted, bootstrap=bootstrap)
    rep.table("fingerprint", FP)
    pairs = fingerprint_pairs(FP)
    rep.table("fingerprint_pairs", pairs)
    if len(pairs):
        rep.section("FOLDING FINGERPRINT · amplification anomaly A = κ_obs / κ_pred under global and local folding")
        rep.text("A ≈ 1: the device answers folding as its model does. A_local ≈ A_global > 1: unmodeled noise on the "
                 "two-qubit gates; A_global > A_local: attached to time (idle windows); A_local > A_global: coherent "
                 "build-up under repetition of the same gate")
        rep.text("Per family: A from the weighted fit κ_obs = A·κ_pred (κ_pred: the model's replicates at the same "
                 "shots; χ²/dof ≫ 1: the circuits do not share one A); sig = circuits with |z_κ| > %d" % Z_SIG)
        S = fingerprint_summary(pairs)
        rep.table("fingerprint_summary", S)
        rep.frame(S.round(3))
    return rep


# ------------------------------------------------------------------ input / output
def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def load_plan(run_dir: str):
    from qemx.divergence import Plan
    return Plan.from_json(_read_json(os.path.join(run_dir, "plan.json")))


def obs_dir(run_dir: str, tag: str) -> str:
    if not tag or any(c in tag for c in "/\\") or tag.startswith("."):
        raise SystemExit("--tag must be a plain name")
    return os.path.join(run_dir, "obs", tag)


def load_models(d: str) -> dict:
    return _read_json(os.path.join(d, "calibration.json"))["models"]


def connect(channel=None, instance=None, token_env="QISKIT_IBM_TOKEN", ask=False):
    """A QiskitRuntimeService. The token comes from ``token_env``, a hidden prompt (``ask``) or a saved account; it is
    held only in this function's local variable and masked in error messages."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get(token_env) if token_env else None
    if not token and ask:
        token = getpass.getpass("IBM Quantum API key (input hidden; Enter: use the saved account): ") or None
    kw = {k: v for k, v in (("channel", channel), ("instance", instance)) if v}
    try:
        return QiskitRuntimeService(token=token, **kw) if token else QiskitRuntimeService(**kw)
    except Exception as e:                               # noqa: BLE001 - reported without the token
        msg = str(e)
        if token:
            msg = msg.replace(token, "***")
        raise SystemExit("could not connect to IBM Quantum (%s): %s" % (type(e).__name__, msg[:300])) from None
    finally:
        token = None                                     # noqa: F841 - drop the reference


def backend_at(backend, when: str | None):
    """(backend-like object, description): the backend with its calibration at ``when`` (ISO time) when the service
    keeps the history (``target_history``/``properties(datetime=…)``), else the backend as it is now."""
    from types import SimpleNamespace
    if not when:
        return backend, "calibration when collected (no execution time reported)"
    try:
        t = _dt.datetime.fromisoformat(str(when).replace("Z", "+00:00"))
        tg = backend.target_history(datetime=t)
        props = backend.properties(datetime=t)
    except Exception:                                    # noqa: BLE001 - history is optional
        return backend, "calibration when collected (history unavailable)"
    qp = getattr(tg, "qubit_properties", None)
    return SimpleNamespace(name=backend.name, dt=backend.dt, num_qubits=backend.num_qubits, target=tg,
                           qubit_properties=lambda q: qp[q] if qp else None,
                           properties=lambda: props), "calibration at execution (%s)" % when


def _progress(label):
    def f(x, what=""):
        sys.stderr.write("\r%s %5.1f %% %-40s" % (label, 100 * x, str(what)[:40]))
        if x >= 1:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return f


def _backend(name: str):
    from qemx.device import FAKE_DEVICES, fake_backend
    return fake_backend(name) if name in FAKE_DEVICES else None


def cmd_plan(a):
    import yaml
    from qemx.divergence import make_plan, suite_from_manifest
    with open(a.manifest) as f:
        m = yaml.safe_load(f)
    suite = suite_from_manifest(m)
    name = a.backend or suite.fake
    b = _backend(name)
    kind = "fake"
    if b is None:
        service = connect(a.channel, a.instance, a.token_env, a.ask_token)
        b = service.backend(name)
        kind = "ibm"
    plan = make_plan(suite, b, kind, progress=_progress("plan"))
    if not plan.instances:
        for e in plan.errors:
            print("not planned: %s — %s" % (e["instance"], e["error"]))
        raise SystemExit("no benchmark could be planned; nothing written")
    os.makedirs(a.out, exist_ok=True)
    _write_json(os.path.join(a.out, "plan.json"), plan.to_json())
    shutil.copyfile(a.manifest, os.path.join(a.out, "manifest.yaml"))
    s = plan.summary()
    print(json.dumps(s, indent=1))
    if plan.errors:                                      # grouped; every one is kept in plan.json ("errors")
        with pd.option_context("display.width", 200, "display.max_colwidth", 90):
            print("skipped or not planned (all of them: plan.json, \"errors\"):")
            print(error_summary(pd.DataFrame(plan.errors)).to_string(index=False))
    print("compiled on %s (%s); %d circuits, %d shots in total (%d per benchmark and method%s)"
          % (plan.backend["name"], kind, s["runs"], s["shots"], plan.suite.budget,
             ", shared by %d variants" % plan.suite.variants if plan.suite.variants > 1 else ""))
    return 0


def _check_backend(plan, b):
    if plan.backend["kind"] == "fake" and getattr(b, "name", "") != plan.backend["name"]:
        print("note: the plan was compiled on %s; submitting to %s. The ISA check below must pass (same coupling map "
              "and gates); for layouts on today's calibration, plan again with --backend %s"
              % (plan.backend["name"], b.name, b.name))


def cmd_run_ibm(a):
    from qemx.divergence import snapshot, submit
    from qemx.hardware import SubmitError
    plan = load_plan(a.run_dir)
    from qemx import campaign as CG
    if not a.ledger or not os.path.exists(a.ledger):
        raise SystemExit("hardware submission needs the campaign ledger (one shared QPU ceiling for the whole thesis): "
                         "create it once with `python -m qemx.campaign init --ledger %s --ceiling 600 --reserve 60`, "
                         "review `python -m qemx.campaign preflight %s --ledger %s`, then run again with --ledger"
                         % (a.ledger or "campaign_ledger.json", a.run_dir, a.ledger or "campaign_ledger.json"))
    ledger = CG.Ledger.load(a.ledger)
    if ledger.unsubmitted_reservations():
        raise SystemExit("the ledger holds reservations without job ids (an interrupted submission): check the "
                         "provider's job list and reconcile them (python -m qemx.campaign reconcile …) first")
    model = CG.UsageModel(per_shot_s=a.per_shot_ms / 1e3, margin=a.margin)
    service = connect(a.channel, a.instance, a.token_env, a.ask_token)
    name = a.backend or plan.suite.device
    b = service.backend(name)
    _check_backend(plan, b)
    d = obs_dir(a.run_dir, a.tag)
    jp = os.path.join(d, "jobs.json")
    earlier, skip = [], set()
    if os.path.exists(jp):
        old = _read_json(jp)
        if a.resume and old.get("partial"):
            earlier = list(old["records"])               # submit only the runs these jobs do not cover
            skip = {k for r in earlier for k in r["runs"]}
        elif a.force:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            os.replace(jp, os.path.join(d, "jobs.replaced-%s.json" % stamp))    # the old job ids are kept
        else:
            raise SystemExit("%s already has %s submission (--resume submits the runs it lacks; --force submits "
                             "everything again, keeping the old job ids in jobs.replaced-*.json; --tag starts a new "
                             "set)" % (d, "a partial" if old.get("partial") else "a complete"))
    models = snapshot(plan, b)
    if not earlier:
        meta = dict(kind="hardware", backend=b.name, calibration="at submission",
                    created=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        _write_json(os.path.join(d, "calibration.json"), dict(source="at submission", models=models))
        _write_json(os.path.join(d, "meta.json"), meta)
    try:
        def keep(recs):                                   # jobs.json is current after every single job
            _write_json(jp, dict(backend=b.name, records=earlier + recs, partial=True))
        records = submit(plan, b, a.max_execution_time, skip=skip, ledger=ledger, model=model, mode=a.mode,
                         campaign=plan.suite.study, on_record=keep)
    except CG.CampaignLimit as e:
        raise SystemExit("not submitted: %s" % e)
    except KeyboardInterrupt:
        raise SystemExit("interrupted: the job ids submitted so far are in %s (partial); after checking the provider's "
                         "job list, continue with run-ibm --resume" % jp)
    except SubmitError as e:
        _write_json(jp, dict(backend=b.name, records=earlier + e.records, partial=True))
        raise SystemExit("submission stopped after %d job(s): %s — their ids are in jobs.json; submit the rest with "
                         "run-ibm --resume (collect-ibm can collect at any time)" % (len(earlier + e.records), e))
    _write_json(jp, dict(backend=b.name, records=earlier + records, partial=False))
    print("submitted %d job(s) to %s%s; collect with: python scripts/dirA_divergence.py collect-ibm %s --tag %s"
          % (len(records), b.name, " (resumed: %d earlier)" % len(earlier) if earlier else "", a.run_dir, a.tag))
    return 0


def cmd_collect_ibm(a):
    from qemx.divergence import collect, snapshot
    plan = load_plan(a.run_dir)
    d = obs_dir(a.run_dir, a.tag)
    jobs = _read_json(os.path.join(d, "jobs.json"))
    service = connect(a.channel, a.instance, a.token_env, a.ask_token)
    ledger = None
    if getattr(a, "ledger", None) and os.path.exists(a.ledger):
        from qemx import campaign as CG
        ledger = CG.Ledger.load(a.ledger)
    counts, status = collect(service, jobs["records"], wait=a.wait, ledger=ledger)
    if ledger is not None:
        print("campaign ledger: %.1f s committed, %.1f s remaining" % (ledger.committed_s(), ledger.remaining_s()))
    for s in status:
        print("%s %s%s" % (s["job_id"], s["status"], " (failed)" if s["failed"] else ""))
    if counts is None:
        print("not finished yet; run collect-ibm again (or --wait)")
        return 1
    b = service.backend(jobs["backend"])
    when = next((s.get("executed") for s in status if s.get("executed")), None)
    bb, how = backend_at(b, when)
    _write_json(os.path.join(d, "counts.json"), counts)
    _write_json(os.path.join(d, "calibration.json"), dict(source=how, models=snapshot(plan, bb)))
    meta = _read_json(os.path.join(d, "meta.json")) if os.path.exists(os.path.join(d, "meta.json")) else {}
    meta.update(collected=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), calibration=how,
                jobs=status)
    _write_json(os.path.join(d, "meta.json"), meta)
    print("counts of %d runs written; %s" % (len(counts), how))
    return 0


def cmd_run_local(a):
    from qemx.divergence import dm_from_json, run_local, snapshot
    plan = load_plan(a.run_dir)
    name = a.fake or plan.suite.fake
    b = _backend(name)
    if b is None:
        raise SystemExit("%s is not a fake backend" % name)
    models = snapshot(plan, b)
    d = obs_dir(a.run_dir, a.tag)
    try:
        counts = run_local(plan, b, {k: dm_from_json(v) for k, v in models.items()}, seed=a.seed,
                           progress=_progress("local"))
    except MemoryError:
        raise SystemExit("out of memory in the local simulation (no size limit is imposed): remove the widest "
                         "circuits from the manifest or use emulate with aer_dm_max_qubits lowered") from None
    _write_json(os.path.join(d, "counts.json"), counts)
    _write_json(os.path.join(d, "calibration.json"), dict(source="fake snapshot %s" % name, models=models))
    _write_json(os.path.join(d, "meta.json"), dict(kind="local", backend=name, seed=a.seed,
                                                   note="Qiskit local testing mode: gate errors (depolarizing + "
                                                        "relaxation during gates) and readout; no idle relaxation"))
    print("local run on %s: %d runs" % (name, len(counts)))
    return 0


def cmd_emulate(a):
    from qemx.divergence import emulate, parse_injection, snapshot
    plan = load_plan(a.run_dir)
    b = _backend(a.fake or plan.suite.fake)
    if b is None:
        raise SystemExit("emulation needs a fake backend for the calibration snapshot")
    models = snapshot(plan, b)
    inj = parse_injection(a.inject)
    counts, errors = emulate(plan, models, a.truth, inj, seed=a.seed, progress=_progress("emulate"),
                             workers=a.workers)
    d = obs_dir(a.run_dir, a.tag)
    _write_json(os.path.join(d, "counts.json"), counts)
    _write_json(os.path.join(d, "calibration.json"), dict(source="fake snapshot %s" % b.name, models=models))
    _write_json(os.path.join(d, "meta.json"), dict(kind="emulated", backend=b.name, truth=a.truth, inject=inj,
                                                   seed=a.seed, errors=errors))
    for e in errors:
        print("not emulated: %s — %s" % (e["instance"], e["error"]))
    print("emulated %s + %s: %d runs (EMULATED data, not hardware)" % (a.truth, inj or "nothing", len(counts)))
    return 0


def _predictions(plan, d, models, kinds, refresh=False, workers=1):
    """Cached predictions of an observation set (they depend on the plan and its calibration only)."""
    import hashlib
    from qemx.divergence import predict
    path, meta_p = os.path.join(d, "predictions.npz"), os.path.join(d, "predictions.json")
    key = dict(calibration=hashlib.sha256(json.dumps(models, sort_keys=True).encode()).hexdigest(),
               runs=hashlib.sha256("".join(r.digest for r in plan.runs).encode()).hexdigest(),
               kinds=list(kinds), exact_max=plan.suite.exact_max_qubits, dm_max=plan.suite.aer_dm_max_qubits,
               mc_shots=plan.suite.mc_shots)
    if not refresh and os.path.exists(path) and os.path.exists(meta_p):
        meta = _read_json(meta_p)
        if meta.get("key") == key:
            z = np.load(path)
            out = {k: {} for k in kinds}
            for i, (k, rk) in enumerate(meta["index"]):
                out[k][rk] = z["a%d" % i]
            return out, meta.get("errors", [])
    out, errors = predict(plan, models, kinds, progress=_progress("predict"), workers=workers)
    index = [(k, rk) for k, d_ in out.items() for rk in d_]
    np.savez_compressed(path, **{"a%d" % i: out[k][rk] for i, (k, rk) in enumerate(index)})
    _write_json(meta_p, dict(key=key, errors=errors, index=index))
    return out, errors


def cmd_analyze(a):
    plan = load_plan(a.run_dir)
    d = obs_dir(a.run_dir, a.tag)
    counts = _read_json(os.path.join(d, "counts.json"))
    models = load_models(d)
    meta = _read_json(os.path.join(d, "meta.json")) if os.path.exists(os.path.join(d, "meta.json")) else {}
    kinds = tuple(a.predictors or plan.suite.predictors)
    pred, perr = _predictions(plan, d, models, kinds, a.refresh, a.workers)
    source = "%s (%s)" % (meta.get("backend", "?"), meta.get("kind", "?"))
    rep = analyse(plan, counts, pred, errors=perr, replicates=a.replicates, bootstrap=a.bootstrap, seed=a.seed,
                  source=source, progress=_progress("compare"))
    if meta.get("kind") == "emulated":
        rep.verdict("emulated", None, "these observations are EMULATED (%s + %s), not hardware data"
                    % (meta.get("truth"), meta.get("inject")))
    DirectoryReportSink(os.path.join(a.out, "tables")).write(rep)
    if not a.no_figures and "divergence" in rep.tables and "delta" in rep.tables["divergence"].columns:
        F.setup_matplotlib()
        sink = DirectoryFigureSink(a.out, a.formats)
        make_figures(rep.tables["divergence"], sink, kinds, title=source)
        make_context_figures(rep, sink, plan, title=source)
    sys.stdout.write(rep.render_text())
    print("tables and report written to %s" % os.path.join(a.out, "tables"))
    return 0


def make_figures(D: pd.DataFrame, sink, predictors=(PRIMARY_PREDICTOR,), title: str = "", x: str = "two_qubit",
                 value: str = "delta", per_method: bool = True) -> list:
    """The gap figures: the fidelity gap of the unmitigated circuits (observed and predicted F_c with their effective
    error rates), the difference volumetric plots (all methods on one scale; each method alone when ``per_method``)
    and predicted-vs-observed. ``x``: two_qubit (default, Chen et al.), depth, norm_depth (Lubinski et al.) or
    two_qubit_executed; ``value``: delta (F_c) or delta_nf (normalized fidelity). Files carry the axis and value in
    their names unless default."""
    saved = []
    ok = D[D["status"] == "ok"]
    sfx = ("" if x == "two_qubit" else "_" + x) + ("" if value == "delta" else "_nf")
    for pr in predictors:
        d = ok[ok["predictor"] == pr]
        if d.empty:
            continue
        methods = [m for m in dict.fromkeys(d["method"]) if "·" not in str(m)]     # per circuit; aggregates below
        aggs = [m for m in dict.fromkeys(d["method"]) if "·" in str(m)]
        if not sfx and "RAW" in methods:
            dd = d[d["variant"] >= 0] if "variant" in d.columns else d
            saved.append(sink.save(F.fidelity_gap(dd, "RAW", pr, title=title), "divergence_gap_%s" % pr))
        saved.append(sink.save(F.difference_volumetric_grid(d, methods, pr, x=x, value=value, title=title),
                               "divergence_volumetric_%s%s" % (pr, sfx)))
        if aggs:
            saved.append(sink.save(F.difference_volumetric_grid(d, aggs, pr, x=x, value=value,
                                                                title=(title + " · variant aggregates").strip(" ·")),
                                   "divergence_volumetric_%s_aggregates%s" % (pr, sfx)))
        if per_method:
            for m in methods:
                saved.append(sink.save(F.difference_volumetric(d, m, pr, x=x, value=value, title=m, circuits=True),
                                       "divergence_volumetric_%s_%s%s" % (pr, F._slug(m), sfx)))
        if not sfx:
            saved.append(sink.save(F.predicted_vs_observed(d, methods, pr, title=title), "divergence_scatter_%s" % pr))
    return saved


def make_context_figures(rep: Report, sink, plan=None, title: str = "") -> list:
    """The figures of the context analyses (tables of :func:`analyse`): divergence budget, folding fingerprint,
    heavy-hex map of the per-qubit excess, spectator-context probes, excess error per two-qubit gate."""
    saved = []
    T = rep.tables
    L = T.get("model_ladder")
    if L is not None and len(L):
        saved.append(sink.save(F.ladder_chart(L, title=title, verdict=ladder_verdict(L)), "divergence_ladder"))
    coef, budget = T.get("budget_coefficients"), T.get("budget_families")
    if coef is not None and len(coef):
        saved.append(sink.save(F.divergence_budget(coef, budget, title=title), "divergence_budget"))
    pairs = T.get("fingerprint_pairs")
    if pairs is not None and len(pairs):
        for pr in pairs["predictor"].unique():
            saved.append(sink.save(F.folding_fingerprint(pairs[pairs["predictor"] == pr], title="%s model%s" % (
                pr, (" · " + title) if title else ""), summary=fingerprint_summary(pairs, pr)),
                "divergence_fingerprint_%s" % pr))
    Q = T.get("qubits")
    if Q is not None and len(Q) and plan is not None and plan.backend.get("edges"):
        saved.append(sink.save(F.heavy_hex_map(Q, plan.backend["edges"], plan.backend["num_qubits"],
                                               title=plan.backend["name"] + (" · " + title if title else "")),
                               "divergence_heavy_hex"))
    C = T.get("context")
    if C is not None and len(C):
        saved.append(sink.save(F.context_probe(C, title=title), "divergence_context_probes"))
    D = T.get("divergence")
    if D is not None and "ex_nf_per_2q" in D.columns:
        d = D[(D["status"] == "ok") & (D["predictor"] == PRIMARY_PREDICTOR)]
        if "variant" in d.columns:
            d = d[d["variant"] >= 0]
        methods = [m for m in dict.fromkeys(d["method"]) if "·" not in str(m)]
        if methods:
            saved.append(sink.save(F.difference_volumetric_grid(d, methods, PRIMARY_PREDICTOR, value="ex_nf_per_2q",
                                                                title=title),
                                   "divergence_volumetric_excess_per_gate"))
    return saved


# ------------------------------------------------------------------ recovery: does each tool name a known cause?
def _short_verdict(v: str) -> str:
    if not v:
        return ""
    head = v.split(":")[0]
    if head == "rescaled calibration":
        k = v.split("×")[1].split(" ")[0] if "×" in v else "?"
        out = "rescaled calibration ×%s" % k
    else:
        out = head
    return out + (" + non-linear" if "beyond the linear budget" in v else "")


def recovery_row(tables: str, label: str, predictor: str = PRIMARY_PREDICTOR, min_share: float = 0.15) -> dict:
    """One injected (or real) data set read back through every diagnostic, from the tables an ``analyze`` wrote:
    the χ² ladder's verdict, the budget's largest shares, the folding fingerprint over all circuits, the context
    probes' mean excess contrast per spectator mode with its combined z (Stouffer over every core depth and layout
    variant), the benchmarks whose layout variants disagree (Cochran p < 0.01) with the worst qubits of the heavy-hex
    map, and ε_eff(obs)/ε_eff(pred)."""
    def read(name):
        path = os.path.join(tables, name + ".csv")
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    row = dict(data=label)
    L = read("model_ladder")
    row["ladder"] = _short_verdict(ladder_verdict(L)) if len(L) else ""
    B = read("budget_coefficients")
    if row["ladder"] == "as modeled":
        row["budget"] = "— (nothing to attribute)"
    elif len(B):
        B = B[np.isfinite(B["share"])].sort_values("share", ascending=False)
        row["budget"] = ", ".join("%s %.0f%%" % (F.MECHANISM_LABEL.get(m, m), 100 * s_)
                                  for m, s_ in zip(B["mechanism"], B["share"]) if s_ >= min_share)
    S = read("fingerprint_summary")
    if len(S) and (S["family"] == "all").any():
        a = S[S["family"] == "all"].iloc[0]
        row["fingerprint"] = "%s (A_g %.2f±%.2f, A_l %.2f±%.2f)" % (a["reading"], a["A_global"], a["A_global_se"],
                                                                   a["A_local"], a["A_local_se"])
        row["fingerprint_reading"] = a["reading"]
    C = read("context")
    if len(C):
        for mode in ("plus", "driven"):
            g = C[C["spectators"] == mode]
            if not len(g):
                continue
            row["probe_%s" % mode] = float(g["contrast_excess"].astype(float).mean())
            if "contrast_z" in g and np.isfinite(g["contrast_z"].astype(float)).any():
                zz = g["contrast_z"].astype(float).dropna()          # Stouffer over every core depth and variant
                row["probe_%s_z" % mode] = float(zz.sum() / math.sqrt(len(zz)))
    D = read("divergence")
    if len(D) and "het_p" in D.columns and "variant" in D.columns:
        g = D[(D["variant"] < 0) & (D["predictor"] == predictor)].drop_duplicates("group")
        g = g[np.isfinite(g["het_p"].astype(float))]
        if len(g):
            row["het_groups"] = int(len(g))
            row["het_sig"] = int((g["het_p"].astype(float) < 0.01).sum())
    Q = read("qubits")
    if len(Q):
        top = Q[Q["coef"] > 0].sort_values("coef", ascending=False).head(2)
        row["worst_qubits"] = ", ".join(str(int(q)) for q in top["qubit"])
    R = read("rates")
    if len(R):
        r = R[(R["method"] == "RAW") & (R["predictor"] == predictor)]
        if len(r):
            row["eps_ratio"] = float(r["ratio"].iloc[0])
    return row


def recovery_table(items) -> pd.DataFrame:
    """[(label, analyze output dir)] → one row per data set (:func:`recovery_row`)."""
    return pd.DataFrame([recovery_row(os.path.join(d, "tables") if os.path.isdir(os.path.join(d, "tables")) else d,
                                      lab) for lab, d in items])


def cmd_recovery(a):
    """The diagnostic matrix of several analyses (e.g. emulations with known injected errors, and the hardware
    run): does each tool name the cause?"""
    items = []
    for it in a.items:
        if "=" not in it:
            raise SystemExit("recovery: give LABEL=DIRECTORY (an analyze --out directory)")
        lab, _, d = it.rpartition("=")
        items.append((lab, d))
    R = recovery_table(items)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "recovery.csv"), index=False)
    F.setup_matplotlib()
    saved = DirectoryFigureSink(a.out, a.formats).save(F.diagnostic_matrix(R, title=a.title), "divergence_recovery")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(R.to_string(index=False))
    print("written: %s, %s" % (os.path.join(a.out, "recovery.csv"), saved))
    return 0


def cmd_refresh_features(a):
    """Recompute the context features of every instance of an existing plan (plans made before 1.4.1 have none, or
    lack gate_error / readout_error) on the plan's fake twin, or on --backend. The circuits, runs and digests do not
    change; the old plan is kept as plan.before-features.json."""
    from qemx.divergence import instance_features
    plan = load_plan(a.run_dir)
    name = a.backend or plan.suite.fake
    b = _backend(name)
    if b is None:
        b = connect(a.channel, a.instance, a.token_env, a.ask_token).backend(name)
    path = os.path.join(a.run_dir, "plan.json")
    shutil.copyfile(path, os.path.join(a.run_dir, "plan.before-features.json"))
    for inst in plan.instances:
        inst.features = instance_features(inst.spec, inst.phys, b)
    _write_json(path, plan.to_json())
    print("context features of %d instances recomputed on %s" % (len(plan.instances), name))
    return 0


def cmd_figures(a):
    """Redraw from saved tables (no counts or simulation needed): a divergence.csv, or an ``analyze --out``
    directory (all figures, context ones included; ``--run-dir`` adds the heavy-hex map from the run's plan)."""
    F.setup_matplotlib(a.font)
    sink = DirectoryFigureSink(a.out, a.formats)
    tdir = a.table if os.path.isdir(a.table) else None
    if tdir and os.path.isdir(os.path.join(tdir, "tables")):
        tdir = os.path.join(tdir, "tables")
    if tdir:
        rep = Report()
        for f in sorted(os.listdir(tdir)):
            if f.endswith(".csv"):
                rep.tables[f[:-4]] = pd.read_csv(os.path.join(tdir, f))
        D = rep.tables.get("divergence")
    else:
        rep, D = None, pd.read_csv(a.table)
    preds = a.predictors or (list(dict.fromkeys(D["predictor"])) if D is not None else [])
    saved = make_figures(D, sink, preds, a.title, a.x, a.value, per_method=not a.no_per_method) \
        if D is not None else []
    if rep is not None:
        plan = load_plan(a.run_dir) if a.run_dir else None
        saved += make_context_figures(rep, sink, plan, title=a.title)
    print("%d figures written to %s" % (len(saved), a.out))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def conn(p):
        p.add_argument("--channel", default=None, help="ibm_quantum_platform | ibm_cloud (default: the saved account's)")
        p.add_argument("--instance", default=None, help="the service instance (CRN or hub/group/project)")
        p.add_argument("--token-env", default="QISKIT_IBM_TOKEN", help="environment variable holding the token")
        p.add_argument("--ask-token", action="store_true", help="ask for the token (hidden input)")
    p = sub.add_parser("plan")
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default=None, help="compile on this backend (default: the manifest's fake twin)")
    conn(p)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("run-ibm")
    p.add_argument("run_dir")
    p.add_argument("--backend", default=None)
    p.add_argument("--tag", default="ibm")
    p.add_argument("--max-execution-time", type=int, default=None,
                   help="per job, seconds of QPU usage (default: each job's own high usage estimate)")
    p.add_argument("--ledger", default=os.environ.get("QEMX_CAMPAIGN_LEDGER", "campaign_ledger.json"),
                   help="the campaign ledger (python -m qemx.campaign init); required")
    p.add_argument("--mode", choices=("job", "batch"), default="job",
                   help="job (default) or batch; the Open plan's batch TTL is 10 min of elapsed time")
    p.add_argument("--per-shot-ms", type=float, default=0.57, help="usage model: QPU ms per shot (calibrate)")
    p.add_argument("--margin", type=float, default=1.3, help="usage model: high = central × margin")
    p.add_argument("--resume", action="store_true", help="after a partial submission: submit the runs it lacks")
    p.add_argument("--force", action="store_true", help="submit everything again (old job ids are kept)")
    conn(p)
    p.set_defaults(func=cmd_run_ibm)
    p = sub.add_parser("collect-ibm")
    p.add_argument("run_dir")
    p.add_argument("--ledger", default=os.environ.get("QEMX_CAMPAIGN_LEDGER", "campaign_ledger.json"),
                   help="reconcile the jobs' reported QPU usage into this ledger (when it exists)")
    p.add_argument("--tag", default="ibm")
    p.add_argument("--wait", action="store_true")
    conn(p)
    p.set_defaults(func=cmd_collect_ibm)
    p = sub.add_parser("run-local")
    p.add_argument("run_dir")
    p.add_argument("--fake", default=None)
    p.add_argument("--tag", default="local")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_run_local)
    p = sub.add_parser("emulate")
    p.add_argument("run_dir")
    p.add_argument("--fake", default=None)
    p.add_argument("--truth", default="device", choices=("device", "depolarizing", "chen"))
    p.add_argument("--inject", action="append", default=[],
                   help="crosstalk=ζ | scale=s | coherent=χ | t2=µs | hotspot=physical qubit (repeatable)")
    p.add_argument("--tag", default="emulated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="parallel processes (instances)")
    p.set_defaults(func=cmd_emulate)
    p = sub.add_parser("analyze")
    p.add_argument("run_dir")
    p.add_argument("--tag", default="ibm")
    p.add_argument("--out", required=True)
    p.add_argument("--predictors", nargs="+", default=None)
    p.add_argument("--replicates", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refresh", action="store_true", help="recompute the cached predictions")
    p.add_argument("--workers", type=int, default=1, help="parallel processes for the predictions (instances)")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.set_defaults(func=cmd_analyze)
    p = sub.add_parser("refresh-features", help="recompute the context features of an existing plan")
    p.add_argument("run_dir")
    p.add_argument("--backend", default=None, help="default: the plan's fake twin")
    conn(p)
    p.set_defaults(func=cmd_refresh_features)
    p = sub.add_parser("recovery", help="diagnostic matrix of several analyses (LABEL=DIR …)")
    p.add_argument("items", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.set_defaults(func=cmd_recovery)
    p = sub.add_parser("figures", help="redraw the figures from a saved divergence.csv or an analyze directory")
    p.add_argument("table", help="divergence.csv, or an analyze --out directory (all figures)")
    p.add_argument("--run-dir", default=None, help="the run directory (its plan gives the heavy-hex map)")
    p.add_argument("--font", choices=sorted(F.FONT_PRESETS), default=None, help="times (default), lm, sans")
    p.add_argument("--no-per-method", action="store_true", help="skip the one-method volumetric plots")
    p.add_argument("--out", required=True)
    p.add_argument("--predictors", nargs="+", default=None)
    p.add_argument("--x", default="two_qubit", choices=sorted(F.DEPTH_AXES))
    p.add_argument("--value", default="delta", choices=("delta", "delta_nf", "ex_nf_per_2q", "ex_per_2q"))
    p.add_argument("--title", default="")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.set_defaults(func=cmd_figures)
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    try:
        return a.func(a)
    except MemoryError:
        raise SystemExit("out of memory (no size limit is imposed): reduce the widest circuits or the replicates")


if __name__ == "__main__":
    sys.exit(main())
