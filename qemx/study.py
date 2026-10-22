r"""Main-study analysis: empirical performance-regime maps, paired comparisons, cost–accuracy, recommender
evaluation and the link to the discrepancy rehearsal — regenerated from stored records, never by re-simulating.

    python -m qemx.study report data/dirA_v2_pilot --manifest manifests/direction_A_v2_pilot.yaml \
        --rehearsal data/rehearsal_v3 --out reports/pilot

Definitions (every caption repeats the ones it uses):

* **Resource** — the total shot budget B of one estimate: target, training, calibration and every extra circuit
  (folded, neighbor, quasi-probability samples) together; the equal-budget ledger ``qemx.phase.ledger`` allocates
  it optimally inside each method (policy "fixed": the protocol's training shots).
* **Outcome** — RMSE(B) of the estimate against the exact ideal value (a trusted reference exists for every
  simulated cell), from the ledger's exact bias and variance components; learning methods include the Monte-Carlo
  training-level terms (their MC error is small at the recorded replicate counts and is not propagated here).
* **Sample unit** — a cell (base circuit × noise scale × coherent fraction) at one budget; uncertainty across
  circuits is a bootstrap over **base circuits** (every scale of a circuit resampled together).
* **Regime label** of a cell at B: the best feasible method; "tie" when another method is within the
  practical-equivalence margin δ = 10 % in RMSE; "raw ≈ best" when unmitigated is within δ of the best (no
  demonstrated benefit); "no feasible method" when nothing runs at B. Points are drawn only where cells exist:
  there is no interpolation between sampled points ("empirical performance-regime map", not a phase diagram).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from . import features as FT
from . import recommend as RC

DELTA = RC.DELTA
METHOD_ORDER = ("RAW", "ZNE-R", "ZNE-L", "ZNE-E", "PEC", "CDR", "FR-R", "FR-L", "NIL-G2", "VD")
# Okabe–Ito (colorblind-safe), fixed per method; RAW is neutral gray
COLORS = {"RAW": "#7A7A7A", "ZNE-R": "#0072B2", "ZNE-L": "#56B4E9", "ZNE-E": "#003F63", "PEC": "#D55E00",
          "CDR": "#009E73", "FR-R": "#E69F00", "FR-L": "#A66F00", "NIL-G2": "#CC79A7", "VD": "#C4B000"}
MARKERS = {"RAW": "o", "ZNE-R": "s", "ZNE-L": "D", "ZNE-E": "P", "PEC": "^", "CDR": "v", "FR-R": "<", "FR-L": ">",
           "NIL-G2": "X", "VD": "*"}
REGIME_OTHER = {"tie": "#BBBBBB", "raw ≈ best": "#FFFFFF", "no feasible method": "#000000"}


def _order(methods) -> list:
    return [m for m in METHOD_ORDER if m in methods] + sorted(m for m in methods if m not in METHOD_ORDER)


# ------------------------------------------------------------------ data
def load(out_dir: str, manifest_path: str | None = None, policy: str = "fixed") -> dict:
    """Cells (features, split, base circuit) and the ledger (cell × method × B) of a phase-study run directory."""
    from .phase import store as ST
    from .phase.manifest import confirmatory_split, load as load_manifest
    tdir = os.path.join(out_dir, "tables")
    if not os.path.exists(os.path.join(tdir, "ledger.csv")):
        ST.collect(out_dir, manifest_path=manifest_path)
    mpath = manifest_path or os.path.join(out_dir, "manifest.yaml")
    man = load_manifest(mpath)
    split = confirmatory_split(man)
    rows, prof_cache, obs_cache = [], {}, {}
    for rec in ST.iter_cells(out_dir):
        if rec.get("status") not in ("ok", "partial") or rec["cell_id"] not in split:
            continue
        c = rec["cell"]
        gkey = json.dumps([c["family"], c["n"], c.get("fparams")], sort_keys=True, default=str)
        if gkey not in prof_cache:
            try:
                prof_cache[gkey] = FT.profile_of_cell(c)
            except Exception:                            # noqa: BLE001 - reported as missing
                prof_cache[gkey] = math.nan
            try:                                         # observable features records before /2 did not store
                obs_cache[gkey] = FT.observable_of_cell(c)
            except Exception:                            # noqa: BLE001 - reported as missing
                obs_cache[gkey] = None
        v, miss = FT.from_record(rec, None, prof_cache[gkey], observable=obs_cache[gkey])
        rows.append(dict(cell_id=rec["cell_id"], family=c["family"], group=gkey, split=split[rec["cell_id"]],
                         label="%s-%d %s" % (c["family"], c["n"], json.dumps(c.get("fparams"), sort_keys=True)),
                         flags=";".join(sorted({f for m in rec.get("methods") or [] for f in (m.get("flags") or [])})),
                         **{("x_" + k): val for k, val in v.items()}))
    cells = pd.DataFrame(rows)
    if len(cells):                       # global grouped-split integrity before any evaluation (F11)
        from .phase.manifest import base_circuit_key, check_split
        recs = {r["cell_id"]: r for r in rows}
        check_split([(cid, base_circuit_key(*json.loads(recs[cid]["group"]))) for cid in recs],
                    dict(zip(cells["cell_id"], cells["split"])))
    led = pd.read_csv(os.path.join(tdir, "ledger.csv"), encoding="utf-8")
    led = led[(led["policy"] == policy) & led["cell_id"].isin(set(cells["cell_id"]))].copy()
    led["rmse"] = pd.to_numeric(led["rmse"], errors="coerce")
    led["feasible"] = led["feasible"].astype(str).str.lower().isin(("true", "1"))
    methods = _order(sorted(set(led["method"])))
    optimistic = set()
    for rec in ST.iter_cells(out_dir):
        for m in rec.get("methods") or []:
            if "optimistic" in (m.get("flags") or []):
                optimistic.add(m["key"])
    # candidates: methods whose simulated protocol is realizable as recorded. "optimistic" methods (e.g. virtual
    # distillation with ideal copies, an ideal derangement and ideal readout) are upper bounds: shown, never chosen.
    candidates = [m for m in methods if m not in optimistic]
    return dict(cells=cells, ledger=led, methods=methods, candidates=candidates, optimistic=sorted(optimistic),
                manifest=man, out_dir=out_dir)


def wide(S: dict, budgets=None) -> pd.DataFrame:
    """One row per (cell, B): RMSE and feasibility of every method."""
    L = S["ledger"]
    if budgets is not None:
        L = L[L["B"].isin(list(budgets))]
    R = L.pivot_table(index=["cell_id", "B"], columns="method", values="rmse", aggfunc="first")
    Fz = L.pivot_table(index=["cell_id", "B"], columns="method", values="feasible", aggfunc="first")
    R.columns = ["rmse_" + c for c in R.columns]
    Fz.columns = ["feas_" + c for c in Fz.columns]
    W = R.join(Fz).reset_index()
    return W.merge(S["cells"], on="cell_id", how="left")


def regimes(W: pd.DataFrame, methods, delta: float = DELTA) -> pd.DataFrame:
    out = []
    for _, r in W.iterrows():
        vals = {m: (r.get("rmse_" + m) if bool(r.get("feas_" + m, False)) else np.inf) for m in methods}
        vals = {m: (float(v) if v is not None and np.isfinite(v) else np.inf) for m, v in vals.items()}
        best_m = min(vals, key=vals.get)
        best = vals[best_m]
        if not np.isfinite(best):
            lab, tie = "no feasible method", []
        else:
            tie = sorted([m for m, v in vals.items() if v <= best * (1 + delta)], key=lambda m: vals[m])
            if "RAW" in tie:
                lab = "raw ≈ best"
            elif len(tie) > 1:
                lab = "tie"
            else:
                lab = best_m
        out.append(dict(cell_id=r["cell_id"], B=r["B"], best=best_m if np.isfinite(best) else None,
                        best_rmse=best, tie="/".join(tie), regime=lab,
                        gain_vs_raw=(vals.get("RAW", np.inf) / best) if np.isfinite(best) and best > 0 else np.nan))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ Monte-Carlo error of the ledger (review 6.8)
WINNER_PROB_MIN = 0.90           # a single-method regime label needs P(label is the winner) ≥ this under MC error


def mc_errors(S: dict) -> pd.DataFrame:
    """Per ledger row (cell, method, B): the Monte-Carlo standard error of its RMSE. The ledger's RMSE of a learning
    method uses training-level moments estimated by Monte Carlo (b, u, v, w with SEs se_b, se_u, se_v, se_w at the
    allocated s_tr); a target-nonlinear row uses a Monte-Carlo moment grid (mean_mcse, var_se); exact methods have
    none. Delta method: SE(MSE)² ≈ (2b·se_b)² + (se_v/N)² + (se_w/s)² + se_u², SE(RMSE) = SE(MSE)/(2·RMSE). The
    estimates of different methods come from independent Monte-Carlo draws (no common random numbers), so their
    errors are treated as independent."""
    from .phase import store as ST
    keep = set(S["cells"]["cell_id"])
    info = {}
    for rec in ST.iter_cells(S["out_dir"]):
        if rec["cell_id"] not in keep:
            continue
        for r in rec.get("methods") or []:
            lv = {int(l["s_tr"]): l for l in (r.get("levels") or []) if l.get("ok", True) and l.get("s_tr")}
            info[(rec["cell_id"], r["key"])] = (lv, r.get("target_moments"))
    L = S["ledger"]
    se = np.zeros(len(L))
    for k, row in enumerate(L.itertuples(index=False)):
        rm = float(getattr(row, "rmse")) if getattr(row, "rmse") == getattr(row, "rmse") else math.nan
        if not (row.feasible and math.isfinite(rm) and rm > 0):
            se[k] = math.nan
            continue
        lv, grid = info.get((row.cell_id, row.method), ({}, None))
        b = float(getattr(row, "b", 0.0) or 0.0)
        N = float(getattr(row, "N", 0) or 0)
        s_ = float(getattr(row, "s", 0) or 0)
        str_ = getattr(row, "s_tr", None)
        var_mse = 0.0
        l = lv.get(int(str_)) if str_ == str_ and str_ is not None and int(str_ or 0) in lv else None
        if l is not None:
            g = lambda key: float(l.get(key) or 0.0) if l.get(key) == l.get(key) else 0.0     # noqa: E731
            var_mse += (2 * b * g("se_b")) ** 2 + (g("se_u")) ** 2
            if N > 0:
                var_mse += (g("se_v") / N) ** 2
            if s_ > 0:
                var_mse += (g("se_w") / s_) ** 2
        if grid and N > 0:
            Ns = np.asarray(grid.get("N") or [], float)
            if len(Ns) and (grid.get("first_order_from") is None or N < float(grid["first_order_from"])):
                i = int(np.argmin(np.abs(np.log(Ns) - math.log(N))))
                mm, vs = float(grid["mean_mcse"][i] or 0.0), float(grid["var_se"][i] or 0.0)
                var_mse += (2 * b * mm) ** 2 + (vs * Ns[i] / N) ** 2
        se[k] = math.sqrt(var_mse) / (2 * rm)
    out = L[["cell_id", "method", "B"]].copy()
    out["rmse_mc_se"] = se
    return out


def winner_stability(W: pd.DataFrame, E: pd.DataFrame, methods, delta: float = DELTA, draws: int = 4000,
                     seed: int = 0) -> pd.DataFrame:
    """Per (cell, B): how stable the regime label is under the Monte-Carlo error of the estimated RMSEs.
    ``p_winner``: probability that the labeled best method is the smallest RMSE when every method's log RMSE is
    drawn from N(log RMSE, (SE/RMSE)²) independently; ``p_in_tie``: probability that it stays within δ of the
    smallest. A practical tie (within δ) is a different statement from an MC-unresolved winner: the first is about
    the size of the difference, the second about how precisely the difference was estimated."""
    Ew = E.pivot_table(index=["cell_id", "B"], columns="method", values="rmse_mc_se", aggfunc="first")
    rng = np.random.default_rng(seed)
    out = []
    for _, r in W.iterrows():
        key = (r["cell_id"], r["B"])
        vals, ses = [], []
        for m in methods:
            v = r.get("rmse_" + m)
            ok = bool(r.get("feas_" + m, False)) and v is not None and np.isfinite(v) and v > 0
            vals.append(float(v) if ok else np.inf)
            sv = Ew.loc[key, m] if (key in Ew.index and m in Ew.columns) else 0.0
            ses.append(float(sv) if ok and sv == sv else 0.0)
        vals, ses = np.asarray(vals), np.asarray(ses)
        if not np.isfinite(vals).any():
            out.append(dict(cell_id=r["cell_id"], B=r["B"], p_winner=math.nan, p_in_tie=math.nan, mc_max_rel_se=0.0))
            continue
        b = int(np.argmin(vals))
        fin = np.isfinite(vals)
        rel = np.where(fin, ses / np.where(fin, vals, 1.0), 0.0)
        if not (rel[fin] > 0).any():
            out.append(dict(cell_id=r["cell_id"], B=r["B"], p_winner=1.0, p_in_tie=1.0, mc_max_rel_se=0.0))
            continue
        lv = np.where(fin, np.log(np.where(fin, vals, 1.0)), np.inf)
        Z = lv[None, :] + rng.normal(size=(int(draws), len(vals))) * rel[None, :]
        Z[:, ~fin] = np.inf
        mn = Z.min(1)
        out.append(dict(cell_id=r["cell_id"], B=r["B"], p_winner=float(np.mean(Z[:, b] <= mn)),
                        p_in_tie=float(np.mean(Z[:, b] <= mn + math.log(1 + delta))),
                        mc_max_rel_se=float(rel[fin].max())))
    return pd.DataFrame(out)


def mc_sensitivity(S: dict, E: pd.DataFrame, feature_set: str = "structural", reps: int = 40, seed: int = 0,
                   alpha: float = 1.0) -> dict:
    """Spread of the grouped-CV headline metrics when the ledger's RMSEs are redrawn from their Monte-Carlo error
    (independent normal draws of log RMSE): the part of the evaluation's uncertainty that a bootstrap over base
    circuits does not contain. Reported as 2.5–97.5 % ranges over ``reps`` redraws."""
    D = rec_dataset(S, feature_set)
    keyed = E.set_index(["cell_id", "B", "method"])["rmse_mc_se"]
    Wd = D["W"]
    rel = np.zeros_like(D["Y"])
    for j, m in enumerate(D["methods"]):
        sev = np.array([keyed.get((c, b, m), 0.0) for c, b in zip(Wd["cell_id"], Wd["B"])], float)
        rm = Wd["rmse_" + m].astype(float).to_numpy()
        rel[:, j] = np.where(np.isfinite(sev) & (rm > 0), sev / np.where(rm > 0, rm, 1.0), 0.0) / math.log(10)
    explore = np.flatnonzero(D["split"] == "explore")
    folds = RC.group_folds(D["groups"][explore], k=5, seed=1)
    rng = np.random.default_rng(seed)
    keys = ("regret_median", "accuracy", "coverage", "policy_regret_median", "harmful", "interval_coverage")
    acc = {k: [] for k in keys}
    for _ in range(int(reps)):
        Yp = D["Y"] + rng.normal(size=D["Y"].shape) * rel
        ev = RC.evaluate(D["X"][explore], Yp[explore], D["F"][explore], D["methods"], folds, D["lam"][explore],
                         D["budget"][explore], alpha, groups=D["groups"][explore])["summary"]
        for k in keys:
            acc[k].append(ev.get(k, math.nan))
    rel_fin = rel[np.isfinite(rel) & (rel > 0)] * math.log(10)
    return dict(reps=int(reps), feature_set=feature_set,
                median_rel_se=float(np.median(rel_fin)) if rel_fin.size else 0.0,
                max_rel_se=float(rel_fin.max()) if rel_fin.size else 0.0,
                ranges={k: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] for k, v in acc.items()
                        if np.isfinite(v).any()})


def group_bootstrap(df: pd.DataFrame, value: str, groups: str = "group", reps: int = 2000, seed: int = 0,
                    stat=np.median) -> tuple:
    """(statistic, 2.5 %, 97.5 %) with whole base circuits resampled."""
    g = df.groupby(groups)[value].apply(lambda s: s.dropna().to_numpy())
    arrs = [a for a in g if len(a)]
    if not arrs:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    est = float(stat(np.concatenate(arrs)))
    bs = []
    for _ in range(int(reps)):
        pick = rng.integers(0, len(arrs), len(arrs))
        bs.append(float(stat(np.concatenate([arrs[i] for i in pick]))))
    return est, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def improvements(W: pd.DataFrame, methods, B: int) -> pd.DataFrame:
    """Per method at budget B, with explicit denominators (1.5.1):

    * ``planned``: cells at B where the method was planned (and raw has a finite RMSE);
    * ``cells`` (= ``feasible``): of those, cells where the method ran with a finite RMSE; ``infeasible`` = planned −
      feasible (cannot run at B, or failed); ``feasible_fraction`` = feasible / planned;
    * ``better``: cells where it beats raw by more than δ; ``share_better_all`` = better / planned (an infeasible case
      counts as *not* better: the operational success rate), ``share_better_feasible`` = better / feasible
      (conditional on running). ``share_better`` is kept as the 1.5.0 name of the conditional share;
    * the median log10(RMSE_m / RMSE_raw) and the median absolute gain are over feasible cells only (conditional on
      running), with base-circuit bootstrap 95 % intervals."""
    d = W[W["B"] == B]
    d = d[np.isfinite(pd.to_numeric(d["rmse_RAW"], errors="coerce"))] if "rmse_RAW" in d else d
    out = []
    for m in methods:
        if m == "RAW" or "rmse_" + m not in d:
            continue
        x = d.copy()
        ok = x["feas_" + m].fillna(False).astype(bool) & np.isfinite(x["rmse_" + m]) & np.isfinite(x["rmse_RAW"])
        x["lr"] = np.where(ok, np.log10(x["rmse_" + m] / x["rmse_RAW"]), np.nan)
        x["absdiff"] = np.where(ok, x["rmse_RAW"] - x["rmse_" + m], np.nan)
        med, lo, hi = group_bootstrap(x, "lr")
        amed, alo, ahi = group_bootstrap(x, "absdiff")
        planned, feas = int(len(x)), int(ok.sum())
        better = int((x.loc[ok, "lr"] < -math.log10(1 + DELTA)).sum())
        cond = better / feas if feas else math.nan
        out.append(dict(method=m, B=B, planned=planned, cells=feas, feasible=feas, infeasible=planned - feas,
                        feasible_fraction=feas / planned if planned else math.nan, better=better,
                        share_better_all=better / planned if planned else math.nan, share_better_feasible=cond,
                        share_better=cond, median_log10_ratio=med, ci_lo=lo, ci_hi=hi, median_abs_gain=amed,
                        abs_ci_lo=alo, abs_ci_hi=ahi, groups=int(x.loc[ok, "group"].nunique()),
                        groups_planned=int(x["group"].nunique())))
    return pd.DataFrame(out)


def budgets_to_eps(S: dict, eps: float) -> pd.DataFrame:
    """Smallest total budget reaching RMSE ≤ ε per cell and method (inf when the bias floor prevents it)."""
    from .phase import store as ST
    from .phase.cell import cost_of
    from .phase.ledger import budget_for
    keep = set(S["cells"]["cell_id"])
    out = []
    for rec in ST.iter_cells(S["out_dir"]):
        if rec["cell_id"] not in keep:
            continue
        for r in rec.get("methods") or []:
            c = cost_of(r)
            Bm = budget_for(c, eps) if c is not None else math.nan
            out.append(dict(cell_id=rec["cell_id"], method=r["key"], eps=eps, B_eps=Bm))
    return pd.DataFrame(out).merge(S["cells"][["cell_id", "group", "family", "split"]], on="cell_id")


def write_context(S: dict, out: str, feature_schema: str | None = None) -> str:
    """``tables/methods.json`` (schema ``qemx.methods/2``): the protocols every cell actually ran (inherited readout
    mitigation, calibration, transforms, training), the noise context, cost policy, budget range and feature schema
    the live recommendation must reproduce (``qemx.rec_context``)."""
    from . import rec_context as RX
    ctx = RX.study_context(S, feature_schema or FT.VERSION)
    return RX.write_methods(os.path.join(out, "tables", "methods.json"), ctx)


# ------------------------------------------------------------------ recommender evaluation
REC_BUDGETS = (1000, 10_000, 100_000, 1_000_000, 10_000_000)


def nearest_budgets(S: dict, targets=REC_BUDGETS) -> list:
    grid = sorted(set(S["ledger"]["B"]))
    return sorted({min(grid, key=lambda b: abs(math.log10(b) - math.log10(t))) for t in targets})


def rec_dataset(S: dict, feature_set: str, budgets=None) -> dict:
    budgets = budgets or nearest_budgets(S)
    W = wide(S, budgets)
    names = list(FT.SETS[feature_set])
    W["x_log10_budget"] = np.log10(W["B"].astype(float))
    X = W[["x_" + n for n in names]].to_numpy(float)
    methods = [m for m in S["candidates"] if "rmse_" + m in W]
    Y = np.column_stack([np.log10(W["rmse_" + m].astype(float).clip(lower=1e-12)) for m in methods])
    F = np.column_stack([W["feas_" + m].astype(bool) & np.isfinite(W["rmse_" + m].astype(float)) for m in methods])
    return dict(W=W, X=X, Y=Y, F=F, methods=methods, names=names, groups=W["group"].to_numpy(),
                family=W["family"].to_numpy(), n=W["x_n"].to_numpy(), split=W["split"].to_numpy(),
                lam=W["x_lam_lc"].to_numpy(float), budget=W["B"].to_numpy(float))


def evaluate_recommender(S: dict, feature_set: str = "structural", alpha: float = 1.0) -> dict:
    D = rec_dataset(S, feature_set)
    res = {}
    explore = np.flatnonzero(D["split"] == "explore")
    confirm = np.flatnonzero(D["split"] == "confirm")
    sub = lambda idx: {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in D.items()}     # noqa: E731
    E = sub(explore)
    folds = RC.group_folds(E["groups"], k=5, seed=1)
    RC.check_no_leakage(folds, E["groups"])
    res["grouped_cv"] = RC.evaluate(E["X"], E["Y"], E["F"], E["methods"], folds, E["lam"], E["budget"], alpha,
                                    groups=E["groups"])
    lofo = RC.leave_one_out(E["family"], "family")
    res["leave_family_out"] = RC.evaluate(E["X"], E["Y"], E["F"], E["methods"], lofo, E["lam"], E["budget"], alpha,
                                          groups=E["groups"])
    nmax = np.nanmax(E["n"])
    lwo = [RC.Fold(np.flatnonzero(E["n"] < nmax), np.flatnonzero(E["n"] == nmax), "width %d held out" % nmax)]
    res["leave_width_out"] = RC.evaluate(E["X"], E["Y"], E["F"], E["methods"], lwo, E["lam"], E["budget"], alpha,
                                         groups=E["groups"])
    if len(confirm):
        Xall = np.r_[D["X"][explore], D["X"][confirm]]
        Yall = np.r_[D["Y"][explore], D["Y"][confirm]]
        Fall = np.r_[D["F"][explore], D["F"][confirm]]
        fold = [RC.Fold(np.arange(len(explore)), np.arange(len(explore), len(explore) + len(confirm)),
                        "confirmatory subset (frozen before outcomes)")]
        res["confirmatory"] = RC.evaluate(Xall, Yall, Fall, D["methods"], fold,
                                          np.r_[D["lam"][explore], D["lam"][confirm]],
                                          np.r_[D["budget"][explore], D["budget"][confirm]], alpha,
                                          groups=np.r_[D["groups"][explore], D["groups"][confirm]])
    res["meta"] = dict(feature_set=feature_set, features=D["names"], methods=D["methods"], rows=len(D["X"]),
                       explore_rows=int(len(explore)), confirm_rows=int(len(confirm)),
                       base_circuits=int(len(set(D["groups"].tolist()))), budgets=nearest_budgets(S))
    return res


def subset(S: dict, keep_cells) -> dict:
    """A study restricted to the cells ``keep_cells`` (for development/holdout splits of one run)."""
    keep = set(keep_cells)
    T = dict(S)
    T["cells"] = S["cells"][S["cells"]["cell_id"].isin(keep)].reset_index(drop=True)
    T["ledger"] = S["ledger"][S["ledger"]["cell_id"].isin(keep)].reset_index(drop=True)
    return T


def evaluate_holdout(S_dev: dict, S_hold: dict, feature_set: str = "structural", alpha: float = 1.0,
                     allow_context_mismatch: bool = False) -> dict:
    """The frozen recommender trained on every development cell and evaluated once on a separate holdout run (the
    final confirmation, docs/FINAL_HOLDOUT.md). Refuses — before evaluating anything, so the once-only holdout stays
    unused — if a base circuit appears in both, or if the two runs did not use the same protocols and noise context
    (qemx.rec_context). ``allow_context_mismatch`` evaluates anyway for diagnosis; the result then says so in
    ``meta.context_mismatch_allowed`` and is not the confirmatory evaluation."""
    from . import rec_context as RX
    D = rec_dataset(S_dev, feature_set)
    H = rec_dataset(S_hold, feature_set, budgets=sorted(set(D["budget"].tolist())))
    shared = set(D["groups"].tolist()) & set(H["groups"].tolist())
    if shared:
        raise ValueError("%d base circuit(s) are in both the development and the holdout data (e.g. %s)"
                         % (len(shared), sorted(shared)[0]))
    problems = []
    try:
        cd, ch = RX.study_context(S_dev, FT.VERSION), RX.study_context(S_hold, FT.VERSION)
        rd = {k: v.get("recipe") for k, v in (cd.get("protocols") or {}).items()}
        rh = {k: v.get("recipe") for k, v in (ch.get("protocols") or {}).items()}
        for k in sorted(set(rd) & set(rh)):
            if json.dumps(rd[k], sort_keys=True, default=str) != json.dumps(rh[k], sort_keys=True, default=str):
                problems.append("protocol %s ran as a different recipe in the holdout" % k)
        nd, nh = cd.get("noise") or {}, ch.get("noise") or {}
        for key in ("devices", "readout", "homogenized", "backends"):
            if nd.get(key) != nh.get(key):
                problems.append("noise context %s differs: %s vs %s" % (key, nd.get(key), nh.get(key)))
        for key in ("scales", "coherent_fractions"):
            lo, hi = min(nd.get(key) or [0]), max(nd.get(key) or [0])
            if any(not (lo <= x <= hi) for x in nh.get(key) or []):
                problems.append("holdout %s outside the development range [%g, %g]" % (key, lo, hi))
    except Exception as e:                               # noqa: BLE001 - reported
        problems.append("context comparison failed: %s: %s" % (type(e).__name__, str(e)[:200]))
    if problems and not allow_context_mismatch:
        raise ValueError("the holdout run does not match the development run's protocols and noise context, so it "
                         "was not evaluated: " + "; ".join(problems))
    methods = [m for m in D["methods"] if m in H["methods"]]
    jd = [D["methods"].index(m) for m in methods]
    jh = [H["methods"].index(m) for m in methods]
    X = np.r_[D["X"], H["X"]]
    Y = np.r_[D["Y"][:, jd], H["Y"][:, jh]]
    F = np.r_[D["F"][:, jd], H["F"][:, jh]]
    fold = [RC.Fold(np.arange(len(D["X"])), np.arange(len(D["X"]), len(X)), "final holdout")]
    ev = RC.evaluate(X, Y, F, methods, fold, np.r_[D["lam"], H["lam"]], np.r_[D["budget"], H["budget"]], alpha,
                     groups=np.r_[D["groups"], H["groups"]])
    ev["meta"] = dict(feature_set=feature_set, features=D["names"], methods=methods, dev_rows=int(len(D["X"])),
                      holdout_rows=int(len(H["X"])), dev_base_circuits=int(len(set(D["groups"].tolist()))),
                      holdout_base_circuits=int(len(set(H["groups"].tolist()))), problems=problems,
                      context_mismatch_allowed=bool(problems and allow_context_mismatch))
    return ev


# ------------------------------------------------------------------ figures
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "savefig.bbox": "tight",
                         "figure.dpi": 110})
    return plt


def _save(fig, out, name):
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out, "%s.%s" % (name, ext))
        fig.savefig(p, dpi=200 if ext == "png" else None)
        paths.append(p)
    return paths


def fig_coverage(S, out):
    plt = _mpl()
    C = S["cells"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fams = sorted(C["family"].unique())
    cmap = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A7A7A"]
    for i, f in enumerate(fams):
        d = C[C["family"] == f]
        for sp, mk in (("explore", "o"), ("confirm", "s")):
            e = d[d["split"] == sp]
            ax[0].scatter(e["x_two_qubit"], e["x_lam_lc"], s=18, marker=mk, color=cmap[i % len(cmap)],
                          edgecolor="white", linewidth=0.4, label=f if sp == "explore" else None)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("two-qubit gates after compilation")
    ax[0].set_ylabel("light-cone fault rate λ_LC")
    ax[0].set_title("a  Feature coverage (○ explore, □ confirm)")
    ax[0].legend(frameon=False, ncol=1, loc="lower right")
    tab = C.groupby(["family", "split"]).agg(groups=("group", "nunique")).reset_index()
    pv = tab.pivot(index="family", columns="split", values="groups").fillna(0)
    y = np.arange(len(pv))
    ax[1].barh(y - 0.18, pv.get("explore", 0), height=0.34, color="#0072B2", label="explore")
    ax[1].barh(y + 0.18, pv.get("confirm", 0), height=0.34, color="#E69F00", label="confirm")
    ax[1].set_yticks(y, pv.index)
    ax[1].set_xlabel("base circuits")
    ax[1].set_title("b  Data per family")
    ax[1].legend(frameon=False)
    return _save(fig, out, "fig1_coverage"), fig


def fig_regimes(S, R, W, out):
    plt = _mpl()
    from matplotlib.lines import Line2D
    show = nearest_budgets(S, (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8))
    M = R[R["B"].isin(show)].merge(W[["cell_id", "B", "x_lam_lc", "split", "x_coherent_fraction"]],
                                   on=["cell_id", "B"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, sp in zip(axes, ("explore", "confirm")):
        d = M[M["split"] == sp]
        if d.empty:
            ax.text(0.5, 0.5, "no %s cells" % sp, ha="center", va="center", transform=ax.transAxes, color="#555555")
        for lab, g in d.groupby("regime"):
            if lab in COLORS:
                ax.scatter(g["x_lam_lc"], g["B"], s=26, color=COLORS[lab], marker=MARKERS.get(lab, "o"),
                           edgecolor="white", linewidth=0.4)
            else:
                ax.scatter(g["x_lam_lc"], g["B"], s=26, facecolor=REGIME_OTHER.get(lab, "#BBBBBB"),
                           edgecolor="#555555", linewidth=0.6, marker="o")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("light-cone fault rate λ_LC")
        ax.set_title("%s  %s cells (%d base circuits)" % ("a" if sp == "explore" else "b", sp,
                                                           d.merge(S["cells"][["cell_id", "group"]])["group"].nunique()))
    axes[0].set_ylabel("total shot budget B")
    present = [m for m in S["methods"] if m in set(M["regime"])]
    h = [Line2D([], [], marker=MARKERS[m], color=COLORS[m], linestyle="", markersize=5, label=m) for m in present]
    h += [Line2D([], [], marker="o", markerfacecolor=REGIME_OTHER[k], markeredgecolor="#555555", linestyle="",
                 markersize=5, label=k) for k in ("tie", "raw ≈ best", "no feasible method") if k in set(M["regime"])]
    fig.subplots_adjust(bottom=0.27, wspace=0.12)
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 8), frameon=False, bbox_to_anchor=(0.5, 0.0))
    return _save(fig, out, "fig2_regime_map"), fig


def fig_improvement(I, out, B, optimistic=()):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    I = I.sort_values("median_log10_ratio")
    y = np.arange(len(I))
    for k, (_, r) in enumerate(I.iterrows()):
        c = COLORS.get(r["method"], "#333333")
        ax[0].plot([r["ci_lo"], r["ci_hi"]], [k, k], color=c, linewidth=2)
        ax[0].plot(r["median_log10_ratio"], k, marker=MARKERS.get(r["method"], "o"), color=c, markersize=6)
    ax[0].axvline(0, color="#333333", linewidth=0.8)
    ax[0].axvspan(-math.log10(1 + DELTA), math.log10(1 + DELTA), color="#EEEEEE", zorder=0)
    lbl = [m + (" †" if m in (optimistic or ()) else "") for m in I["method"]]
    ax[0].set_yticks(y, lbl)
    ax[0].set_xlabel("median log10(RMSE_method / RMSE_raw)")
    ax[0].set_title("a  Paired change vs raw at B = %s" % f"{B:,}")
    cols = [COLORS.get(m, "#333333") for m in I["method"]]
    # all planned cells: solid bar (an infeasible case counts as not better); conditional on running: open marker
    ax[1].barh(y, I["share_better_all"], color=cols)
    ax[1].scatter(I["share_better_feasible"], y, marker="|", s=90, color="#333333", zorder=3,
                  label="among feasible cells")
    ax[1].set_yticks(y, lbl)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("share of planned cells better than raw by > 10 %")
    ax[1].set_title("b  How often it helps")
    ax[1].legend(loc="lower right", fontsize=7, frameon=False)
    for k, (_, r) in enumerate(I.iterrows()):
        txt = "%d/%d" % (r["better"], r["planned"]) + ("; %d cannot run" % r["infeasible"] if r["infeasible"] else "")
        ax[1].text(min(max(r["share_better_all"], r["share_better_feasible"] if np.isfinite(r["share_better_feasible"])
                           else 0), 0.62) + 0.03, k, txt, va="center", fontsize=7, color="#333333")
    return _save(fig, out, "fig3_paired_improvement"), fig


def fig_pareto(BE, out, eps):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    methods = _order(sorted(set(BE["method"])))
    for k, m in enumerate(methods):
        d = BE[BE["method"] == m]
        fin = d["B_eps"].replace([np.inf], np.nan).dropna()
        unreach = int(np.isinf(d["B_eps"]).sum())
        if len(fin):
            jitter = (np.random.default_rng(k).random(len(fin)) - 0.5) * 0.35
            ax.scatter(k + jitter, fin, s=8, color=COLORS.get(m, "#333333"), alpha=0.7, edgecolor="none")
            ax.plot([k - 0.3, k + 0.3], [np.median(fin)] * 2, color="#222222", linewidth=1.4)
        ax.text(k, 0.02, "%d/%d\nunreach." % (unreach, len(d)), ha="center", va="bottom", fontsize=6.5,
                color="#555555", transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)), methods)
    ax.set_ylabel("total shots to reach RMSE ≤ %g" % eps)
    ax.set_title("Cost to reach a stated precision (bar: median over reachable cells)")
    return _save(fig, out, "fig4_cost_to_precision"), fig


def fig_discrepancy(reh_dir, out):
    """Rehearsal (EMULATION): z under the matched null and under controlled alternatives, and whether the
    alternative changes the ranking of methods by fidelity (Kendall τ between predicted and observed orderings)."""
    plt = _mpl()
    from scipy.stats import kendalltau
    tags = sorted(d for d in os.listdir(os.path.join(reh_dir, "analysis")))
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
    taus, zs = {}, {}
    for t in tags:
        p = os.path.join(reh_dir, "analysis", t, "tables", "divergence.csv")
        if not os.path.exists(p):
            continue
        D = pd.read_csv(p, encoding="utf-8")
        D = D[(D["status"] == "ok") & (D["predictor"] == "depolarizing") & D["method"].isin(["RAW", "RO", "ZNE-G",
                                                                                          "ZNE-L"])]
        z = D["z"].to_numpy(float)
        z = z[np.isfinite(z)]
        lab = "null (model true)" if t.startswith("null") else t.split("_", 1)[1].replace("_", "=").replace("p", ".")
        zs.setdefault(lab, []).extend(z.tolist())
        tt = []
        for inst, g in D.groupby("instance"):
            if len(g) >= 3:
                tt.append(kendalltau(g["fc_pred"], g["fc_obs"]).statistic)
        taus.setdefault(lab, []).extend([x for x in tt if np.isfinite(x)])
    for (lab, z), c in zip(sorted(zs.items(), key=lambda kv: kv[0] != "null (model true)"),
                           ("#333333", "#D55E00", "#0072B2", "#009E73", "#CC79A7")):
        ax[0].hist(np.clip(z, -8, 12), bins=np.linspace(-8, 12, 41), histtype="step", linewidth=1.4, color=c,
                   density=True, label="%s (%d rows)" % (lab, len(z)))
    ax[0].set_xlabel("z = (predicted − observed F_c) / SD")
    ax[0].set_ylabel("density")
    ax[0].set_title("a  Discrepancy statistic (EMULATION)")
    ax[0].legend(frameon=False, fontsize=6.5)
    labs = sorted(taus, key=lambda l: l != "null (model true)")
    ax[1].boxplot([taus[l] for l in labs], vert=False, widths=0.5, flierprops=dict(markersize=3))
    ax[1].set_yticks(range(1, len(labs) + 1), [l.replace(" (model true)", "\n(model true)") for l in labs])
    ax[1].set_xlabel("Kendall τ: method ranking predicted vs observed (per circuit)")
    ax[1].set_title("b  Does mismatch reorder the methods?")
    return _save(fig, out, "fig5_discrepancy_rehearsal"), fig


def fig_recommender(ev, out):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    names = [k for k in ("grouped_cv", "leave_family_out", "leave_width_out", "confirmatory") if k in ev]
    labels = {"grouped_cv": "grouped CV", "leave_family_out": "leave family out", "leave_width_out":
              "leave width out", "confirmatory": "confirmatory"}
    x = np.arange(len(names))
    keys = [("regret_median", "recommender (decided rows)", "#0072B2"),
            ("baseline_fixed_regret_median", "best fixed method", "#E69F00"),
            ("baseline_rule_regret_median", "rule (λ, B)", "#009E73"),
            ("baseline_raw_regret_median", "always raw", "#7A7A7A")]
    for k, (key, lab, c) in enumerate(keys):
        vals = [ev[n]["summary"].get(key, np.nan) for n in names]
        ax[0].bar(x + (k - 1.5) * 0.2, vals, width=0.2, color=c, label=lab)
        for xi, v in zip(x + (k - 1.5) * 0.2, vals):
            if np.isfinite(v):
                ax[0].text(xi, max(v, 0) * 1.08 + 0.003, "%.2g" % v, ha="center", va="bottom", fontsize=5.5,
                           rotation=90)
    ax[0].set_xticks(x, [labels[n] for n in names], rotation=15)
    ax[0].set_ylabel("median regret RMSE/RMSE_best − 1")
    ax[0].set_yscale("symlog", linthresh=0.1)
    ax[0].set_title("a  Regret on held-out data")
    ax[0].set_ylim(0, 20)
    ax[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    for n, mk in zip(names, ("o", "s", "D", "^")):
        rc = RC.risk_coverage(ev[n]["rows"])
        if rc:
            ax[1].plot([a for a, _ in rc], [b for _, b in rc], marker=mk, markevery=max(1, len(rc) // 8),
                       markersize=4, label=labels[n])
    ax[1].set_yscale("symlog", linthresh=0.1)
    ax[1].set_xlabel("coverage (share of rows given a recommendation)")
    ax[1].set_ylabel("median regret of the admitted rows")
    ax[1].set_title("b  Selective risk vs coverage")
    ax[1].legend(frameon=False, fontsize=6.5)
    return _save(fig, out, "fig6_recommender"), fig


# ------------------------------------------------------------------ report
def report(out_dir: str, manifest_path: str | None, out: str, rehearsal: str | None = None, B_ref: int = 100_000,
           eps: float = 0.02) -> dict:
    os.makedirs(out, exist_ok=True)
    tdir = os.path.join(out, "tables")
    os.makedirs(tdir, exist_ok=True)
    S = load(out_dir, manifest_path)
    W = wide(S)
    R = regimes(W, S["candidates"])
    Bgrid = sorted(set(W["B"]))
    Bref = min(Bgrid, key=lambda b: abs(math.log10(b) - math.log10(B_ref)))
    I = improvements(W, S["methods"], Bref)
    BE = budgets_to_eps(S, eps)
    S["cells"].to_csv(os.path.join(tdir, "cells_features.csv"), index=False, encoding="utf-8")
    write_context(S, out)                                # the resolved protocols and context (qemx.methods/2)
    W.to_csv(os.path.join(tdir, "rmse_wide.csv"), index=False, encoding="utf-8")
    R.to_csv(os.path.join(tdir, "regimes.csv"), index=False, encoding="utf-8")
    I.to_csv(os.path.join(tdir, "paired_improvement.csv"), index=False, encoding="utf-8")
    BE.to_csv(os.path.join(tdir, "budget_to_eps.csv"), index=False, encoding="utf-8")
    E = mc_errors(S)
    E.to_csv(os.path.join(tdir, "ledger_mc_errors.csv"), index=False, encoding="utf-8")
    WS = winner_stability(W, E, S["candidates"])
    R = R.merge(WS, on=["cell_id", "B"], how="left")
    R.to_csv(os.path.join(tdir, "regimes.csv"), index=False, encoding="utf-8")
    ev = {}
    for fs in ("circuit_observable", "structural", "normalized", "with_state"):
        try:
            ev[fs] = evaluate_recommender(S, fs)
        except Exception as e:                           # noqa: BLE001 - reported, not fatal
            ev[fs] = dict(error="%s: %s" % (type(e).__name__, str(e)[:300]))
    try:
        sens = mc_sensitivity(S, E, "structural")
    except Exception as e:                               # noqa: BLE001 - reported, not fatal
        sens = dict(error="%s: %s" % (type(e).__name__, str(e)[:300]))
    with open(os.path.join(tdir, "mc_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(sens, f, indent=1, default=float)
    with open(os.path.join(tdir, "recommender_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump({k: ({kk: (vv["summary"] if isinstance(vv, dict) and "summary" in vv else vv)
                        for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in ev.items()}, f, indent=1,
                  default=float)
    figs = {}
    figs["fig1"], _ = fig_coverage(S, out)
    figs["fig2"], _ = fig_regimes(S, R, W, out)
    figs["fig3"], _ = fig_improvement(I, out, Bref, S["optimistic"])
    figs["fig4"], _ = fig_pareto(BE, out, eps)
    if rehearsal and os.path.isdir(os.path.join(rehearsal, "analysis")):
        figs["fig5"], _ = fig_discrepancy(rehearsal, out)
    if "error" not in ev.get("structural", {}):
        figs["fig6"], _ = fig_recommender(ev["structural"], out)
    md = _report_md(S, R, I, BE, ev, Bref, eps, rehearsal, figs, sens)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return dict(figures=figs, tables=sorted(os.listdir(tdir)), report=os.path.join(out, "report.md"))


def _fmt(x, p=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else ("%.*g" % (p, x))


def _report_md(S, R, I, BE, ev, Bref, eps, rehearsal, figs, sens=None) -> str:
    C = S["cells"]
    man = S["manifest"]
    n_cells, n_groups = len(C), C["group"].nunique()
    L = ["# Main-study figure report — %s" % man.get("study", "study"), "",
         "Simulation study on the device model named in the manifest (no hardware data in figures 1–4 and 6; figure 5 "
         "is an EMULATION rehearsal). %d cells from %d base circuits (%d explore / %d confirm base circuits, split by "
         "hash before any outcome). Resource: the total shot budget B of one estimate (training, calibration and every "
         "extra circuit included), allocated by the equal-budget ledger (policy *fixed*). Outcome: RMSE against the "
         "exact ideal value. Intervals: 95 %% bootstrap over base circuits. Practical equivalence δ = 10 %% in RMSE." % (
             n_cells, n_groups, C[C["split"] == "explore"]["group"].nunique(),
             C[C["split"] == "confirm"]["group"].nunique()), ""]
    if S["optimistic"]:
        L += ["† Upper bounds, never candidates: %s (flag `optimistic`: simulated with ideal resources, e.g. virtual "
              "distillation with ideal copies, an ideal derangement and ideal readout). They appear in Figures 3–4 "
              "for reference and are excluded from the regime maps and the recommender." % ", ".join(S["optimistic"]),
              ""]
    flags = sorted({f for fl in C["flags"] for f in fl.split(";") if f})
    if flags:
        L += ["Approximation flags present in the records: %s." % ", ".join("`%s`" % f for f in flags), ""]
    L += ["## Figure 1 — Feature coverage", "", "![](fig1_coverage.png)", "",
          "(a) Compiled two-qubit count against the light-cone fault rate λ_LC per cell (○ explore, □ confirmatory). "
          "(b) Base circuits per family and split. Regions without points are not covered: no figure below "
          "extrapolates into them.", "",
          "## Figure 2 — Empirical performance-regime map", "", "![](fig2_regime_map.png)", "",
          "Each point is one cell at one budget of the grid (no interpolation), colored by the best method; gray: a tie "
          "within δ; white: raw within δ of the best (no demonstrated benefit); black: no method can run at that "
          "budget. (a) exploration cells, (b) confirmatory cells. Regime shares:", ""]
    share = R["regime"].value_counts()
    L += ["| regime | (cell, B) points | share |", "|---|---|---|"] + [
        "| %s | %d | %.3f |" % (k, v, v / len(R)) for k, v in share.items()]
    L += ["", "Points are (cell, budget) pairs on a grid: %d points from %d base circuits. They are not independent "
          "observations — inference in this report resamples whole base circuits." % (len(R), S["cells"]["group"].nunique())]
    if "p_winner" in R:
        single = R[R["regime"].isin(S["candidates"]) & (R["regime"] != "raw ≈ best")]
        unres = single[single["p_winner"] < WINNER_PROB_MIN]
        L += ["", "**Monte-Carlo stability of the labels.** The RMSEs of the learning methods (and of target-nonlinear "
              "estimators) come from Monte-Carlo moments with their own errors. Redrawing every method's RMSE from its "
              "Monte-Carlo error (independent draws), %d of %d single-method labels have P(label is the winner) < %.2f "
              "(`regimes.csv`: `p_winner`, `p_in_tie`). Such a label is *MC-unresolved*: more Monte-Carlo precision, "
              "not a different method, could change it. This is not the same as a practical tie (difference below "
              "δ)." % (len(unres), len(single), WINNER_PROB_MIN)]
    L += ["", "## Figure 3 — Paired change against raw", "", "![](fig3_paired_improvement.png)", "",
          "(a) Median over the cells where the method runs (conditional on feasibility) of log10(RMSE_method / "
          "RMSE_raw) at B = %s with 95 %% base-circuit bootstrap intervals; the gray band is ±δ. (b) Bars: share of "
          "*all planned* cells in which the method is better than raw by more than δ — a case that cannot run at this "
          "budget counts as not better; ticks: the same share among the cells where it runs. The labels give "
          "better/planned and the number of cells where it cannot run." % f"{Bref:,}", "",
          "| method | planned | feasible (fraction) | median log10 ratio, feasible cells [95 % CI] | median absolute "
          "RMSE gain, feasible cells [95 % CI] | better / planned | better / feasible |",
          "|---|---|---|---|---|---|---|"]
    for _, r in I.iterrows():
        L.append("| %s | %d | %d (%s) | %s [%s, %s] | %s [%s, %s] | %d/%d = %s | %d/%d = %s |" % (
            r["method"], r["planned"], r["feasible"], _fmt(r["feasible_fraction"], 2), _fmt(r["median_log10_ratio"]),
            _fmt(r["ci_lo"]), _fmt(r["ci_hi"]), _fmt(r["median_abs_gain"]), _fmt(r["abs_ci_lo"]),
            _fmt(r["abs_ci_hi"]), r["better"], r["planned"], _fmt(r["share_better_all"], 2), r["better"],
            r["feasible"], _fmt(r["share_better_feasible"], 2)))
    L += ["", "## Figure 4 — Cost to reach a stated precision", "", "![](fig4_cost_to_precision.png)", "",
          "Smallest total budget at which RMSE ≤ %g, per cell (points) and its median over reachable cells (bar). "
          "'Unreachable' cells have a bias floor above %g at any budget: more shots cannot fix them." % (eps, eps), ""]
    if "fig5" in figs:
        try:
            with open(os.path.join(rehearsal, "rehearsal.json"), encoding="utf-8") as f:
                rj = json.load(f)
        except Exception:                                # noqa: BLE001
            rj = {}
        fp = rj.get("false_positive", {})
        L += ["## Figure 5 — Discrepancy diagnostics and method rankings (EMULATION, not hardware)", "",
              "![](fig5_discrepancy_rehearsal.png)", "",
              "End-to-end rehearsal of the hardware path on emulated observations of the divergence suite: (a) the "
              "discrepancy statistic z when the truth is the predictor's own model (null) and under controlled "
              "unmodeled effects; (b) per circuit, Kendall τ between the ordering of the four methods by predicted "
              "and by observed classical fidelity F_c (Chen et al. Eq. 1, unnormalized). Under the null, Holm-adjusted "
              "flags: %s of %s rows; unadjusted |z| > 3: %s." % (
                  fp.get("holm", {}).get("count", "—"), fp.get("holm", {}).get("n", "—"),
                  fp.get("abs_z_gt_3", {}).get("count", "—")), ""]
    st = ev.get("structural", {})
    if "error" not in st and st:
        L += ["## Figure 6 — Baseline recommender on held-out data", "", "![](fig6_recommender.png)", "",
              "Ridge models of log10 RMSE per method from recommendation-time features (%s), fitted inside each "
              "training fold. Folds group every row of a base circuit; leave-family-out and leave-width-out test "
              "transfer; the confirmatory subset was fixed before any outcome (it has since been inspected during "
              "development, so it is a development benchmark, not an untouched holdout). (a) Median regret "
              "(RMSE_chosen / RMSE_best − 1) of the recommender on the rows where it decided, and of the baselines on "
              "all rows. (b) Selective risk: regret as rows are admitted from the most to the least confident." % ", ".join(
                  st["meta"]["features"]), "",
              "Denominators: *rows* are (cell, budget) pairs, *decided* the rows with a recommendation; intervals "
              "resample base circuits. The **full policy** falls back on abstention to the training fold's best fixed "
              "method (raw if it cannot run), so it is defined on every row. *Harmful*: the choice is worse than raw "
              "by more than δ. *Paired log10 RMSE ratio*: the mean over the same rows of log10(RMSE_choice / "
              "RMSE_best fixed method); negative means the recommender's choice is better.", "",
              "| evaluation | feature set | rows (base circuits) | decided | coverage | accuracy (in δ set) | median "
              "regret, decided [p90] | harmful, decided | full policy: median regret [p90] | full policy harmful | "
              "paired log10 RMSE ratio vs fixed, decided [95 % CI] | paired log10 RMSE ratio, full policy vs fixed "
              "[95 % CI] | interval coverage (nominal 90 %, n) | fixed: median regret, harmful | rule regret | raw regret |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fs, e in ev.items():
            if "error" in e:
                L.append("| — | %s | error: %s |" % (fs, e["error"]))
                continue
            for k in ("grouped_cv", "leave_family_out", "leave_width_out", "confirmatory"):
                if k not in e:
                    continue
                s_ = e[k]["summary"]
                L.append("| %s | %s | %d (%s) | %s | %s | %s | %s [%s] | %s | %s [%s] | %s | %s [%s, %s] | "
                         "%s [%s, %s] | %s (%s) | %s, %s | %s | %s |" % (
                             k.replace("_", " "), fs, s_.get("rows", 0), s_.get("groups", "—"), s_.get("decided", 0),
                             _fmt(s_.get("coverage"), 2), _fmt(s_.get("accuracy"), 2), _fmt(s_.get("regret_median")),
                             _fmt(s_.get("regret_p90")), _fmt(s_.get("harmful"), 2),
                             _fmt(s_.get("policy_regret_median")), _fmt(s_.get("policy_regret_p90")),
                             _fmt(s_.get("policy_harmful"), 2), _fmt(s_.get("paired_vs_fixed_decided")),
                             _fmt(s_.get("paired_vs_fixed_decided_lo")), _fmt(s_.get("paired_vs_fixed_decided_hi")),
                             _fmt(s_.get("paired_policy_vs_fixed")), _fmt(s_.get("paired_policy_vs_fixed_lo")),
                             _fmt(s_.get("paired_policy_vs_fixed_hi")), _fmt(s_.get("interval_coverage"), 2),
                             s_.get("interval_n", "—"), _fmt(s_.get("baseline_fixed_regret_median")),
                             _fmt(s_.get("fixed_harmful"), 2), _fmt(s_.get("baseline_rule_regret_median")),
                             _fmt(s_.get("baseline_raw_regret_median"))))
        kinds = sorted({e[k]["summary"].get("interval_kind") for e in ev.values() if "error" not in e
                        for k in e if isinstance(e[k], dict) and "summary" in e[k]} - {None})
        L += ["", "Feature sets are acquisition tiers: *circuit_observable* uses only the circuit, the observable and "
              "the budget; *structural* adds the calibrated noise model (λ_LC, λ_ro, scale, coherent fraction); "
              "*with_state* adds exact ideal-state entropies (exponential in n). A tier earns its place only if it "
              "lowers held-out regret beyond the uncertainty shown. Intervals: %s. Feature weights are associations "
              "within the sampled design, not causes. A recommendation outside the training coverage is withheld "
              "('abstain-coverage')." % ("; ".join(kinds) or "—")]
        if sens and "error" not in sens:
            rg = sens.get("ranges", {})
            L += ["", "**Monte-Carlo sensitivity of this evaluation** (grouped CV, structural features, %d redraws of "
                  "the ledger's RMSEs from their Monte-Carlo errors; median relative SE %.3g, largest %.3g): "
                  % (sens["reps"], sens["median_rel_se"], sens["max_rel_se"]) + "; ".join(
                      "%s %s–%s" % (k.replace("_", " "), _fmt(v[0]), _fmt(v[1])) for k, v in rg.items()) + "."]
    L += ["", "## Limits of this report", "",
          "* Simulation on one device model; the hardware transfer test is a separate, budgeted campaign "
          "(docs/hardware_preflight.md). No hardware result is shown here.",
          "* A pilot: few base circuits per family. Intervals are wide by design; read them, not the point estimates.",
          "* The base-circuit bootstrap intervals do not contain the Monte-Carlo error of the estimated RMSEs; its "
          "effect on the labels (`p_winner`) and on the recommender evaluation (redraw ranges) is reported separately "
          "above.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(prog="python -m qemx.study", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("run_dir")
    p.add_argument("--manifest", default=None)
    p.add_argument("--rehearsal", default=None, help="a scripts/rehearse_divergence.py output directory")
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=1e5)
    p.add_argument("--eps", type=float, default=0.02)
    p = sub.add_parser("holdout", help="evaluate the frozen recommender trained on a development run on a separate "
                                       "final-holdout run (once; docs/FINAL_HOLDOUT.md)")
    p.add_argument("dev_run")
    p.add_argument("holdout_run")
    p.add_argument("--dev-manifest", default=None)
    p.add_argument("--holdout-manifest", default=None)
    p.add_argument("--feature-set", default="structural", choices=sorted(FT.SETS))
    p.add_argument("--allow-context-mismatch", action="store_true",
                   help="evaluate although protocols or noise context differ (diagnosis only; labeled in the output)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("features", help="write the feature dictionary (Markdown)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("context", help="write tables/methods.json (qemx.methods/2) of an existing report directory "
                                       "from the study's run directory (migrates a 1.5.0 qemx.methods/1 file)")
    p.add_argument("run_dir")
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", required=True, help="the report directory (its tables/ are kept)")
    a = ap.parse_args(argv)
    if a.cmd == "holdout":
        ev = evaluate_holdout(load(a.dev_run, a.dev_manifest), load(a.holdout_run, a.holdout_manifest), a.feature_set,
                              allow_context_mismatch=a.allow_context_mismatch)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "holdout_evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(dict(meta=ev["meta"], summary=ev["summary"]), f, indent=1, default=float)
        import pandas as pd
        pd.DataFrame(ev["rows"]).to_csv(os.path.join(a.out, "holdout_rows.csv"), index=False, encoding="utf-8")
        print(json.dumps(dict(meta=ev["meta"], summary=ev["summary"]), indent=1, default=float))
        return 0
    if a.cmd == "features":
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(FT.dictionary_markdown())
        return 0
    if a.cmd == "context":
        from . import rec_context as RX
        tdir = os.path.join(a.out, "tables")
        old = os.path.join(tdir, "methods.json")
        backup = None
        if os.path.exists(old):
            with open(old, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and prev.get("schema") is None:      # keep the 1.5.0 file next to the new one
                backup = os.path.join(tdir, "methods.v1.json")
                with open(backup, "w", encoding="utf-8") as f:
                    json.dump(prev, f, indent=1)
        schema = RX.detect_table_schema(tdir)
        S = load(a.run_dir, a.manifest)
        path = write_context(S, a.out, schema)
        with open(path, encoding="utf-8") as f:
            problems = json.load(f).get("problems")
        print(json.dumps(dict(written=path, previous_saved_as=backup, tables_feature_schema=schema,
                              problems=problems), indent=1))
        return 0
    res = report(a.run_dir, a.manifest, a.out, a.rehearsal, int(a.budget), a.eps)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
