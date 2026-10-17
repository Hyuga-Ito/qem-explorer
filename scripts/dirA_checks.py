#!/usr/bin/env python3
r"""Acceptance checks and summary tables for a Direction A run.

    python scripts/dirA_checks.py data/dirA_v1                       # print everything, write CSVs to data/dirA_v1/checks
    python scripts/dirA_checks.py data/dirA_v1 --eps 0.01 0.03 --out checks

From a notebook or another pipeline (no file is read or written by the checks themselves)::

    import dirA_checks as C, dirA_io as IO
    R = C.Run(IO.DirectorySource("data/dirA_v1"))            # or IO.MemorySource({...tables...}, records)
    rep = C.run_checks(R, eps=(0.01,))
    print(rep.render_text()); rep.summary; rep.tables["validation"]

Part 1 runs the pre-registered acceptance checks of the design document (§3.6) plus two provenance checks:

  env          package versions recorded by the run (run.json and every cell record) against requirements-lock.txt
  master       each cell's compiled circuit (two-qubit count, depth, layout, λ_LC) against the pre-registered master
               list docs/v13/direction_A_v1_cells.csv — differences mean the circuit is not the pre-registered one
  errors       failed cells, methods, training levels and checks, grouped by message (check 1: every entry explained)
  missing      winner rows decided while a method of that cell had failed (those tiles may show the wrong winner)
  validation   identity, KS against N(0, 1) at p > 0.01, |z| > 4 in fewer than 1 % of checks (check 2), worst rows
  stage_d      Aer (stage D) against the exact engine (A1): b and v equal to 1e-7 for RAW, ZNE-R, CDR, FR-R;
               PEC-AER within 4 Monte Carlo SE of PEC (check 3; the SE is read from the cell records)
  linearity    λ_LC(s) / (s·λ_LC(1)) within 5 % for s ≤ 2 (check 5). Check 4 (fold guard) needs no table: a failed
               guard raises inside the cell, so every cell with status ok passed it.

Part 2 writes the tables the phase diagrams need but do not show:

  achievability  per cell: the best RMSE at the largest budget, the smallest grid budget reaching each ε, and whether
                 any method reaches ε at all (the "nothing works" region); per family: the λ_LC horizon
  shares         share of (cell, budget) points won by each method, near-ties (runner-up within 5 %), per stage
  seeds          brickwork circuits that differ only in their random angles (identical λ_LC): how often the seeds
                 agree on the winner — the quick look at hypothesis A3
  kappa          κ = −ln(⟨O⟩_RAW/⟨O⟩_ideal) / λ_LC: how strongly the unmitigated signal decays per unit λ_LC
  undefined      method rows without a finite estimate (ZNE-exponential when the folded values change sign)
  transfer       stage C: winner agreement with fake_fez at the same noise scale s and at the nearest λ_LC
  variants       stages E (E-sensitivity, E-nil-sensitivity): each variant's |bias| and variance relative to its base
  transpiler     stage F: optimization level 3 against level 1 (gate counts, λ_LC, winner agreement)

Matching across stages (stage D against A1, C against A1, F against A1) uses the full cell coordinates — circuit,
noise scale, readout arm, coherent fraction, optimization level and (where it is not the varied factor) device — and
reports a match that is not unique instead of taking the first one. Circuits are told apart by their parameters, not
by their labels (a label does not show every parameter).

Only pandas, numpy and scipy are needed. Exit status 0; every failed check is reported as FAIL in the text and in
checks/summary.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dirA_io import DirectoryReportSink, Report, read_csv, read_text, source  # noqa: E402

TIE = 0.05
LOCK_KEYS = {"qiskit": "qiskit", "qiskit-aer": "qiskit_aer", "qiskit-ibm-runtime": "qiskit_ibm_runtime",
             "numpy": "numpy", "scipy": "scipy", "scikit-learn": "sklearn"}
CIRCUIT = ["family", "n", "fparams"]                 # what identifies a circuit (labels do not show every parameter)


# ------------------------------------------------------------------ loading
def label(family, n, fp):
    if family == "qft":
        return "QFT-%d" % n if fp.get("m") is None else "AQFT-%d (m=%d)" % (n, fp["m"])
    if family == "ghz":
        return "GHZ-%d%s" % (n, " (Z)" if fp.get("basis", "X") == "Z" else "")
    if family == "ising":
        return "Ising-%d (r=%d)" % (n, fp["steps"])
    if family == "brickwork":
        return "Brick-%d (L=%d, seed %d)" % (n, fp["layers"], fp["seed"])
    if family == "grover":
        return "Grover-%d (k=%d)" % (n, fp.get("k", 1))
    if family == "bv":
        return "BV-%d" % n
    if family == "context":
        return "Context-%d (L=%d, %s)" % (n, fp.get("layers", 4), fp.get("spectators", "zero"))
    return "%s-%d" % (family, n)


LEGACY_LEARNING_KEYS = ("NIL-L", "NIL-R", "NIL-L-a3", "NIL-L-a2", "NIL-R-a2", "NIL-R-32")


class Run:
    """The collected tables of one run, keys canonicalized (runs 1-2 keyed feature regression NIL-L / NIL-R; the
    thesis and direction_A_nil_v1 call it FR-L / FR-R), with one row per cell in ``cells``."""

    def __init__(self, src):
        self.src = source(src)
        self.dir = getattr(self.src, "dir", None)
        self.methods = self.src.table("methods")
        self.winners = self.src.table("winners")
        self.validation = self.src.table("validation", required=False)
        self.errors = self.src.table("errors", required=False)
        est = self.methods.groupby("method")["estimator"].first() if len(self.methods) else pd.Series(dtype=object)
        self.renamed = {k: "FR-" + k[4:] for k in LEGACY_LEARNING_KEYS if k in est.index and est[k] == "learning"}
        if self.renamed:
            for df, cols in ((self.methods, ["method"]), (self.winners, ["winner", "runner_up"]),
                             (self.validation, ["method"]), (self.errors, ["method"])):
                if df is not None and len(df):
                    for c in cols:
                        if c in df.columns:
                            df[c] = df[c].replace(self.renamed)
        m = self.methods
        fp = m["fparams"].map(json.loads)
        m["label"] = [label(f, n, p) for f, n, p in zip(m["family"], m["n"], fp)]
        cols = ["cell_id", "stage", "family", "n", "fparams", "label", "device", "engine", "scale", "readout",
                "coherent_fraction", "opt_level", "lam_lc", "two_qubit", "depth", "phys", "ideal"]
        self.cells = m[[c for c in cols if c in m.columns]].drop_duplicates("cell_id").set_index("cell_id")

    def records(self, ids=None):
        return self.src.records(ids)

    def run_info(self) -> dict:
        return self.src.run_info()


def _match(df: pd.DataFrame, row, keys) -> pd.DataFrame:
    """Rows of df equal to ``row`` on every key (floats compared with isclose)."""
    out = df
    for k in keys:
        if k not in df.columns:
            continue
        v = getattr(row, k) if not isinstance(row, (dict, pd.Series)) else row[k]
        if isinstance(v, float) and not pd.isna(v):
            out = out[np.isclose(out[k].astype(float), v)]
        elif pd.isna(v):
            out = out[out[k].isna()]
        else:
            out = out[out[k] == v]
    return out


# ------------------------------------------------------------------ part 1: acceptance and provenance
def check_env(R: Run, rep: Report, lock_text: str | None):
    rep.section("ENV · package versions of the run against the lock file")
    prov = R.run_info().get("provenance") or {}
    combos = {}
    for rec in R.records():
        p = rec.get("provenance") or {}
        key = tuple((k, str(p.get(k))) for k in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "numpy", "qemx"))
        combos[key] = combos.get(key, 0) + 1
    lock = {}
    for line in (lock_text or "").splitlines():
        mm = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line)
        if mm:
            lock[mm.group(1).lower()] = mm.group(2)
    rows = []
    for pkg, key in LOCK_KEYS.items():
        rows.append(dict(package=pkg, run=prov.get(key), lock=lock.get(pkg),
                         match=(None if lock.get(pkg) is None or prov.get(key) is None
                                else str(prov.get(key)) == lock.get(pkg))))
    df = pd.DataFrame(rows)
    rep.frame(df)
    if combos:
        rep.text("version combinations found in the cell records (cells per combination):\n" + "\n".join(
            "  %5d  %s" % (v, ", ".join("%s=%s" % kv for kv in k)) for k, v in sorted(combos.items(),
                                                                                       key=lambda kv: -kv[1])))
    bad = df[df["match"] == False]                              # noqa: E712
    if not prov:
        rep.verdict("env", None, "run.json has no provenance block")
    elif lock:
        rep.verdict("env", bad.empty, "all locked packages match" if bad.empty else
                    "differs from the lock file: " + ", ".join("%s %s (lock %s)" % (r.package, r.run, r.lock)
                                                                for r in bad.itertuples()))
    if len(combos) > 1:
        rep.verdict("env-mixed", False, "%d different version combinations among the cell records" % len(combos))
    return df


def check_master(R: Run, rep: Report, M: pd.DataFrame | None):
    rep.section("MASTER · compiled circuits against the pre-registered master list")
    if M is None:
        rep.verdict("master", None, "master list not given")
        return pd.DataFrame()
    need = {"cell_id", "two_qubit", "depth", "phys", "lam_lc"}
    if not need <= set(M.columns):
        rep.verdict("master", None, "master list has no λ columns (make it with: run masterlist --with-lambda)")
        return pd.DataFrame()
    c = R.cells.reset_index()
    x = c.merge(M[list(need)], on="cell_id", how="left", suffixes=("", "_master"))
    unknown = x["two_qubit_master"].isna()
    x = x[~unknown].copy()

    def norm_phys(v):
        try:
            return tuple(json.loads(v)) if isinstance(v, str) else tuple(v)
        except (TypeError, ValueError):
            return None
    x["d_two_qubit"] = x["two_qubit"].astype(float) - x["two_qubit_master"].astype(float)
    x["d_depth"] = x["depth"].astype(float) - x["depth_master"].astype(float)
    x["same_phys"] = [norm_phys(a) == norm_phys(b) for a, b in zip(x["phys"], x["phys_master"])]
    x["lam_ratio"] = x["lam_lc"].astype(float) / x["lam_lc_master"].astype(float)
    x["differs"] = (x["d_two_qubit"] != 0) | (x["d_depth"] != 0) | ~x["same_phys"] | ((x["lam_ratio"] - 1).abs() > 1e-5)
    d = x[x["differs"]]
    rep.text("cells compared: %d (not in the master list: %d)" % (len(x), int(unknown.sum())))
    if x.empty:
        rep.verdict("master", None, "no cell of this run is in the master list (different manifest?)")
        return pd.DataFrame()
    if len(d):
        rep.frame(d.groupby(["stage", "label"]).agg(cells=("lam_ratio", "size"), d_two_qubit=("d_two_qubit", "first"),
                                                    d_depth=("d_depth", "first"), lam_ratio=("lam_ratio", "median"),
                                                    same_phys=("same_phys", "all")))
    rep.verdict("master", d.empty, "every compiled circuit matches the master list" if d.empty else
                "%d cells (%d circuits) differ from the pre-registered compilation" % (len(d), d["label"].nunique()))
    return x[["cell_id", "stage", "label", "scale", "two_qubit", "two_qubit_master", "depth", "depth_master",
              "same_phys", "lam_lc", "lam_lc_master", "lam_ratio", "differs"]]


def check_errors(R: Run, rep: Report):
    rep.section("ERRORS · failed cells, methods, training levels and checks (check 1: every entry explained)")
    E = R.errors
    if E is None or E.empty:
        rep.verdict("errors", True, "errors table is empty")
        return pd.DataFrame()
    E = E.copy()
    E["kind"] = np.where(E["method"].isna() | (E["method"].astype(str) == ""), "whole cell", "one method")
    E["message"] = E["error"].fillna("").astype(str).str.strip().str.split("\n").str[-1].str.slice(0, 120)
    g = E.groupby(["stage", "kind", "status", "method", "message"], dropna=False).size().rename("count").reset_index()
    rep.frame(g.sort_values("count", ascending=False))
    rep.verdict("errors", None, "%d entries (%d whole cells, %d single methods, levels or checks): explain each "
                "group above" % (len(E), int((E["kind"] == "whole cell").sum()), int((E["kind"] == "one method").sum())))
    return g


def check_missing(R: Run, rep: Report):
    rep.section("MISSING · winners decided without a method that failed in that cell")
    w = R.winners
    miss = w[w["missing"].fillna("").astype(str) != ""] if "missing" in w.columns else w.iloc[0:0]
    if miss.empty:
        rep.verdict("missing", True, "no winner row lacks a method")
        return pd.DataFrame()
    x = miss.drop_duplicates("cell_id").join(R.cells, on="cell_id")
    rep.frame(x.groupby(["stage", "missing"]).agg(cells=("cell_id", "size"),
                                                  circuits=("label", lambda s: ", ".join(sorted(set(s))[:6]))))
    rep.verdict("missing", False, "%d cells: their tiles may show the wrong winner (mark or exclude them)" %
                x["cell_id"].nunique())
    return x[["stage", "label", "scale", "coherent_fraction", "missing"]].reset_index()


def check_validation(R: Run, rep: Report):
    rep.section("VALIDATION · Monte Carlo of complete experiments against the ledger (check 2)")
    V = R.validation
    if V is None or V.empty:
        rep.verdict("validation", None, "no validation table")
        return pd.DataFrame()
    from scipy import stats
    if "identity" in V:
        idn = V["identity"].dropna().astype(bool)
        rep.verdict("validation-identity", bool(idn.all()), "%d of %d checks hold the budget identity" % (
            int(idn.sum()), len(idn)))
    ok = V["feasible"].fillna(False).astype(bool) & V["z_rmse"].notna() & V["independent"].fillna(False).astype(bool)
    Z = V[ok].join(R.cells[["stage", "label", "scale", "coherent_fraction", "lam_lc"]], on="cell_id")
    Z = Z.assign(z_rmse=Z["z_rmse"].astype(float), z_bias=Z["z_bias"].astype(float))
    for col in ("z_rmse", "z_bias"):
        z = Z[col].dropna().values
        if len(z) < 2:
            rep.verdict("validation-KS-" + col, None, "fewer than two checks")
            continue
        fin = z[np.isfinite(z)]
        p = stats.kstest(fin, "norm").pvalue if len(fin) > 1 else float("nan")
        big = float(np.mean(np.abs(z) > 4))
        inner = z[np.abs(z) <= 5]
        rep.text("%s: n=%d (%d infinite: a nonzero difference with zero standard error) mean %.3f sd %.3f | "
                 "inside ±5: n=%d sd %.3f | outside ±5: %d | KS p=%.3g | |z|>4: %.2f %%"
                 % (col, len(z), int((~np.isfinite(z)).sum()), fin.mean() if len(fin) else float("nan"),
                    fin.std(ddof=1) if len(fin) > 1 else float("nan"), len(inner),
                    inner.std(ddof=1) if len(inner) > 1 else float("nan"), int(np.sum(np.abs(z) > 5)), p, 100 * big))
        rep.verdict("validation-KS-" + col, bool(p > 0.01) and bool(np.isfinite(z).all()),
                    "KS p = %.3g (pre-registered: > 0.01)%s" % (p, "" if np.isfinite(z).all() else
                                                                 "; %d infinite z" % int((~np.isfinite(z)).sum())))
        rep.verdict("validation-tail-" + col, bool(big < 0.01),
                    "|z| > 4 in %.2f %% of %d checks (pre-registered: < 1 %%)" % (100 * big, len(z)))
    Z = Z.assign(zmax=np.maximum(Z["z_rmse"].abs(), Z["z_bias"].abs()))
    rep.text("by stage and method:")
    rep.frame(Z.groupby(["stage", "method"]).agg(checks=("zmax", "size"), sd_z_rmse=("z_rmse", "std"),
                                                 sd_z_bias=("z_bias", "std"),
                                                 over4=("zmax", lambda s: int((s > 4).sum()))))
    if "N" in Z.columns:
        Z = Z.assign(N_bin=pd.cut(Z["N"].astype(float), [0, 1e3, 1e4, 1e5, 1e6, 1e9],
                                  labels=["<1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", ">1e6"]))
        nl = Z.groupby(["method", "N_bin"], observed=True).agg(checks=("zmax", "size"),
                                                                 over4=("zmax", lambda s: int((s > 4).sum())),
                                                                 mean_z_bias=("z_bias", "mean"))
        rep.text("|z| > 4 by method and number of target shots N (finite-N effects of nonlinear estimators show up "
                 "at small N):")
        rep.frame(nl[nl.checks > 0])
    if "mc_frac_nonfinite" in Z.columns:
        nf = Z[Z["mc_frac_nonfinite"].fillna(0) > 0]
        rep.verdict("validation-nonfinite", None, "%d checks dropped non-finite experiments (they condition the "
                    "Monte Carlo sample; %d of them have |z| > 4)" % (len(nf), int((nf["zmax"] > 4).sum())))
    cols = [c for c in ["stage", "label", "scale", "coherent_fraction", "method", "policy", "B", "N", "s_tr", "z_rmse",
                        "z_bias", "var_ratio", "mc_frac_nonfinite", "mc_bias", "an_b", "mc_rmse", "an_rmse"]
            if c in Z.columns]
    rep.text("worst 25 checks:")
    rep.frame(Z.sort_values("zmax", ascending=False)[cols].head(25))
    return Z[cols + ["cell_id"]].assign(zmax=Z["zmax"]).sort_values("zmax", ascending=False)


def check_stage_d(R: Run, rep: Report, tol=1e-7, nse=4.0, stage="D-aer-check", ref_stage="A1-primary"):
    rep.section("STAGE D · Aer (E3) against the exact engine (A1) (check 3)")
    m = R.methods
    D = m[m["stage"] == stage]
    if D.empty:
        rep.verdict("stage-D", None, "no stage-D cells in this run")
        return pd.DataFrame()
    A = m[m["stage"] == ref_stage]
    key = CIRCUIT + ["scale", "device", "readout", "coherent_fraction", "opt_level"]
    mc_err = {}
    for rec in R.records(set(D["cell_id"])):
        for r in rec.get("methods") or []:
            mc_err[(rec["cell_id"], r["key"])] = (r.get("details") or {}).get("mc_err")
    rows, ambiguous = [], 0
    for (cid, meth), d in D.groupby(["cell_id", "method"]):
        d = d.iloc[0]
        ref_m = "PEC" if meth == "PEC-AER" else meth
        a = _match(A[A["method"] == ref_m], d, key)
        if a.empty:
            continue
        if len(a) > 1:
            ambiguous += 1
            continue
        a = a.iloc[0]
        row = dict(label=d["label"], scale=d["scale"], method=meth, b_D=d["b"], b_A1=a["b"], db=d["b"] - a["b"],
                   v_D=d["v"], v_A1=a["v"], dv=d["v"] - a["v"])
        if meth == "PEC-AER":
            se = mc_err.get((cid, meth))
            row["mc_err"] = se
            row["ok"] = bool(se is not None and se > 0 and abs(row["db"]) <= nse * se)
        else:
            row["ok"] = bool(abs(row["db"]) <= tol and abs(row["dv"]) <= tol * max(1.0, abs(a["v"])))
        rows.append(row)
    df = pd.DataFrame(rows)
    if ambiguous:
        rep.verdict("stage-D-match", False, "%d stage-D rows match several A1 cells (coordinates not unique)" %
                    ambiguous)
    if df.empty:
        rep.verdict("stage-D", None, "no A1 partner cells found for the stage-D cells")
        return df
    rep.frame(df)
    rep.verdict("stage-D", bool(df["ok"].all()), "%d of %d comparisons within tolerance (exact: %g; PEC-AER: %g SE)"
                % (int(df["ok"].sum()), len(df), tol, nse))
    return df


def check_linearity(R: Run, rep: Report, smax=2.0, tol=0.05):
    rep.section("LINEARITY · λ_LC(s) / (s·λ_LC(1)) for s ≤ 2 (check 5)")
    c = R.cells.reset_index()
    keys = ["stage", "device", "readout", "coherent_fraction", "opt_level"] + CIRCUIT
    keys = [k for k in keys if k in c.columns]
    rows = []
    for k, g in c.groupby(keys, dropna=False):
        kk = dict(zip(keys, k))
        one = g[np.isclose(g["scale"].astype(float), 1.0)]
        if one.empty:
            continue
        if len(one) > 1:
            rep.verdict("linearity-match", False, "%d cells at s = 1 for %s" % (len(one), kk))
            continue
        l1 = float(one["lam_lc"].iloc[0])
        for r in g[g["scale"].astype(float) <= smax].itertuples():
            rows.append(dict(stage=kk["stage"], device=kk.get("device"), coherent_fraction=kk.get("coherent_fraction"),
                             label=r.label, fparams=kk.get("fparams"), scale=r.scale,
                             ratio=float(r.lam_lc) / (float(r.scale) * l1)))
    df = pd.DataFrame(rows)
    if df.empty:
        rep.verdict("linearity", None, "no cells at s = 1")
        return df
    bad = df[(df["ratio"] - 1).abs() > tol]
    rep.text("ratio range: %.4f – %.4f over %d cells" % (df["ratio"].min(), df["ratio"].max(), len(df)))
    if len(bad):
        rep.frame(bad)
    rep.verdict("linearity", bad.empty, "%d of %d cells within %g %%" % (len(df) - len(bad), len(df), 100 * tol))
    return df


def undefined_methods(R: Run, rep: Report):
    rep.section("UNDEFINED · method rows without a finite estimate (status nonfinite / error)")
    m = R.methods
    bad = m[m["status"] != "ok"]
    if bad.empty:
        rep.verdict("undefined", True, "every method produced a finite estimate in every cell")
        return pd.DataFrame()
    rep.frame(bad.groupby(["stage", "method", "status", "family"]).size().rename("cells").reset_index())
    if "coherent_fraction" in bad:
        rep.text("by coherent fraction:")
        rep.frame(bad.groupby(["stage", "coherent_fraction"]).size().rename("rows").reset_index())
    rep.verdict("undefined", None, "%d method rows without an estimate (e.g. ZNE-exponential when the folded values "
                "change sign); those cells' winners were decided without them" % len(bad))
    return bad[["cell_id", "stage", "label", "scale", "coherent_fraction", "device", "method", "status"]]


def _w(R: Run, policy, candidates):
    w = R.winners
    return w[(w["policy"] == policy) & (w["candidates"] == candidates)]


def _winner_matrix(R: Run, policy="fixed", candidates="no_optimistic"):
    return _w(R, policy, candidates).pivot_table(index="cell_id", columns="B", values="winner", aggfunc="first")


def transfer(R: Run, rep: Report, policy="fixed", candidates="no_optimistic", ref_stage="A1-primary",
             stage="C-transfer"):
    rep.section("TRANSFER · stage C: does the winner follow s, or λ_LC? (%s policy, %s)" % (policy, candidates))
    c = R.cells.reset_index()
    C, A = c[c["stage"] == stage], c[(c["stage"] == ref_stage) & (c["device"] == "fake_fez")]
    if C.empty or A.empty:
        rep.verdict("transfer", None, "no %s cells (or no %s reference)" % (stage, ref_stage))
        return pd.DataFrame()
    W = _winner_matrix(R, policy, candidates)
    key = CIRCUIT + ["readout", "coherent_fraction", "opt_level"]
    rows, ambiguous = [], 0
    for r in C.itertuples():
        ref = _match(A, r, key)
        if ref.empty or r.cell_id not in W.index:
            continue
        same_s = ref[np.isclose(ref["scale"].astype(float), float(r.scale))]
        if len(same_s) > 1:
            ambiguous += 1
            continue
        near = ref.iloc[[int(np.argmin(np.abs(np.log(ref["lam_lc"].astype(float) / float(r.lam_lc)))))]]
        row = dict(device=r.device, label=r.label, scale=r.scale, lam_lc=float(r.lam_lc))
        for tag, rr in (("same_s", same_s), ("same_lam", near)):
            if rr.empty or rr["cell_id"].iloc[0] not in W.index:
                row["agree_" + tag] = np.nan
                continue
            a, b = W.loc[r.cell_id], W.loc[rr["cell_id"].iloc[0]]
            ok = a.notna() & b.notna()
            row["agree_" + tag] = float((a[ok] == b[ok]).mean()) if ok.any() else np.nan
            row["lam_ratio_" + tag] = float(r.lam_lc) / float(rr["lam_lc"].iloc[0])
        rows.append(row)
    if ambiguous:
        rep.verdict("transfer-match", False, "%d transfer cells match several reference cells" % ambiguous)
    df = pd.DataFrame(rows)
    if df.empty:
        rep.verdict("transfer", None, "no matched cells")
        return df
    s = df.groupby("device").agg(cells=("label", "size"), agree_same_s=("agree_same_s", "mean"),
                                 agree_same_lam=("agree_same_lam", "mean"),
                                 lam_ratio_same_s=("lam_ratio_same_s", "median"),
                                 lam_ratio_same_lam=("lam_ratio_same_lam", "median"))
    rep.text("share of budget points with the same winner as on fake_fez, matching either the noise scale s or the "
             "nearest λ_LC (λ_ratio: device / fez):")
    rep.frame(s.round(3))
    rep.verdict("transfer", None, "winner agreement with fake_fez: %.0f %% at equal s, %.0f %% at the nearest λ_LC" % (
        100 * df["agree_same_s"].mean(), 100 * df["agree_same_lam"].mean()))
    return df


VARIANTS = {"ZNE-R123": "ZNE-R", "ZNE-LIN": "ZNE-R", "ZNE-R-RC": "ZNE-R", "PEC-GATE": "PEC", "CDR-32": "CDR",
            "FR-L-a3": "FR-L", "FR-L-a2": "FR-L", "FR-R-a2": "FR-R", "FR-R-32": "FR-R", "VD-3": "VD",
            # direction_A_nil_v1, stage E-nil-sensitivity
            "NIL-G1": "NIL-G2", "NIL-G3": "NIL-G2", "NIL-G5": "NIL-G2", "NIL-LASSO-a3": "NIL-LASSO",
            "NIL-LASSO-a2": "NIL-LASSO", "NIL-RIDGE-a2": "NIL-RIDGE", "NIL-RIDGE-a4": "NIL-RIDGE",
            "NIL-G2-T32": "NIL-G2", "NIL-G2-S6": "NIL-G2", "NIL-G2-S24": "NIL-G2", "NIL-G2-NB": "NIL-G2",
            "NIL-G2-INT": "NIL-G2", "NIL-G2-EQ": "NIL-G2", "NIL-G2-UNI": "NIL-G2"}
VARIANT_STAGES = ("E-sensitivity", "E-nil-sensitivity")


def variants(R: Run, rep: Report, stages=VARIANT_STAGES):
    rep.section("VARIANTS · stage E: each pre-registered variant against its base method in the same cell")
    m = R.methods[R.methods["stage"].isin(stages) & (R.methods["status"] == "ok")]
    if m.empty:
        rep.verdict("variants", None, "no cells of %s" % ", ".join(stages))
        return pd.DataFrame()
    rows = []
    for cid, g in m.groupby("cell_id"):
        g = g.set_index("method")
        for var, base in VARIANTS.items():
            if var in g.index and base in g.index:
                rows.append(dict(stage=g["stage"].iloc[0], label=g["label"].iloc[0], scale=g["scale"].iloc[0],
                                 variant=var, base=base, b_base=g.at[base, "b"], b_var=g.at[var, "b"],
                                 bias_ratio=abs(g.at[var, "b"]) / max(abs(g.at[base, "b"]), 1e-12),
                                 v_ratio=g.at[var, "v"] / max(g.at[base, "v"], 1e-300)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    s = df.groupby(["stage", "variant"]).agg(cells=("label", "size"), median_bias_ratio=("bias_ratio", "median"),
                                             min_bias_ratio=("bias_ratio", "min"),
                                             max_bias_ratio=("bias_ratio", "max"),
                                             median_v_ratio=("v_ratio", "median"))
    rep.text("|b_variant| / |b_base| and v_variant / v_base (below 1 = the variant is better on that axis):")
    rep.frame(s.round(3))
    return df


def transpiler(R: Run, rep: Report, stage="F-transpiler", ref_stage="A1-primary", policy="fixed",
               candidates="no_optimistic"):
    rep.section("TRANSPILER · stage F (optimization level 3) against the same circuit and scale at level 1")
    c = R.cells.reset_index()
    F, A = c[c["stage"] == stage], c[c["stage"] == ref_stage]
    if F.empty:
        rep.verdict("transpiler", None, "no %s cells" % stage)
        return pd.DataFrame()
    W = _winner_matrix(R, policy, candidates)
    key = CIRCUIT + ["device", "scale", "readout", "coherent_fraction"]
    rows, ambiguous = [], 0
    for r in F.itertuples():
        ref = _match(A, r, key)
        if ref.empty:
            continue
        if len(ref) > 1:
            ambiguous += 1
            continue
        q = ref.iloc[0]
        row = dict(label=r.label, scale=r.scale, two_qubit_o3=r.two_qubit, two_qubit_o1=q["two_qubit"],
                   depth_o3=r.depth, depth_o1=q["depth"], lam_ratio=float(r.lam_lc) / float(q["lam_lc"]))
        if r.cell_id in W.index and q["cell_id"] in W.index:
            a, b = W.loc[r.cell_id], W.loc[q["cell_id"]]
            ok = a.notna() & b.notna()
            row["winner_agreement"] = float((a[ok] == b[ok]).mean())
        rows.append(row)
    if ambiguous:
        rep.verdict("transpiler-match", False, "%d stage-F cells match several level-1 cells" % ambiguous)
    df = pd.DataFrame(rows)
    rep.frame(df.round(3))
    if "winner_agreement" in df:
        rep.verdict("transpiler", None, "optimization level 3 keeps the level-1 winner on %.0f %% of budget points" %
                    (100 * df["winner_agreement"].mean()))
    return df


# ------------------------------------------------------------------ part 2: summary tables
def achievability(R: Run, rep: Report, eps=(0.01,), policy="fixed", candidates="no_optimistic"):
    rep.section("ACHIEVABILITY · can any method reach ε, and at which budget? (%s policy, %s)" % (policy, candidates))
    w = _w(R, policy, candidates)
    if w.empty:
        rep.verdict("achievability", None, "no winner rows")
        return pd.DataFrame(), pd.DataFrame()
    Bmax = w["B"].max()
    top = w[w["B"] == Bmax].set_index("cell_id")
    out = R.cells.join(top[["winner", "rmse"]], how="inner").rename(columns={"winner": "best_at_Bmax",
                                                                              "rmse": "rmse_at_Bmax"})
    for e in eps:
        hit = w[w["rmse"] <= e].groupby("cell_id")["B"].min()
        out["Bmin_eps%g" % e] = hit.reindex(out.index)
        out["reach_eps%g" % e] = out["Bmin_eps%g" % e].notna()
    out["rel_rmse_at_Bmax"] = out["rmse_at_Bmax"] / out["ideal"].abs().where(out["ideal"].abs() > 0)
    keys = ["stage", "family", "n", "coherent_fraction", "device"]
    rows = []
    for k, g in out.groupby(keys, dropna=False):
        row = dict(zip(keys, k), cells=len(g))
        for e in eps:
            r = g["reach_eps%g" % e]
            row["reachable_eps%g" % e] = "%d/%d" % (int(r.sum()), len(r))
            row["max_lam_reached_eps%g" % e] = g.loc[r, "lam_lc"].max() if r.any() else float("nan")
            row["min_lam_failed_eps%g" % e] = g.loc[~r, "lam_lc"].min() if (~r).any() else float("nan")
        row["median_|ideal|"] = g["ideal"].abs().median()
        rows.append(row)
    S = pd.DataFrame(rows)
    rep.text("B grid top = %g. A family's λ horizon lies between the last λ reached and the first λ failed (they "
             "overlap when the reachable set is not an interval in λ)." % Bmax)
    rep.frame(S)
    small = out[out["ideal"].abs() < 0.1]
    if len(small):
        rep.verdict("achievability-scale", None, "%d cells have |⟨O⟩_ideal| < 0.1 (e.g. %s): absolute RMSE flatters "
                    "them; also read rel_rmse_at_Bmax" % (len(small), ", ".join(sorted(small["label"].unique())[:3])))
    return out.reset_index(), S


def shares(R: Run, rep: Report):
    rep.section("SHARES · who wins how much of each plane (share of cell × budget points)")
    w = R.winners.join(R.cells[["stage", "coherent_fraction"]], on="cell_id")
    w = w.assign(tie=(w["margin"] < TIE))
    rows = []
    for k, g in w.groupby(["stage", "coherent_fraction", "policy", "candidates"], dropna=False):
        row = dict(stage=k[0], coherent_fraction=k[1], policy=k[2], candidates=k[3], points=len(g),
                   near_tie=round(100 * g["tie"].mean(), 1))
        for meth, f in g["winner"].value_counts(normalize=True).items():
            row[meth] = round(100 * f, 1)
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0)
    rep.text("percent of points (near_tie: runner-up within 5 %):")
    rep.frame(df)
    return df


def seeds(R: Run, rep: Report, policy="fixed", candidates="no_optimistic"):
    rep.section("SEEDS · brickwork circuits with identical structure and λ_LC, different random angles (A3 quick look)")
    c = R.cells[R.cells["family"] == "brickwork"].copy()
    if c.empty:
        rep.verdict("seeds", None, "no brickwork cells")
        return pd.DataFrame()
    fp = c["fparams"].map(json.loads)
    c["layers"] = [p.get("layers") for p in fp]
    c["seed"] = [p.get("seed") for p in fp]
    w = _w(R, policy, candidates)
    Bmax = w["B"].max()
    rows = []
    keys = ["stage", "n", "layers", "scale", "coherent_fraction", "device", "readout", "opt_level"]
    keys = [k for k in keys if k in c.columns]
    for k, g in c.groupby(keys, dropna=False):
        if g["seed"].nunique() < 2:
            continue
        kk = dict(zip(keys, k))
        sub = w[w["cell_id"].isin(g.index)]
        piv = sub.pivot_table(index="B", columns="cell_id", values="winner", aggfunc="first")
        agree = piv.nunique(axis=1).eq(1)
        top = sub[sub["B"] == Bmax]
        rows.append(dict(kk, seeds=g["seed"].nunique(), lam_lc_spread=float(g["lam_lc"].max() / g["lam_lc"].min() - 1),
                         agree_pct=round(100 * agree.mean(), 1),
                         winners_at_Bmax=",".join(sorted(top["winner"].unique())),
                         rmse_ratio_at_Bmax=float(top["rmse"].max() / top["rmse"].min()) if len(top) else np.nan,
                         ideal_values=",".join("%.3f" % v for v in g.sort_values("seed")["ideal"])))
    df = pd.DataFrame(rows)
    if df.empty:
        rep.verdict("seeds", None, "no group with two or more seeds")
        return df
    rep.frame(df)
    rep.text("Overall agreement of the seeds on the winner: %.1f %% of budget points (median over groups %.1f %%)" % (
        df["agree_pct"].mean(), df["agree_pct"].median()))
    rep.verdict("seeds", None, "seeds agree on %.1f %% of points on average; λ_LC spread ≤ %.1e" % (
        df["agree_pct"].mean(), df["lam_lc_spread"].max()))
    return df


def kappa(R: Run, rep: Report, min_ideal=0.1):
    rep.section("KAPPA · decay of the unmitigated signal per unit λ_LC: ⟨O⟩_RAW = ⟨O⟩_ideal · exp(−κ λ_LC)")
    m = R.methods
    r = m[(m["method"] == "RAW") & (m["status"] == "ok") & (m["ideal"].abs() >= min_ideal)].copy()
    ratio = (r["ideal"] + r["b"]) / r["ideal"]
    r = r[(ratio > 0) & (r["lam_lc"] > 0)]
    r["kappa"] = -np.log((r["ideal"] + r["b"]) / r["ideal"]) / r["lam_lc"]
    per = r.groupby(["stage"] + CIRCUIT).agg(label=("label", "first"), cells=("kappa", "size"),
                                            kappa_median=("kappa", "median"), kappa_min=("kappa", "min"),
                                            kappa_max=("kappa", "max"), ideal=("ideal", "first"))
    fam = r.groupby(["stage", "family"]).agg(circuits=("fparams", "nunique"), kappa_median=("kappa", "median"),
                                            kappa_q10=("kappa", lambda s: s.quantile(0.1)),
                                            kappa_q90=("kappa", lambda s: s.quantile(0.9)))
    rep.text("per family (cells with |⟨O⟩_ideal| ≥ %g):" % min_ideal)
    rep.frame(fam)
    rep.text("per circuit:")
    rep.frame(per)
    return per.reset_index()


# ------------------------------------------------------------------ all checks: computation only
def run_checks(R: Run, eps=(0.01, 0.03), lock_text: str | None = None, master: pd.DataFrame | None = None) -> Report:
    """Every check and summary table of this module on one run; the report holds the text, verdicts and tables."""
    rep = Report()
    rep.table("env", check_env(R, rep, lock_text))
    rep.table("master_diff", check_master(R, rep, master))
    rep.table("errors", check_errors(R, rep))
    rep.table("missing", check_missing(R, rep))
    rep.table("validation", check_validation(R, rep))
    rep.table("stage_d", check_stage_d(R, rep))
    rep.table("linearity", check_linearity(R, rep))
    cells, fam = achievability(R, rep, tuple(eps))
    rep.table("achievability_cells", cells)
    rep.table("achievability_families", fam)
    rep.table("shares", shares(R, rep))
    rep.table("undefined", undefined_methods(R, rep))
    rep.table("transfer", transfer(R, rep))
    rep.table("variants", variants(R, rep))
    rep.table("transpiler", transpiler(R, rep))
    rep.table("seeds", seeds(R, rep))
    rep.table("kappa", kappa(R, rep))
    rep.section("SUMMARY")
    rep.frame(rep.summary)
    return rep


# ------------------------------------------------------------------ driver: input → computation → output
def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="the run directory (contains tables/, cells/, run.json)")
    ap.add_argument("--master", default=os.path.join(root, "docs", "v13", "direction_A_v1_cells.csv"))
    ap.add_argument("--lock", default=os.path.join(root, "requirements-lock.txt"))
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.03])
    ap.add_argument("--out", default=None, help="directory for the CSV tables and the report (default: RUN/checks)")
    ap.add_argument("--quiet", action="store_true", help="do not print the report")
    a = ap.parse_args(argv)
    R = Run(a.run)                                                   # input
    rep = run_checks(R, tuple(a.eps), read_text(a.lock), read_csv(a.master))   # computation
    out = a.out or os.path.join(a.run, "checks")
    DirectoryReportSink(out).write(rep)                              # output
    if not a.quiet:
        sys.stdout.write(rep.render_text())
        print("CSV tables, summary.csv and report.md written to %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
