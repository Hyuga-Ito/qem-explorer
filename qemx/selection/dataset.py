"""Decision dataset: one row per (task, budget) with features, per-protocol labels of both halves, feasibility and
ancestry (1.6.0; 1.7.0 failure semantics). Built from a labels directory (``labels.label_task``) — the only input of
models, policies, evaluation, certification and figures, so every result regenerates from saved trials.

1.7.0 corrections (review findings S2, S4, R1):

* records without labels (``error``, ``timeout``, ``memory``) are collected in ``Dataset.failures`` instead of
  crashing the collection (their arrays are never opened); an ``ok`` record with missing or inconsistent arrays raises
  ``labels.CorruptedRecord``;
* ``fail`` is the share of evaluation-half trials with *at least one* failure or fallback flag (the union); 1.6.0
  summed the four flag rates, which counts a trial with two flags twice. The individual rates are in ``flag_rates``;
* ``SE_eval`` is the standard error matching the labels' estimand (design-level for learning protocols under fresh
  designs, ``labels.risk_se``); the trial-level one is ``SE_eval_trial``;
* ``task_ids`` restricts the collection to a run's task manifest (``run.collect``), so stale task files of another
  grid are never mixed in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import features as FE
from . import labels as LB
from . import protocols as PR


@dataclass
class Dataset:
    protocols: list
    budgets: list
    rows: list = field(default_factory=list)             # dicts: task_id, base_id, family, n, device, truth, B, feats
    Y_sel: np.ndarray = None                             # (rows, P) MSE of the selection half (NaN infeasible)
    Y_eval: np.ndarray = None                            # (rows, P) MSE of the evaluation half
    SE_eval: np.ndarray = None                           # (rows, P) standard error of Y_eval (matching the estimand)
    F: np.ndarray = None                                 # (rows, P) feasible
    Yc_eval: np.ndarray = None                           # (rows, P) clipped-estimate MSE (certificates)
    fail: np.ndarray = None                              # (rows, P) share of trials with any failure/fallback flag
    SE_eval_trial: np.ndarray = None                     # (rows, P) trial-level standard error of Y_eval
    flag_rates: np.ndarray = None                        # (rows, P, flags) individual flag rates (evaluation half)
    ERR: np.ndarray = None                               # (rows, P) the method could not be run (software failure)
    failures: list = field(default_factory=list)         # records without labels: task_id, status, kind, error
    design_mode: str = "fixed-shared"

    @property
    def groups(self) -> np.ndarray:
        return np.array([r["base_id"] for r in self.rows])

    def col(self, name: str) -> np.ndarray:
        return np.array([r.get(name) for r in self.rows])

    def X(self, names) -> np.ndarray:
        return FE.matrix([dict(r["feats"], log10_budget=math.log10(r["B"])) for r in self.rows], names)

    def subset(self, idx) -> "Dataset":
        idx = np.asarray(idx, int)

        def sub(a):
            return None if a is None else a[idx]
        return Dataset(self.protocols, self.budgets, [self.rows[i] for i in idx], self.Y_sel[idx], self.Y_eval[idx],
                       self.SE_eval[idx], self.F[idx], self.Yc_eval[idx], self.fail[idx], sub(self.SE_eval_trial),
                       sub(self.flag_rates), sub(self.ERR), list(self.failures), self.design_mode)


def with_protocols(ds: Dataset, keys) -> Dataset:
    """The same rows with only the candidates ``keys`` (1.8.0: a portfolio restricted to a subset, e.g. portfolio-1.6
    inside a portfolio-1.8 study, so that a larger portfolio and learned selection are compared on identical tasks
    and labels)."""
    j = [ds.protocols.index(k) for k in keys]

    def col(a):
        return None if a is None else a[:, j]
    return Dataset(list(keys), ds.budgets, list(ds.rows), col(ds.Y_sel), col(ds.Y_eval), col(ds.SE_eval), col(ds.F),
                   col(ds.Yc_eval), col(ds.fail), col(ds.SE_eval_trial),
                   None if ds.flag_rates is None else ds.flag_rates[:, j], col(ds.ERR), list(ds.failures),
                   ds.design_mode)


def build(out_dir: str, task_ids=None) -> Dataset:
    """Every labeled task and budget of a labels directory (or only ``task_ids``, a run's manifest)."""
    ids = LB.task_ids(out_dir) if task_ids is None else sorted(task_ids)
    protocols, budgets = None, None
    rows, ys, ye, se, fs, yc, fl, set_, fr, er = [], [], [], [], [], [], [], [], [], []
    failures, modes = [], set()
    for tid in ids:
        rec = LB.load_record(out_dir, tid)
        if rec.get("status") != "ok":
            failures.append(dict(task_id=tid, base_id=rec.get("base_id"), status=rec.get("status"),
                                 error_kind=rec.get("error_kind", ""), error=str(rec.get("error", ""))[:300]))
            continue
        arr = LB.load_arrays(out_dir, tid, rec)
        if protocols is None:
            protocols, budgets = list(rec["protocols"]), list(rec["budgets"])
        if list(rec["protocols"]) != protocols or list(rec["budgets"]) != budgets:
            raise ValueError("task %s was labeled with other protocols or budgets" % tid)
        plan = LB.plan_of(rec)
        modes.add(plan.get("mode", "fixed-shared"))
        mu = float(rec["ideal"])
        est, flags, design = arr["est"], arr["flags"], arr["design"]
        nB = len(budgets)
        for j, B in enumerate(budgets):
            allocs = [rec["allocations"][i * nB + j] for i in range(len(protocols))]
            feas = np.array([bool(a["feasible"]) for a in allocs])
            err = np.array([LB.allocation_status(a) == "method-error" for a in allocs])
            row_s, row_e, row_se, row_c, row_f, row_set, row_fr = [], [], [], [], [], [], []
            for i in range(len(protocols)):
                if not feas[i]:
                    for lst in (row_s, row_e, row_se, row_c, row_f, row_set):
                        lst.append(np.nan)
                    row_fr.append(np.full(flags.shape[3], np.nan))
                    continue
                xs, xe = est[0, i, j], est[1, i, j]
                es = np.where(np.isfinite(xs), xs, 0.0) - mu
                ee = np.where(np.isfinite(xe), xe, 0.0) - mu
                ec = np.clip(np.where(np.isfinite(xe), xe, 0.0), -1, 1) - mu
                row_s.append(float(np.mean(es ** 2)))
                row_e.append(float(np.mean(ee ** 2)))
                st = LB.design_stats(ee ** 2, design[1, i, j])
                row_se.append(float(LB.risk_se(st, plan, PR.PORTFOLIO[protocols[i]].learning)))
                row_set.append(float(st["se_trial"]))
                row_c.append(float(np.mean(ec ** 2)))
                f_ = flags[1, i, j]                                      # (flags, R)
                row_f.append(float(f_.any(axis=0).mean()))               # union: a trial counts once
                row_fr.append(f_.mean(axis=1))
            sp = rec["spec"]
            rows.append(dict(task_id=tid, base_id=rec["base_id"], family=sp["family"], n=int(rec["features"].get(
                "n", sp["n"])), device=sp["device"], truth=sp.get("truth", ""), observable=sp["observable"],
                inputs=sp["inputs"], layout=sp["layout"], B=int(B), ideal=mu, feats=rec.get("features") or {}))
            ys.append(row_s), ye.append(row_e), se.append(row_se), fs.append(feas), yc.append(row_c), fl.append(row_f)
            set_.append(row_set), fr.append(row_fr), er.append(err)
    if protocols is None:
        raise ValueError("no labeled tasks in %s%s" % (out_dir, " (%d records without labels)" % len(failures)
                                                       if failures else ""))
    if len(modes) > 1:
        raise ValueError("%s mixes labels of different design plans (%s): their risks estimate different "
                         "quantities" % (out_dir, ", ".join(sorted(modes))))
    return Dataset(protocols, budgets, rows, np.array(ys, float), np.array(ye, float), np.array(se, float),
                   np.array(fs, bool), np.array(yc, float), np.array(fl, float), np.array(set_, float),
                   np.array(fr, float), np.array(er, bool), failures, modes.pop())


def concat(a: Dataset, b: Dataset) -> Dataset:
    """Rows of two datasets labeled with the same protocols, budgets and design plan (e.g. a development set and a
    historical final test used together for training)."""
    if a.protocols != b.protocols or a.budgets != b.budgets:
        raise ValueError("datasets with different protocols or budgets cannot be combined")
    if a.design_mode != b.design_mode:
        raise ValueError("datasets of different design plans (%s, %s) estimate different risks"
                         % (a.design_mode, b.design_mode))

    def cat(x, y):
        return None if x is None or y is None else np.concatenate([x, y])
    return Dataset(a.protocols, a.budgets, a.rows + b.rows, cat(a.Y_sel, b.Y_sel), cat(a.Y_eval, b.Y_eval),
                   cat(a.SE_eval, b.SE_eval), cat(a.F, b.F), cat(a.Yc_eval, b.Yc_eval), cat(a.fail, b.fail),
                   cat(a.SE_eval_trial, b.SE_eval_trial), cat(a.flag_rates, b.flag_rates), cat(a.ERR, b.ERR),
                   a.failures + b.failures, a.design_mode)
