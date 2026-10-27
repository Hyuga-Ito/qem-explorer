r"""Diagnostic-conditioned QEM decisions: does a cheap model-discrepancy check, paid for from the same budget,
improve the choice of a mitigation protocol or justify a safer fallback? (supporting study, 1.5.1)

Setting. A model of the device (a noise model with supplied rates) predicts every candidate protocol's RMSE; a
model-based policy picks the protocol with the smallest predicted RMSE. The device may differ from the model. Before
committing the main budget, a **diagnostic** spends D of the B shots on the target circuit without mitigation and
compares the observed distribution with the model's prediction (total-variation distance against its matched-shot
null, level α). The **diagnostic-conditioned policy** then picks, at the remaining B − D shots, the model's choice if
the check passes and a declared fallback protocol if it flags.

Losses are evaluated under the *truth*. Because the flag is random, the conditioned policy's loss on one case is an
expectation over the flag: P(flag)·L(fallback, B − D) + (1 − P(flag))·L(model choice, B − D). P(flag) is computed by
Monte Carlo over the diagnostic's counts under the truth, with the critical value taken from the matched-shot null
under the model (its Monte-Carlo errors are recorded).

Reported, per scenario group, with denominators: mean regret (RMSE / RMSE of the best protocol under the truth at
the full budget B − 1), the probability of a harmful choice (worse than unmitigated by more than δ), coverage (the
probability that the model's choice is used), detection power (mean P(flag) under a mismatch) and the false-positive
rate (under the null). The diagnostic's own cost is inside the comparison: the conditioned policy always has only
B − D shots for its estimate. A diagnostic that never changes the decision can only lose.

Scope. This tests one diagnostic statistic on one observable family; it does not identify the physical cause of a
mismatch, and an undetected mismatch is not evidence that the model is adequate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DELTA = 0.10


def tvd_rows(F, p) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(F, float) - np.asarray(p, float)[None, :]).sum(1)


def flag_probability(p_model, p_truth, shots: int, alpha: float = 0.01, null_reps: int = 20000,
                     alt_reps: int = 4000, rng=None) -> dict:
    """P(the TVD check flags) when the truth is ``p_truth`` and the model predicts ``p_model``, at ``shots`` shots:
    critical value = the (1 − α) quantile of the TVD between the model and ``shots`` samples from the model; power =
    the share of ``alt_reps`` samples from the truth whose TVD exceeds it. With p_truth = p_model this is the
    realized false-positive rate (≈ α up to the discreteness of the statistic)."""
    p = np.asarray(p_model, float)
    p = np.clip(p, 0, None) / np.clip(p, 0, None).sum()
    q = np.asarray(p_truth, float)
    q = np.clip(q, 0, None) / np.clip(q, 0, None).sum()
    rng = np.random.default_rng(rng)
    null = tvd_rows(rng.multinomial(int(shots), p, size=int(null_reps)) / float(shots), p)
    crit = float(np.quantile(null, 1 - alpha, method="higher"))
    size = float(np.mean(null > crit))
    alt = tvd_rows(rng.multinomial(int(shots), q, size=int(alt_reps)) / float(shots), p)
    pf = float(np.mean(alt > crit))
    return dict(p_flag=pf, p_flag_mcse=math.sqrt(max(pf * (1 - pf), 1e-12) / alt_reps), critical_tvd=crit,
                realized_size=size, shots=int(shots), alpha=float(alpha), true_tvd=float(0.5 * np.abs(p - q).sum()))


@dataclass
class Case:
    """One (circuit, truth) pair. ``true[b]`` and ``model[b]``: {protocol: RMSE} at total budget b under the truth
    and as predicted by the model (inf: cannot run). ``p_flag[D]``: the diagnostic's flag probability at D shots."""
    circuit: str
    group: str
    family: str
    scenario: str
    mismatch: bool
    split: str
    true: dict
    model: dict
    p_flag: dict = field(default_factory=dict)


def _argmin(d: dict, cands) -> str:
    vals = [(d.get(m, math.inf), i, m) for i, m in enumerate(cands)]
    return min(vals)[2]


def _harm(true_b: dict, m: str, raw: str, delta: float) -> float:
    r = true_b.get(raw, math.inf)
    return float(true_b.get(m, math.inf) > r * (1 + delta)) if math.isfinite(r) else math.nan


def case_outcomes(c: Case, B: int, D: int, candidates, fallback: str, raw: str = "RAW",
                  delta: float = DELTA) -> dict:
    """Expected regret and harm of the policies on one case (see the module text)."""
    tB, tD = c.true[B], c.true[B - D]
    best = min(tB.get(m, math.inf) for m in candidates)
    reg = lambda t, m: t.get(m, math.inf) / best - 1 if best > 0 else math.nan         # noqa: E731
    m0 = _argmin(c.model[B], candidates)
    mD = _argmin(c.model[B - D], candidates)
    pf = float(c.p_flag.get(D, math.nan))
    out = dict(circuit=c.circuit, group=c.group, family=c.family, scenario=c.scenario, mismatch=c.mismatch,
               split=c.split, B=B, D=D, model_choice=m0, model_choice_after_diag=mD, p_flag=pf,
               oracle=_argmin(tB, candidates),
               regret_model=reg(tB, m0), harm_model=_harm(tB, m0, raw, delta),
               regret_fallback=reg(tB, fallback), harm_fallback=_harm(tB, fallback, raw, delta),
               regret_raw=reg(tB, raw))
    out["regret_diag"] = pf * reg(tD, fallback) + (1 - pf) * reg(tD, mD)
    out["harm_diag"] = pf * _harm(tD, fallback, raw, delta) + (1 - pf) * _harm(tD, mD, raw, delta)
    out["coverage_diag"] = 1 - pf
    return out


def summarize(rows: list, by=("split", "mismatch")) -> list:
    """Means per group with counts; differences diagnostic − model with a bootstrap over base circuits."""
    import pandas as pd
    df = pd.DataFrame(rows)
    out = []
    for key, g in df.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        d = (g["regret_diag"] - g["regret_model"]).to_numpy(float)
        lo, hi = _group_ci(g["group"].to_numpy(), d)
        out.append(dict(zip(by, key), cases=int(len(g)), circuits=int(g["group"].nunique()),
                        scenarios=",".join(sorted(set(g["scenario"]))),
                        p_flag=float(g["p_flag"].mean()), coverage_diag=float(g["coverage_diag"].mean()),
                        regret_model=float(g["regret_model"].mean()), regret_diag=float(g["regret_diag"].mean()),
                        regret_fallback=float(g["regret_fallback"].mean()),
                        harm_model=float(g["harm_model"].mean()), harm_diag=float(g["harm_diag"].mean()),
                        harm_fallback=float(g["harm_fallback"].mean()),
                        diff_regret=float(np.mean(d)), diff_lo=lo, diff_hi=hi))
    return out


def _group_ci(groups, values, reps: int = 4000, seed: int = 0) -> tuple:
    by = {}
    for gk, v in zip(groups, values):
        if math.isfinite(v):
            by.setdefault(gk, []).append(v)
    arrs = [np.asarray(v, float) for v in by.values()]
    if len(arrs) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    bs = [float(np.mean(np.concatenate([arrs[k] for k in rng.integers(0, len(arrs), len(arrs))])))
          for _ in range(int(reps))]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def choose_D(rows_explore: list, Ds) -> int:
    """The diagnostic budget with the smallest mean expected regret on the exploration cases (null and mismatch
    cases weighted equally) — frozen before the held-out cases are looked at."""
    best = None
    for D in Ds:
        r = [x for x in rows_explore if x["D"] == D]
        if not r:
            continue
        m = np.mean([np.mean([x["regret_diag"] for x in r if x["mismatch"] == mm] or [0.0]) for mm in (False, True)])
        if best is None or m < best[0]:
            best = (m, D)
    return best[1] if best else None
