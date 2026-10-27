r"""Monte-Carlo validation of the equal-budget ledger at selected budgets.

For a method in a cell and a total budget B, :func:`validate_method` simulates R complete experiments that spend
*exactly* B shots in the integer split of :func:`qemx.phase.ledger.allocate`:

* target and training shots: the method's own finite-shot sampler at N_t target shots (multinomial counts of every
  circuit, then the full estimator — ZNE refits its extrapolation on each replicate, CDR/FR/NIL refit their regression
  on sampled training features, PEC samples (sign, outcome) pairs from the exact quasi-probability distribution);
  ZNE's per-circuit shots are the exact integer split the ledger uses;
* readout calibration: the calibration circuits are sampled with s shots each (multinomial over the full output
  distribution) and the flip probabilities θ̂ are re-estimated. For the learning methods (``joint_calibration``)
  each experiment's training and target data are mitigated with its own θ̂ inside the sampler, exactly as the
  protocol does. For the other methods the estimate is recomputed with the mitigation built from θ̂ and added
  (``est = x_target + [f(θ̂) − f(θ)]``), which neglects only the product term of the two errors, O(1/(N_t s)).

Reported per (method, policy, B): the Monte-Carlo bias, variance and RMSE with standard errors, the ledger's values
at the same allocation, the z-scores of the differences, and the fraction of non-finite replicates (exponential ZNE
fails when a sampled E(λ) changes sign). The z-scores divide by the combined standard error of the Monte-Carlo
experiment *and* of the ledger value where the ledger itself was measured by Monte Carlo (the training levels of the
learning methods: their b, u and w carry the standard errors ``se_b``, ``se_u``, ``se_w``). A difference with a zero
standard error gives z = ±∞ (z = 0 only if the difference is exactly zero). ``pass`` means |z| ≤ 4 for the RMSE and
the bias and every replicate is finite. The pre-registered z-scores use plug-in standard errors (the sample variance's
from the fourth central moment); for heavy-tailed estimators these run low (by about 10–20 % for Student-t(5) or
log-normal samples of 500), so a bootstrap SE (``mc.se_rmse_boot``) and the z-score with it (``z_rmse_boot``) are
recorded beside them as a diagnostic. Known limits of the check, stated: the ledger's own SE ignores the correlation
of its b and u estimates (same refits); z-scores of one cell share its level estimates, so a KS test across checks
is approximate; the opt policy picks the best of several noisy levels, which biases its ledger MSE slightly low. For the learning methods the training-shot part of the ledger is measured
(not modeled), so their check tests the composition — target noise, training noise, calibration noise — and the
first-order target variance. A failure is a *finding* (the first-order model does not hold at that budget), not an exception.

Virtual distillation's sampler is Gaussian by construction (its variance comes from exact moments, not a shot
simulation), and PEC on the Aer engine (E3) has no exact sampler either; those checks are marked
``independent: false``.
"""
from __future__ import annotations

import math

import numpy as np


def _cal_draws(ctx, s: int, R: int, rng) -> np.ndarray | None:
    """R samples of θ̂ from calibration circuits run with s shots each (Context.sample_cal_theta)."""
    return ctx.sample_cal_theta(int(s), int(R), rng)


def _z(diff: float, se: float) -> float:
    """diff/se; a nonzero difference with a zero standard error is infinitely significant."""
    if se > 0:
        return float(diff / se)
    return 0.0 if diff == 0 else math.copysign(math.inf, diff)


BOOT = 200                       # bootstrap resamples of the diagnostic standard errors


def _stats(est: np.ndarray, ideal: float) -> dict:
    fin = np.isfinite(est)
    e = est[fin] - ideal
    R = int(fin.sum())
    if R < 2:
        return dict(reps=int(len(est)), finite=R, frac_nonfinite=float(1 - R / max(len(est), 1)))
    mse = float(np.mean(e ** 2))
    rm = math.sqrt(mse)
    se_mse = float(np.std(e ** 2, ddof=1) / math.sqrt(R))
    var = float(np.var(est[fin], ddof=1))
    m4 = float(np.mean((est[fin] - est[fin].mean()) ** 4))
    out = dict(reps=int(len(est)), finite=R, frac_nonfinite=float(1 - R / len(est)), bias=float(np.mean(e)),
               se_bias=float(np.std(e, ddof=1) / math.sqrt(R)), var=var,
               se_var=float(math.sqrt(max(m4 - var * var * (R - 3) / (R - 1), 0.0) / R)), rmse=rm,
               se_rmse=se_mse / (2 * rm) if rm > 0 else 0.0, median_abs_err=float(np.median(np.abs(e))))
    if R >= 50:
        # bootstrap standard errors (diagnostic): the plug-in fourth-moment SEs above are what the pre-registered
        # z-scores use; for heavy-tailed estimators (NIL, exponential ZNE at small N) they run low, and these show by
        # how much (fixed seed: the bootstrap is a property of the sample, not a new random experiment)
        idx = np.random.default_rng(0x5EB007).integers(0, R, size=(BOOT, R))
        eb = e[idx]
        out.update(se_rmse_boot=float(np.std(np.sqrt(np.mean(eb ** 2, 1)), ddof=1)),
                   se_var_boot=float(np.std(np.var(eb, 1, ddof=1), ddof=1)))
    return out


def validate_method(ctx, res, cost, B: int, policy, ideal: float, R: int, rng, rerun=None,
                    level_se: dict | None = None) -> dict:
    """One Monte-Carlo check at the ledger's integer allocation under ``policy`` (a ledger.Policy). ``rerun(s_tr)``
    re-runs a learning method with another number of training shots; ``ctx`` is a Context of the method's recipe (for
    the calibration circuits and the mitigator)."""
    from .ledger import allocate
    al = allocate(cost, int(B), policy)
    out = dict(method=cost.method, policy=policy.name, B=int(B), feasible=bool(al.feasible))
    if not al.feasible:
        return out
    out.update(N=int(al.N), s=int(al.s), s_tr=int(al.s_tr), N_j=list(al.N_j),
               identity=int(al.N + cost.c * al.s + cost.T * al.s_tr) == int(B))
    if res.sampler is None:
        out.update(skipped="no sampler")
        return out
    from ..resources import check
    r_used = res
    if cost.T > 0 and al.s_tr != cost.s_tr:
        if rerun is None:
            out.update(skipped="needs a re-run with %d training shots" % al.s_tr)
            return out
        r_used = rerun(int(al.s_tr))
    check()
    joint = bool(getattr(r_used, "joint_calibration", False)) and cost.c > 0 and al.s > 0
    th_hat = _cal_draws(ctx, al.s, int(R), rng) if cost.c > 0 and al.s > 0 else None
    if joint and th_hat is not None:
        # training and target data of experiment r mitigated with its own calibration draw θ̂_r (the protocol)
        x = np.asarray(r_used.sampler(int(al.N), int(R), rng, theta=th_hat), float)
        delta = np.zeros_like(x)
        out["calibration_mc"] = "joint (sampled, inside the sampler)"
    else:
        x = np.asarray(r_used.sampler(int(al.N), int(R), rng), float)
        delta = np.zeros_like(x)
        if th_hat is not None:
            fun = getattr(r_used, "cal_fun", None)
            if fun is not None:
                theta = np.asarray(r_used.cal_theta, float)
                f0 = fun(ctx._mitigator(theta))
                for i in range(int(R)):
                    if i % 64 == 0:
                        check()
                    t = th_hat[i]
                    if not ctx.theta_invertible(t):
                        delta[i] = np.nan               # a calibration this bad cannot be inverted
                        continue
                    delta[i] = fun(ctx._mitigator(t)) - f0
                out["calibration_mc"] = "sampled"
            else:
                delta = rng.normal(0.0, math.sqrt(cost.w / al.s), size=x.shape)
                out["calibration_mc"] = "gaussian (no recompute function)"
    est = x + delta
    st = _stats(est, ideal)
    var_an = float(al.var)
    rmse_an = math.sqrt(al.mse)
    out.update(mc=st, analytic=dict(b=al.b, var=var_an, rmse=rmse_an, var_target=al.var_target,
                                    var_cal=al.var_cal, var_train=al.var_train))
    # the ledger's own Monte-Carlo error (learning methods: b, u, v and w at the training level are MC estimates)
    se = (level_se or {}).get(int(al.s_tr), ()) if cost.T > 0 else ()
    se_b_an, se_u_an, se_v_an, se_w_an = (tuple(se) + (0.0, 0.0, 0.0, 0.0))[:4]
    se_mse_an = math.sqrt((2 * al.b * se_b_an) ** 2 + se_u_an ** 2 + (se_v_an / max(al.N, 1)) ** 2
                          + ((se_w_an / al.s) if cost.c > 0 and al.s > 0 else 0.0) ** 2)
    se_rmse_an = se_mse_an / (2 * rmse_an) if rmse_an > 0 else 0.0
    out["analytic"].update(se_b=se_b_an, se_u=se_u_an, se_v=se_v_an, se_w=se_w_an, se_rmse=se_rmse_an)
    if "rmse" in st:
        den_r = math.sqrt(st["se_rmse"] ** 2 + se_rmse_an ** 2)
        den_b = math.sqrt(st["se_bias"] ** 2 + se_b_an ** 2)
        z_rmse = _z(st["rmse"] - rmse_an, den_r)
        z_bias = _z(st["bias"] - al.b, den_b)
        if "se_rmse_boot" in st:
            out["z_rmse_boot"] = _z(st["rmse"] - rmse_an, math.sqrt(st["se_rmse_boot"] ** 2 + se_rmse_an ** 2))
        out.update(z_rmse=z_rmse, z_bias=z_bias, var_ratio=float(st["var"] / var_an) if var_an > 0
                   else None, u_excess=float(max(st["var"] - var_an, 0.0)),
                   passed=bool(abs(z_rmse) <= 4.0 and abs(z_bias) <= 4.0 and st["frac_nonfinite"] == 0.0))
    else:
        out.update(passed=False)
    return out


def stream_seed(spec, *key) -> np.random.SeedSequence:
    """A SeedSequence from the cell's seed, its id and a key (method, policy, budget, …): collision-free by
    construction (SHA-256 of the full key), independent of the order in which other streams are used."""
    import hashlib
    import json
    h = hashlib.sha256(json.dumps([int(spec.seed), spec.cell_id, list(key)], default=str).encode()).digest()
    return np.random.SeedSequence([int.from_bytes(h[i:i + 4], "little") for i in range(0, 16, 4)])


def validate_cell(spec, comp, fc, noise, backend, runner, recipes, rows, ctxs, ideal) -> list:
    """Every successful method × policy × budget of the cell's validation request."""
    from ..methods import Context, REGISTRY, Recipe
    from ..resources import RunTimeout
    from .cell import cost_of, runner_noise
    from .ledger import Policy
    noise_used = noise if backend == "numpy" else runner_noise(runner)
    out = []
    for row in rows:
        if row.get("status") != "ok" or row["key"] not in ctxs:
            continue
        rc, res = ctxs[row["key"]]
        cost = cost_of(row)
        if cost is None:
            continue
        ctx = Context(comp, noise_used, fc.observable, rc, runner, shots=4096, seed=int(spec.seed))

        cache = {}

        def rerun(s_tr, rc=rc, cache=cache):
            """The method re-run with s_tr training shots (once per s_tr: the result is deterministic)."""
            if int(s_tr) not in cache:
                p = dict(rc.p)
                p["train_shots"] = int(s_tr)
                rc2 = Recipe.make(rc.estimator, rc.twirl, rc.dd, rc.readout, **p)
                c2 = Context(comp, noise_used, fc.observable, rc2, runner, shots=4096, seed=int(spec.seed))
                cache[int(s_tr)] = REGISTRY[rc2.estimator].run(c2, **rc2.p)
            return cache[int(s_tr)]
        level_se = {int(l["s_tr"]): (float(l["se_b"]), float(l["se_u"]), float(l.get("se_v", 0.0)),
                                     float(l.get("se_w", 0.0)))
                    for l in (row.get("levels") or ()) if l.get("ok", True)}
        for pname in spec.validate_policy:
            if pname == "opt" and cost.T == 0:
                continue
            for B in spec.validate_budgets:
                # one independent stream per (cell, method, policy, budget), whatever else succeeds or fails
                rng = np.random.default_rng(stream_seed(spec, "validate", row["key"], pname, int(B)))
                try:
                    pol = Policy(pname, True, int(spec.s_min), int(spec.s_tr_min))
                    r = validate_method(ctx, res, cost, int(B), pol, ideal, int(spec.validate_reps), rng, rerun,
                                        level_se)
                except (RunTimeout, MemoryError, KeyboardInterrupt):
                    raise                                # the runner records the whole cell
                except Exception as e:                   # noqa: BLE001 - recorded, the cell goes on
                    r = dict(method=cost.method, policy=pname, B=int(B), error="%s: %s" % (type(e).__name__, e))
                flags = row.get("flags", [])
                r["independent"] = not ("gaussian-sampler" in flags or "mc-mean" in flags)
                out.append(r)
    return out
