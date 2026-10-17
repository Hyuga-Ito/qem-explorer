r"""Chained mitigation: an *outer* estimator that combines circuits, applied to an *inner* mitigation of every circuit
it runs.

.. math::
    \hat O=\mathcal C\bigl(\{\hat E_{\rm inner}(C_j)\}_j\bigr)

The outer estimator decides which circuits :math:`C_j` are run and how their values are combined; the inner method
turns each circuit into a mitigated value :math:`\hat E_{\rm inner}(C_j)` with its own per-shot variance, cost and
finite-shot sampler. Recipe transforms (twirling, dynamical decoupling) and the readout mitigation act on every
circuit, below both.

==========  ==========================================================================================================
outer       ``zne``  folded copies :math:`C_j=U(U^\dagger U)^{k_j}`, extrapolated to λ = 0 (Richardson, linear,
                     exponential); target shots split :math:`N_j\propto|\eta_j|\sigma_j` with :math:`\sigma_j` the
                     *inner* per-shot standard deviation.
            ``cdr``  Clifford data regression (Czarnik et al.): ideal ≈ a·x + b on near-Clifford training circuits,
                     x the inner-mitigated value of each training circuit (a fixed training ensemble; the adaptive
                     escalation of the ``learning`` estimator is not used).
            ``nil``  neighbor-informed learning (Chen et al., arXiv:2512.12578): features are inner-mitigated values
                     of the neighbor circuits; any of NIL's regression rules. With fold neighbors and OLS without an
                     intercept this is variable-noise CDR (Lowe et al., PRR 3, 033098 (2021)); with an inner VD it
                     combines ZNE, CDR and VD data in the spirit of UNITED (Bultrini et al., Quantum 7, 1034 (2023)).
inner       ``raw``       the circuit's noisy value (``chain(outer, raw)`` reproduces the outer method alone)
            ``symmetry``  post-selection on Z-parities (every circuit, training circuits included)
            ``vd``        virtual distillation with M copies (upper bound: ideal copies and derangement)
            ``pec``       probabilistic error cancellation of every circuit
==========  ==========================================================================================================

Accounting (equal-budget ledger). Every circuit the outer method runs costs what its inner method costs: per-shot
variances are the inner ones (symmetry verification counts rejected shots, PEC includes γ²), the training shots of
cdr/nil are T × k circuits × ``train_shots`` (each spent on the inner method), extra qubits are the inner method's.
The smallest usable target budget is :math:`\sum_j\lceil5/P_j\rceil` with post-selection (reported through an
effective acceptance 5/Σ⌈5/P_j⌉, so the cost model's ⌈5/P⌉ equals it). The training-shot variance u is measured by
Monte-Carlo refits on training data sampled with the inner samplers. The readout calibration enters every inner
value; its coefficient w is measured by Monte Carlo at exact (infinite-shot) features — calibration draws θ̂, the whole
estimate recomputed (refit included) — and validated the same way (``cal_fun``).

Exactness. Means are exact (exact probabilities through every inner method); ``chain(zne, raw)`` equals ``zne`` and
``chain(nil, raw)`` equals ``nil`` (mean, per-shot variance, training-shot variance, cost), which the tests check.
Samplers are exact where the inner sampler is (raw, symmetry, PEC with one shot per sampled circuit); VD's sampler is
Gaussian. Hardware: chained protocols run in simulation only in this version.
"""
from __future__ import annotations

import dataclasses
import json
import math

import numpy as np

from .base import FAMILY_LABEL, REGISTRY, Context, MethodResult, Param, QEMMethod, Recipe, register
from .raw import cal_fields, display_circuit
from ..budget import MIN_ACCEPTED, optimal_integer_split
from ..resources import check

CHAIN_OUTER = ("zne", "cdr", "nil")
CHAIN_INNER = ("raw", "symmetry", "vd", "pec")
OUTER_LABEL = {"zne": "ZNE", "cdr": "CDR", "nil": "NIL"}
FAMILY_LABEL.setdefault("chain", "Chained (outer ∘ inner)")
CHAIN_CAL_DRAWS = 200            # calibration draws θ̂ for the Monte-Carlo calibration coefficient w

# parameters copied from the component methods, prefixed (zne_…, nil_…, symmetry_…, vd_…, pec_…)


def _prefixed(method_key: str, prefix: str, gate: tuple, skip=()) -> tuple:
    """The parameters of a registered method as chain parameters: renamed prefix_name, shown only when ``gate``
    holds and the method's own visibility conditions (renamed) hold."""
    out = []
    for p in REGISTRY[method_key].params:
        if p.name in skip:
            continue
        when = gate + tuple((prefix + other, tuple(vals)) for other, vals in (p.when or ()))
        out.append(dataclasses.replace(p, name=prefix + p.name, when=when))
    return tuple(out)


def _strip(params: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


def _chain_params() -> tuple:
    learn = (("outer", ("cdr", "nil")),)
    return ((Param("outer", "outer estimator", "choice", "zne", CHAIN_OUTER,
                   help="zne: extrapolate folded copies; cdr: regress ideal on noisy training values; nil: learn a "
                        "combination of neighbor circuits. Every circuit it runs is mitigated by the inner method."),
             Param("inner", "inner mitigation (every circuit)", "choice", "raw", CHAIN_INNER,
                   help="raw: none (the outer method alone); symmetry: post-selection on Z-parities (choose a parity "
                        "the ideal state has); vd: virtual distillation (upper bound); pec: probabilistic error "
                        "cancellation."))
            + _prefixed("zne", "zne_", (("outer", ("zne",)),), skip=("fold_draws",))    # one fold per scale
            + _prefixed("nil", "nil_", (("outer", ("nil",)),), skip=("n_train", "train_shots"))
            + (Param("cdr_ensemble", "training angles", "choice", "near", ("near", "design4", "uniform"),
                     help="near: Clifford angles near the original ones, P ∝ exp(−Δθ²) (Czarnik et al.); design4: "
                          "{0, π/2, π, 3π/2} uniformly; uniform: any angle (not Clifford).",
                     when=(("outer", ("cdr",)),)),
               Param("n_train", "training circuits (T)", "int", 16, lo=2, hi=4096, step=1, when=learn),
               Param("train_shots", "shots per training circuit", "int", 1024, lo=16, hi=10 ** 8, step=64,
                     help="Spent on the inner method of every training circuit (T × k circuits).", when=learn))
            + _prefixed("symmetry", "symmetry_", (("inner", ("symmetry",)),))
            + _prefixed("vd", "vd_", (("inner", ("vd",)),))
            + _prefixed("pec", "pec_", (("inner", ("pec",)),)))


class _Inner:
    """The inner method on any circuit with the target's register, measurement map, noise and recipe transforms
    (memoized per circuit)."""

    def __init__(self, ctx, inner: str, params: dict):
        self.ctx, self.inner, self.params = ctx, inner, dict(params)
        self.recipe = Recipe.make(inner, ctx.recipe.twirl, ctx.recipe.dd, ctx.recipe.readout, **params)
        self.memo = {}

    def __call__(self, spec) -> MethodResult:
        key = spec.key()
        if key not in self.memo:
            check()
            ctx = self.ctx
            comp = dataclasses.replace(ctx.compiled, implemented=spec)
            sub = Context(comp, ctx.noise, ctx.obs, self.recipe, ctx.runner, shots=ctx.shots, symmetry=ctx.symmetry,
                          seed=ctx.seed)
            r = REGISTRY[self.inner].run(sub, **self.recipe.p)
            if not math.isfinite(float(r.mean)):
                raise ValueError("the inner method (%s) gave no finite value on %s" % (self.inner, spec.name))
            self.memo[key] = r
        return self.memo[key]


def _accept_eff(results) -> float:
    """Effective acceptance whose ⌈5/P⌉ is Σ_j ⌈5/P_j⌉ (every circuit needs its own accepted shots)."""
    acc = [float(r.accept) for r in results]
    if all(a >= 1 - 1e-12 for a in acc):
        return 1.0
    need = sum(int(math.ceil(min(MIN_ACCEPTED / max(a, 1e-300), 1e18) - 1e-9)) if a < 1 - 1e-12 else 1 for a in acc)
    return float(MIN_ACCEPTED) / need


@register
class Chain(QEMMethod):
    key = "chain"
    label = "Chained mitigation (outer ∘ inner)"
    family = "chain"
    tex = r"\hat O=\mathcal C\bigl(\{\hat E_{\rm inner}(C_j)\}_j\bigr)"
    why = (r"Mitigation methods act at different levels: post-selection, PEC and virtual distillation turn one "
           r"circuit's shots into a better value of that circuit; ZNE, CDR and NIL combine *several* circuits. "
           r"Chaining applies the first kind to every circuit the second kind runs, so their biases can cancel "
           r"further — at the product of their costs: every circuit pays the inner method's overhead.")
    params = _chain_params()

    # ------------------------------------------------------------------ titles, hardware
    def short(self, p):
        outer, inner = p.get("outer", "zne"), p.get("inner", "symmetry")
        if outer == "zne":
            o = REGISTRY["zne"].short(dict(REGISTRY["zne"].defaults(), **_strip(p, "zne_")))
        elif outer == "nil":
            q = dict(REGISTRY["nil"].defaults(), **_strip(p, "nil_"))
            q.update(n_train=int(p.get("n_train", 16)))
            o = REGISTRY["nil"].short(q)
        else:
            o = "CDR (T=%d, %s)" % (int(p.get("n_train", 16)), p.get("cdr_ensemble", "near"))
        ip = dict(REGISTRY[inner].defaults(), **_strip(p, inner + "_"))
        i = REGISTRY[inner].short(ip) if hasattr(REGISTRY[inner], "short") else REGISTRY[inner].label
        return "%s ∘ %s" % (o, i)

    def min_cost(self, ctx, **p) -> dict | None:
        """The outer learning step's training shots (T × k × s_tr on the inner method); ZNE outer: none."""
        from ..transforms import non_clifford_slots
        p = dict(self.defaults(), **p)
        if p.get("outer") == "zne" or not non_clifford_slots(ctx.spec):
            return dict(training=0, target_circuits=1, note="no training circuits")
        if p["outer"] == "nil":
            q = dict(REGISTRY["nil"].defaults(), **_strip(p, "nil_"))
            try:
                k = len(REGISTRY["nil"].neighbors(ctx, q["neighbors"], q["positions"], q["n_neighbors"],
                                                  q["include_bare"], q["scales"], q.get("folding", "global")))
            except Exception:                      # noqa: BLE001 - reported by run()
                return None
        else:
            k = 1
        T, s_tr = int(p["n_train"]), int(p["train_shots"])
        return dict(training=T * k * s_tr, target_circuits=1,
                    note="%d training circuits × %d feature circuits × %d shots" % (T, k, s_tr))

    def hardware_supported(self, params) -> tuple:
        return False, ("chained protocols run in simulation only in this version: the inner method's per-circuit "
                       "processing (post-selection, PEC sampling, VD copies) and the outer combination are not planned "
                       "as one hardware batch")

    # ------------------------------------------------------------------ run
    def run(self, ctx, **p) -> MethodResult:
        p = dict(self.defaults(), **p)
        outer, inner = p["outer"], p["inner"]
        if outer not in CHAIN_OUTER or inner not in CHAIN_INNER:
            raise ValueError("chain: outer must be one of %s and inner one of %s" % (CHAIN_OUTER, CHAIN_INNER))
        if getattr(ctx.runner, "backend", None) == "hardware":
            from ..backends.base import NotSupported
            raise NotSupported(self.hardware_supported(p)[1])
        ev = _Inner(ctx, inner, _strip(p, inner + "_"))
        if outer == "zne":
            res = self._zne(ctx, ev, **_strip(p, "zne_"))
        else:
            res = self._learn(ctx, ev, outer, p)
        res.details.update(outer=outer, inner=inner, inner_recipe=ev.recipe.to_json())
        return res

    # ------------------------------------------------------------------ outer: ZNE
    def _zne(self, ctx, ev, scales=(1.0, 3.0, 5.0), folding="global", rule="richardson",
             lam_est="gates", subset="random") -> MethodResult:
        from .zne import (LOCAL_SUBSETS, RULES, extrapolate, fault_rate_lambdas, largest_remainder_split,
                          local_fold_weights, weights)
        from ..transforms import achieved_lambda, fold_global, fold_local
        scales = sorted(float(s) for s in scales)
        if len(scales) < 2 or min(scales) < 1:
            raise ValueError("ZNE needs at least two noise scales, all at least 1")
        if not any(o.kind == "gate" for o in ctx.spec.ops):
            raise ValueError("the circuit has no gates, so there is nothing to fold")
        if subset not in LOCAL_SUBSETS:
            raise ValueError("subset must be one of %s" % ", ".join(LOCAL_SUBSETS))
        fw = local_fold_weights(ctx, ctx.spec) if (folding == "local" and subset == "balanced") else None
        specs = [fold_global(ctx.spec, lam) if folding == "global" else fold_local(ctx.spec, lam, ctx.seed, subset, fw)
                 for lam in scales]
        lams = [achieved_lambda(ctx.spec, fs) for fs in specs]
        if len(set(np.round(lams, 9))) < len(lams):
            raise ValueError("the scales %s fold to the same λ on this circuit; choose scales that give distinct λ"
                             % ", ".join("%g" % s for s in scales))
        if lam_est not in ("gates", "fault-rate"):
            raise ValueError("lam_est must be 'gates' or 'fault-rate'")
        lams_gates = list(lams)
        lams_fault, fault_note = fault_rate_lambdas(ctx, specs)     # the light-cone fault rate of the folded circuits
        if lam_est == "fault-rate":                                 # (the inner method does not change λ_LC)
            if lams_fault is None:
                raise ValueError("λ from the fault rate is not available: %s" % fault_note)
            if np.any(np.diff(lams_fault) <= 1e-9):
                raise ValueError("the folded circuits do not raise the fault rate monotonically (λ_eff = %s)"
                                 % ", ".join("%.4g" % l for l in lams_fault))
            lams = list(lams_fault)
        rs = [ev(fs) for fs in specs]
        E = [float(r.mean) for r in rs]
        sig = [math.sqrt(max(float(r.var1), 0.0)) for r in rs]
        est = extrapolate(rule, lams, E)
        eta = weights(rule, lams, E) if np.isfinite(est) else np.full(len(lams), np.nan)
        w = np.abs(eta) * np.asarray(sig)
        var1 = float(np.sum(w)) ** 2
        alloc = w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else np.full(len(w), 1 / len(w))
        notes = []
        if not np.isfinite(est):
            notes.append("The %s fit gave no finite value at λ = 0." % RULES[rule])
        if lams_fault is not None:
            notes.append("Effective noise scale (light-cone fault rate ÷ original): λ_eff = %s; %s." % (
                ", ".join("%.3f" % l for l in lams_fault),
                "used for the fit" if lam_est == "fault-rate" else "the fit uses the nominal λ = %s" % ", ".join(
                    "%.3g" % l for l in lams_gates)))

        def sampler(N, B, rng):
            N = int(N)
            if N < len(rs):
                return np.full(int(B), np.nan)
            Nj = optimal_integer_split(w, N)[1] if np.isfinite(w).all() and w.sum() > 0 else \
                largest_remainder_split(alloc, N)
            Ej = np.stack([np.asarray(r.sampler(int(nj), int(B), rng), float) for r, nj in zip(rs, Nj)], 1)
            return np.array([extrapolate(rule, lams, e) for e in Ej])
        cal = {}
        if all(r.cal_fun is not None for r in rs) and ctx.cal_circuits():
            cal = cal_fields(ctx, lambda mit: float(extrapolate(rule, lams, [r.cal_fun(mit) for r in rs])))
        notes += _inner_notes(rs)
        return MethodResult(
            mean=float(est), var1=var1, circuits=int(sum(int(r.circuits or 1) for r in rs)), **cal,
            shots_per_circuit=rs[0].shots_per_circuit, gamma=float(np.nansum(np.abs(eta) * [r.gamma for r in rs])),
            extra_qubits=max(int(r.extra_qubits or 0) for r in rs), accept=_accept_eff(rs),
            implemented=_implemented(rs, ["λ = %.3g" % l for l in lams]),
            details=dict(lams=lams, E=E, sigma=sig, eta=eta, alloc=alloc, rule=rule, folding=folding,
                         lam_est=lam_est, lams_gates=lams_gates, lams_fault=lams_fault,
                         subset=subset if folding == "local" else None,
                         depths=[int(sp.depth()) for sp in specs], twoq=[int(sp.count("2q")) for sp in specs],
                         accept=[float(r.accept) for r in rs], inner_gamma=[float(r.gamma) for r in rs]),
            notes=notes, sampler=sampler)

    # ------------------------------------------------------------------ outer: CDR / NIL
    def _learn(self, ctx, ev, outer: str, p: dict) -> MethodResult:
        from .learning import CAL_REF_SHOTS, MC_REPS
        from .nil import (NIL_MODELS, _nil_seed, _nil_twins, nil_fit, nil_training_circuit)
        from ..transforms import non_clifford_slots
        T, s_tr = int(p["n_train"]), int(p["train_shots"])
        ensemble = p["nil_ensemble"] if outer == "nil" else p["cdr_ensemble"]
        if T < 2 or s_tr < 1:
            raise ValueError("learning needs at least 2 training circuits and 1 shot per training circuit")
        if outer == "nil":
            q = dict(REGISTRY["nil"].defaults(), **_strip(p, "nil_"))
            if q["model"] not in NIL_MODELS:
                raise ValueError("unknown regression model %r" % q["model"])
            feats = REGISTRY["nil"].neighbors(ctx, q["neighbors"], q["positions"], q["n_neighbors"],
                                              q["include_bare"], q["scales"], q.get("folding", "global"))
            model, target_split = q["model"], q["target_split"]
            fit_kw = dict(alpha=float(q["alpha"]), l1_ratio=float(q["l1_ratio"]), l1_bound=float(q["l1_bound"]),
                          intercept=bool(q["intercept"]))
        else:
            feats = [("bare", (lambda s: s), dict(kind="bare"))]
            model, target_split = "ols", "optimal"
            fit_kw = dict(alpha=0.0, l1_ratio=0.5, l1_bound=1.0, intercept=True)        # ideal ≈ a·x + b
        k = len(feats)
        names = [f[0] for f in feats]
        crn_key = "chain:" + json.dumps(dict(outer=outer, inner=ev.recipe.to_json(), names=names, n_train=T,
                                             ensemble=ensemble), sort_keys=True, default=str)

        def fallback(note, extra=None):
            r = ev(ctx.spec)
            d = dict(r.details or {})
            d.update(extra or {})
            return dataclasses.replace(r, details=d, notes=list(r.notes or []) + [note])
        if not non_clifford_slots(ctx.spec):
            return fallback("The circuit is Clifford: every training circuit would equal the target. The outer "
                            "learning step is not applied; the inner method's value is reported.")
        rng_tr = _nil_seed(ctx.seed, 2)
        tr_specs = [nil_training_circuit(ctx.spec, rng_tr, ensemble) for _ in range(T)]
        y = np.array([ctx.ideal_value(sp) for sp in tr_specs], float)
        base = dict(names=names, n_features=k, n_train=T, model=model, ensemble=ensemble, target_split=target_split,
                    y_train=y, crn_key=crn_key)
        if np.ptp(y) < 1e-9:
            return fallback("All %d training circuits have the same ideal value %.4g: the outer learning step cannot "
                            "be trained and is not applied; the inner method's value is reported." % (T, y[0]),
                            dict(base, degenerate=True))
        R_tr = [[ev(f[1](sp)) for f in feats] for sp in tr_specs]
        R_tg = [ev(f[1](ctx.spec)) for f in feats]
        X = np.array([[float(r.mean) for r in row] for row in R_tr])
        xt = np.array([float(r.mean) for r in R_tg])
        coef, b0, info = nil_fit(X, y, model, **fit_kw)
        mean = float(b0 + coef @ xt)
        sig = np.sqrt(np.maximum([float(r.var1) for r in R_tg], 0.0))

        def target_var(c):
            if target_split == "equal":
                return float(k * np.sum((c * sig) ** 2))
            return float(np.sum(np.abs(c) * sig)) ** 2
        var1 = target_var(coef)
        if target_split == "equal":
            circuits = k
        else:
            circuits = max(int(np.sum(coef != 0)), 1)

        def split(c, N):
            if target_split == "equal":
                if N < k:
                    return None
                n = np.full(k, N // k, dtype=np.int64)
                n[: N - int(n.sum())] += 1
                return n
            act = np.flatnonzero(c != 0)
            n = np.zeros(k, dtype=np.int64)
            if not len(act):
                return n
            if N < len(act):
                return None
            n[act] = optimal_integer_split(np.abs(c[act]) * sig[act], int(N))[1]
            return n

        def sample_training(shots, R, rng):
            out = np.empty((int(R), T, k))
            for t in range(T):
                for j in range(k):
                    out[:, t, j] = np.asarray(R_tr[t][j].sampler(int(shots), int(R), rng), float)
            return out

        def refit(Xb):
            if not np.all(np.isfinite(Xb)):
                return None, math.nan
            c, b, _ = nil_fit(Xb, y, model, warm=(coef, info), **fit_kw)
            return c, b

        def sampler(N, B, rng):
            Xs = sample_training(s_tr, B, rng)
            est = np.full(int(B), np.nan)
            for bi in range(int(B)):
                if bi % 16 == 0:
                    check()
                c, b = refit(Xs[bi])
                if c is None:
                    continue
                n = split(c, int(N))
                if n is None:
                    continue
                val = b
                for j in np.flatnonzero(n > 0):
                    val += c[j] * float(np.asarray(R_tg[j].sampler(int(n[j]), 1, rng), float)[0])
                est[bi] = val
            return est

        def train_resampler(shots, R, rng, theta=None):
            if theta is not None:
                raise ValueError("chain: per-replicate calibration draws are not supported (joint_calibration=False)")
            Xs = sample_training(shots, R, rng)
            est, v1 = np.full(int(R), np.nan), np.full(int(R), np.nan)
            for bi in range(int(R)):
                if bi % 16 == 0:
                    check()
                c, b = refit(Xs[bi])
                if c is not None:
                    est[bi], v1[bi] = b + xt @ c, target_var(c)
            return est, v1

        from ..stats import ProtocolFailure, protocol_contract, training_moments
        seed64 = int(ctx.seed) % (2 ** 63)
        notes = []
        try:
            # protocol moments at s_tr (qemx/stats.py); refits whose training data has no estimate (no accepted shot)
            # count as failures and are reported, never silently dropped
            mom = training_moments(train_resampler, s_tr, np.random.default_rng([seed64, 0x4E49, 1]))
        except ProtocolFailure as e:
            mom = None
            notes.append("The finite-training protocol fails too often to report its moments (%s); the values shown "
                         "are the reference fit to exact training data, with an infinite training variance." % e)
        if mom is not None:
            u_train = mom.u
            contract = protocol_contract(ctx, mom, mean, var1, notes)
            mean, var1 = contract.pop("mean"), contract.pop("var1")
        else:
            u_train = math.inf
            contract = dict(estimate_kind="reference", reference_mean=float(mean), reference_var1=float(var1),
                            moments=None, flags=("protocol-failed",))
        # ---- readout calibration: through every inner value and the refit
        cal = {}
        if all(r.cal_fun is not None for row in R_tr for r in row) and all(r.cal_fun is not None for r in R_tg) \
                and ctx.cal_circuits():
            def fun(mit):
                Xm = np.array([[r.cal_fun(mit) for r in row] for row in R_tr])
                xm = np.array([r.cal_fun(mit) for r in R_tg])
                cm, bm, _ = nil_fit(Xm, y, model, warm=(coef, info), **fit_kw)
                return float(bm + xm @ cm)
            cal = cal_fields(ctx, fun)
            th = ctx.sample_cal_theta(int(CAL_REF_SHOTS), CHAIN_CAL_DRAWS,
                                      np.random.default_rng([seed64, 0x4341, 7]))
            f0 = fun(ctx._mitigator(ctx.cal_theta()))
            d = np.array([fun(ctx._mitigator(t)) - f0 for t in th if ctx.theta_invertible(t)])
            d = d[np.isfinite(d)]
            cal["cal_w"] = float(CAL_REF_SHOTS * np.var(d, ddof=1)) if len(d) > 1 else 0.0
        twins = _nil_twins(X)
        if twins:
            notes.append("%d group%s of features give identical values on every training circuit; they share their "
                         "weight equally." % (len(twins), "" if len(twins) == 1 else "s"))
        notes.append("Training: %d circuits × %d feature circuit%s × %d shots on the inner method = %s shots; "
                     "training-shot variance u = %.3g (%d Monte-Carlo refits)." % (
                         T, k, "" if k == 1 else "s", s_tr, f"{T * k * s_tr:,}", u_train,
                         mom.repeats if mom is not None else MC_REPS))
        notes += _inner_notes([r for row in R_tr for r in row] + R_tg)
        allr = [r for row in R_tr for r in row] + R_tg
        a = np.abs(coef) * sig
        alloc = (np.full(k, 1.0 / k) if target_split == "equal" else
                 (a / a.sum() if a.sum() > 0 else np.where(coef != 0, 1.0, 0.0) / max(int(np.sum(coef != 0)), 1)))
        if cal:
            contract["flags"] = tuple(contract["flags"]) + ("calibration-mean-first-order",)
        return MethodResult(
            **contract,
            mean=mean, var1=var1, circuits=circuits, fixed_shots=T * k * s_tr, train_circuits=T * k, **cal,
            var_fixed=u_train, gamma=float(np.sum(np.abs(coef) * [r.gamma for r in R_tg])),
            extra_qubits=max(int(r.extra_qubits or 0) for r in allr), accept=_accept_eff(R_tg),
            shots_per_circuit=R_tg[0].shots_per_circuit,
            implemented=_implemented(R_tg[:1] + R_tr[0][:1], ["target", "training circuit 1"]),
            details=dict(base, coef=coef, intercept=b0, X=X, xt=xt, sigma=sig, alloc=alloc,
                         fitted=X @ coef + b0, nonzero=int(np.sum(coef != 0)), l1=float(np.abs(coef).sum()),
                         lam=float(info["lam"]), eps=float(info["eps"]), solver_ok=bool(info["ok"]),
                         twins=[[names[j] for j in g] for g in twins], degenerate=False,
                         accept=[float(r.accept) for r in R_tg]),
            notes=notes, sampler=sampler, train_resampler=train_resampler)


def _implemented(results, titles) -> list:
    out = []
    for r, t in zip(results, titles):
        for title, sp in (r.implemented or [])[:1]:
            out.append(("%s · %s" % (t, title), sp))
    return out


def _inner_notes(results) -> list:
    seen, out = set(), []
    for r in results:
        for n in r.notes or []:
            if n not in seen:
                seen.add(n)
                out.append("Inner method: " + n)
    return out[:4]
