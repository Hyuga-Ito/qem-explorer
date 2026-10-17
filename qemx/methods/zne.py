r"""Zero-noise extrapolation (Temme, Bravyi & Gambetta, PRL 119, 180509 (2017); Li & Benjamin,
PRX 7, 021050 (2017); Endo, Benjamin & Li, PRX 8, 031027 (2018); Giurgica-Tiron et al., IEEE QCE 2020).

Run the circuit at noise scales :math:`\lambda_j\ge1` (by folding :math:`U\to U(U^\dagger U)^k`),
fit :math:`E(\lambda)` and evaluate the fit at :math:`\lambda=0`:

* Richardson   :math:`\hat E(0)=\sum_j\eta_jE(\lambda_j)`, :math:`\eta_j=\prod_{m\ne j}\frac{\lambda_m}{\lambda_m-\lambda_j}`
* linear       least-squares line
* poly2        least-squares parabola (for four or more closely spaced, fractional scales)
* exponential  least-squares fit of :math:`\log|E|` against λ (all points of one sign), i.e. :math:`E=Ae^{-b\lambda}`

The estimator is (or is linearized as) :math:`\sum_j\eta_jE_j`; with the optimal shot split
:math:`N_j\propto|\eta_j|\sigma_j` its variance is :math:`(\sum_j|\eta_j|\sigma_j)^2/N`.

Folding and the noise scale λ
-----------------------------
*Global* folding repeats the whole circuit, U(U†U)^k (a tail of it once more for a fractional λ); *local* folding
repeats each two-qubit gate, G(G†G)^k, and for a fractional λ a subset of the gates once more (``subset``: a random
subset, or a nested order — random, stratified over the circuit's duration, or error-balanced with the calibrated
gate errors; :func:`qemx.transforms.fold_order`; ``fold_draws`` K > 1 averages K independent random subsets per
fractional λ at the same total shots — the choice of subset moves E(λ) by more than shot noise, and a high-weight
fit such as poly2 amplifies that). Fractional scales between 1 and 3 (*fractional ZNE*, e.g.
λ = 1, 1.33, 1.67, 2) sample E(λ) before deep circuits decay to noise, at the price of a longer extrapolation lever
arm: their weights must come from a low-degree least-squares fit (linear, poly2), not from Richardson. Both reach the same two-qubit gate count, λ·n₂, which is the
nominal scale (``lam_est="gates"``). They do not amplify the *noise* by the same factor: global folding also repeats
every one-qubit gate and stretches every idle window, local folding leaves the one-qubit gates unfolded and adds idle
time only to the qubits waiting while a folded gate runs. ``lam_est="fault-rate"`` uses instead the factor by which
folding raises the light-cone fault rate of the noise model, λ_eff = λ_LC(folded)/λ_LC(original)
(:mod:`qemx.phase.lightcone`): the λ the observable actually sees. The result records both (``lams_gates``,
``lams_fault``) and the depth and two-qubit count of every folded circuit.
"""
from __future__ import annotations

import numpy as np

from ..budget import optimal_integer_split
from .base import MethodResult, Param, QEMMethod, register
from .raw import cal_fields, display_circuit
from ..transforms import achieved_lambda, fold_global, fold_local

RULES = {"richardson": "Richardson", "linear": "linear", "poly2": "quadratic", "exponential": "exponential"}
LOCAL_SUBSETS = ("random", "nested", "stratified", "balanced")
DRAW_SUBSETS = ("random", "nested")       # the subsets that can be redrawn (fold_draws > 1)


def is_fractional(lam: float) -> bool:
    """Whether λ folds a *part* of the gates once more (λ ≠ 1 + 2k)."""
    x = (float(lam) - 1) / 2
    return abs(x - round(x)) > 1e-9


def draw_seed(seed: int, k: int) -> int:
    """Seed of the k-th independent fold subset (k = 0 is the method's own seed: fold_draws = 1 is unchanged)."""
    return int(seed) + 7919 * int(k)


def richardson_weights(lams) -> np.ndarray:
    lams = np.asarray(lams, float)
    w = np.ones(len(lams))
    for j in range(len(lams)):
        for m in range(len(lams)):
            if m != j:
                w[j] *= lams[m] / (lams[m] - lams[j])
    return w


def linear_weights(lams) -> np.ndarray:
    V = np.vander(np.asarray(lams, float), 2, increasing=True)
    return np.linalg.pinv(V)[0]


def poly_weights(lams, degree: int) -> np.ndarray:
    """Weights of the value at λ = 0 of the least-squares polynomial of the given degree (first row of the
    Vandermonde pseudo-inverse). Degree m−1 through m points is Richardson; a lower degree through many closely
    spaced fractional scales keeps the weights, hence the variance, small."""
    lams = np.asarray(lams, float)
    if len(lams) < degree + 1:
        raise ValueError("a degree-%d fit needs at least %d distinct scales" % (degree, degree + 1))
    return np.linalg.pinv(np.vander(lams, degree + 1, increasing=True))[0]


def _exp_fit(lams, E):
    E = np.asarray(E, float)
    s = np.sign(E[0]) if E[0] != 0 else 1.0
    if np.any(s * E <= 0):
        return None
    c = np.polyfit(np.asarray(lams, float), np.log(s * E), 1)          # log|E| = c1 λ + c0
    return s, c


def extrapolate(rule: str, lams, E) -> float:
    if rule == "richardson":
        return float(richardson_weights(lams) @ np.asarray(E, float))
    if rule == "linear":
        return float(linear_weights(lams) @ np.asarray(E, float))
    if rule == "poly2":
        return float(poly_weights(lams, 2) @ np.asarray(E, float))
    if rule == "exponential":
        f = _exp_fit(lams, E)
        return float("nan") if f is None else float(f[0] * np.exp(f[1][1]))
    raise ValueError(rule)


def weights(rule: str, lams, E) -> np.ndarray:
    if rule == "richardson":
        return richardson_weights(lams)
    if rule == "linear":
        return linear_weights(lams)
    if rule == "poly2":
        return poly_weights(lams, 2)
    E = np.asarray(E, float)
    base = extrapolate(rule, lams, E)
    g = np.zeros(len(E))
    for j in range(len(E)):
        h = 1e-6 * max(abs(E[j]), 1e-3)
        Ep = E.copy()
        Ep[j] += h
        g[j] = (extrapolate(rule, lams, Ep) - base) / h
    return g


def largest_remainder_split(shares, N: int) -> tuple:
    """Integers N_j ≥ 1 with Σ N_j = N (N ≥ len(shares)) closest to N·shares: one shot each, then the remaining
    N − k distributed by largest remainders (Hamilton's method). Non-finite or all-zero shares split evenly."""
    a = np.asarray(shares, float)
    k = len(a)
    N = int(N)
    if k == 0 or N < k:
        raise ValueError("need at least one shot per circuit (N = %d < %d circuits)" % (N, k))
    if not np.isfinite(a).all() or np.any(a < 0) or a.sum() <= 0:
        a = np.ones(k)
    q = (N - k) * a / a.sum()
    n = np.floor(q).astype(np.int64)
    rest = (N - k) - int(n.sum())
    if rest > 0:
        n[np.argsort(-(q - n), kind="stable")[:rest]] += 1
    return tuple(int(x) + 1 for x in n)


def curve(rule: str, lams, E, grid) -> np.ndarray:
    lams, E, grid = np.asarray(lams, float), np.asarray(E, float), np.asarray(grid, float)
    if rule == "richardson":
        return np.polyval(np.polyfit(lams, E, len(lams) - 1), grid)
    if rule == "linear":
        return np.polyval(np.polyfit(lams, E, 1), grid)
    if rule == "poly2":
        return np.polyval(np.polyfit(lams, E, 2), grid)
    f = _exp_fit(lams, E)
    if f is None:
        return np.full_like(grid, np.nan)
    return f[0] * np.exp(np.polyval(f[1], grid))


def fault_rate_lambdas(ctx, specs) -> tuple:
    """(λ_eff of every folded circuit = λ_LC(folded) / λ_LC(original), note); (None, reason) when the context's noise
    model has no fault rate (a hardware run without a calibration model, or a noiseless model)."""
    from ..phase.lightcone import calculate_lambda_lc
    model = getattr(ctx.runner, "device_model", None)
    if model is None:
        model = ctx.noise
    if model is None:
        return None, "the context has no noise model"
    try:
        lam = [calculate_lambda_lc(sp, model, ctx.obs, ctx.mm, twirl=ctx.recipe.twirl, dd=ctx.recipe.dd).lam_lc
               for sp in [ctx.spec] + list(specs)]
    except Exception as e:                               # noqa: BLE001 - reported; the nominal λ stays usable
        return None, "%s: %s" % (type(e).__name__, str(e)[:200])
    if not lam[0] > 0:
        return None, "the original circuit has no fault in the observable's light cone (noiseless model)"
    return [float(l / lam[0]) for l in lam[1:]], ""


def local_fold_weights(ctx, spec):
    """Calibrated error of every foldable two-qubit gate of ``spec`` (in circuit order) from the context's device model,
    for the error-balanced fold order; None without a device model (uniform noise models: every order is balanced)."""
    dm = getattr(getattr(ctx, "runner", None), "device_model", None)
    if dm is None:
        return None
    out = []
    for o in spec.ops:
        if o.kind == "gate" and o.k == 2:
            e = dm.errors.get((o.name, tuple(o.qubits)))
            if e is None:
                e = dm.errors.get((o.name, tuple(reversed(o.qubits))))
            if e is None:
                e = dm.errors.get((dm.two_qubit, tuple(o.qubits)), dm.errors.get((dm.two_qubit, tuple(reversed(o.qubits)))))
            out.append(np.nan if e is None else float(e))
    w = np.asarray(out, float)
    if not len(w) or not np.isfinite(w).any():
        return None
    w[~np.isfinite(w)] = np.nanmean(w)
    return list(w)


@register
class ZNE(QEMMethod):
    key = "zne"
    label = "Zero-noise extrapolation"
    family = "zne"
    tex = r"\hat E(0)=\sum_j\eta_j\,E(\lambda_j),\qquad \eta_j=\prod_{m\neq j}\frac{\lambda_m}{\lambda_m-\lambda_j}"
    why = (r"Folding $U\to U(U^\dagger U)^k$ runs every noisy gate $\lambda=2k+1$ times, so $E(\lambda)$ can "
           r"only be measured for $\lambda\ge1$. The value at $\lambda=0$ is a guess about the *shape* of "
           r"$E(\lambda)$: a rule that bends to pass through every point (Richardson) has a small bias but large "
           r"weights $|\eta_j|$, which multiply every point's shot noise: $\sigma_{\hat E}=\sum_j|\eta_j|\sigma_j/\sqrt N$.")
    params = (
        Param("scales", r"noise scales $\lambda_j$", "floats", (1.0, 3.0, 5.0), help="Odd integers fold whole circuits; "
              "other values fold part of the circuit."),
        Param("folding", "folding", "choice", "global", ("global", "local"),
              help="global: U(U†U)^k; local: every two-qubit gate G → G(G†G)^k."),
        Param("subset", "gates folded for a fractional λ", "choice", "random", LOCAL_SUBSETS,
              help="Local folding with a fractional λ folds m of the n two-qubit gates once more. random: a seeded "
                   "random subset (1.4.0); nested: a seeded random order, the same gates at every larger λ; "
                   "stratified: spread evenly over the circuit (bit-reversal order); balanced: the subset whose mean "
                   "calibrated gate error matches the circuit's, so λ in gates ≈ λ in noise (needs a device model; "
                   "stratified otherwise).", when=(("folding", ("local",)),)),
        Param("fold_draws", "fold subsets averaged per fractional λ", "int", 1, lo=1, hi=64, step=1,
              help="Which gates a fractional fold repeats changes E(λ) by more than shot noise (the gates differ in "
                   "how much the observable feels them). K > 1 runs K independent random subsets at every fractional "
                   "λ and averages them: the subset's effect averages out, the shots are split over the K circuits "
                   "(no extra shots; K× more distinct circuits).",
              when=(("folding", ("local",)), ("subset", DRAW_SUBSETS))),
        Param("rule", "extrapolation", "choice", "richardson", tuple(RULES)),
        Param("lam_est", "noise scale λ of the fit", "choice", "gates", ("gates", "fault-rate"),
              help="gates: λ = two-qubit gates of the folded circuit ÷ those of the original (the nominal scale). "
                   "fault-rate: λ = light-cone fault rate of the folded circuit ÷ that of the original, from the noise "
                   "model — what the fold does to the noise the observable sees (local folding leaves one-qubit gates "
                   "and most idle time unfolded, so its effective λ is smaller than its nominal one)."),
    )

    def short(self, p):
        sub = p.get("subset", "random")
        fold = p["folding"] + (" %s" % sub if p["folding"] == "local" and sub != "random"
                               and any(abs((float(x) - 1) / 2 - round((float(x) - 1) / 2)) > 1e-9 for x in p["scales"])
                               else "")
        if int(p.get("fold_draws", 1) or 1) > 1 and p["folding"] == "local":
            fold += " ×%d" % int(p["fold_draws"])
        return "ZNE %s (%s %s%s)" % (RULES[p["rule"]], fold, "·".join("%g" % s for s in p["scales"]),
                                     ", λ from fault rate" if p.get("lam_est", "gates") == "fault-rate" else "")

    def run(self, ctx, scales=(1.0, 3.0, 5.0), folding="global", rule="richardson", lam_est="gates",
            subset="random", fold_draws=1) -> MethodResult:
        scales = sorted(float(s) for s in scales)
        if len(scales) < 2:
            raise ValueError("ZNE needs at least two noise scales")
        if min(scales) < 1:
            raise ValueError("noise scales must be at least 1 (the device cannot be made less noisy)")
        if not any(o.kind == "gate" for o in ctx.spec.ops):
            raise ValueError("the circuit has no gates, so there is nothing to fold")
        if subset not in LOCAL_SUBSETS:
            raise ValueError("subset must be one of %s" % ", ".join(LOCAL_SUBSETS))
        if rule == "poly2" and len(set(scales)) < 3:
            raise ValueError("the quadratic fit needs at least three scales")
        draws = int(fold_draws)
        if draws < 1:
            raise ValueError("fold_draws must be at least 1")
        if draws > 1 and (folding != "local" or subset not in DRAW_SUBSETS):
            raise ValueError("fold_draws > 1 averages random fold subsets: it needs local folding with subset "
                             "%s" % " or ".join(DRAW_SUBSETS))
        fw = local_fold_weights(ctx, ctx.spec) if (folding == "local" and subset == "balanced") else None
        # groups[j] = the folded circuits of scale j: one, or `draws` independent subsets for a fractional local fold
        groups = []
        for lam in scales:
            if folding == "global":
                groups.append([fold_global(ctx.spec, lam)])
                continue
            K = draws if is_fractional(lam) else 1
            groups.append([fold_local(ctx.spec, lam, draw_seed(ctx.seed, k), subset, fw) for k in range(K)])
        flat = [fs for g in groups for fs in g]
        where = [j for j, g in enumerate(groups) for _ in g]              # scale of every circuit
        K_of = np.array([len(groups[j]) for j in where], float)
        runs = [ctx.run(fs) for fs in flat]
        Ek = np.array([ctx.expval(rr) for rr in runs], float)
        sk = np.array([np.sqrt(ctx.var_shot(rr)) for rr in runs], float)
        specs = [g[0] for g in groups]
        lams = [achieved_lambda(ctx.spec, fs) for fs in specs]           # every draw of a scale folds as many gates
        E = [float(Ek[[i for i, j in enumerate(where) if j == jj]].mean()) for jj in range(len(groups))]
        # per-shot SD of scale j's average under the optimal split: mean_k σ_jk (so (Σ_j |η_j| σ_j)² = var1)
        sig = [float(np.mean(sk[[i for i, j in enumerate(where) if j == jj]])) for jj in range(len(groups))]
        if len(set(np.round(lams, 9))) < len(lams):
            raise ValueError("the scales %s all fold to λ = %s on this circuit (a partial fold can only add whole "
                             "gates); choose scales that give distinct λ" % (
                                 ", ".join("%g" % s for s in scales), ", ".join("%.3g" % l for l in lams)))
        if lam_est not in ("gates", "fault-rate"):
            raise ValueError("lam_est must be 'gates' or 'fault-rate'")
        lams_gates = list(lams)
        lf_flat, fault_note = fault_rate_lambdas(ctx, flat)
        lams_fault = None if lf_flat is None else [
            float(np.mean([lf_flat[i] for i, j in enumerate(where) if j == jj])) for jj in range(len(groups))]
        if lam_est == "fault-rate":
            if lams_fault is None:
                raise ValueError("λ from the fault rate is not available: %s" % fault_note)
            if len(set(np.round(lams_fault, 9))) < len(lams_fault) or np.any(np.diff(lams_fault) <= 0):
                raise ValueError("the folded circuits do not raise the fault rate monotonically (λ_eff = %s)"
                                 % ", ".join("%.4g" % l for l in lams_fault))
            lams = list(lams_fault)
        est = extrapolate(rule, lams, E)
        eta = weights(rule, lams, E) if np.isfinite(est) else np.full(len(lams), np.nan)
        # the estimate is Σ_j η_j · mean_k E_jk = Σ_jk (η_j / K_j) E_jk: one linear estimator over every circuit
        w = np.abs(np.asarray(eta)[where]) / K_of * sk
        var1 = float(np.sum(w)) ** 2
        alloc = w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else np.full(len(w), 1 / len(w))

        def per_scale(vals) -> list:
            v = np.asarray(vals, float)
            return [float(v[[i for i, j in enumerate(where) if j == jj]].mean()) for jj in range(len(groups))]
        notes = []
        if not np.isfinite(est):
            notes.append("The exponential rule needs all E(λ) of one sign; here they change sign."
                         if rule == "exponential" else "The %s fit gave no finite value at λ = 0." % RULES[rule])
        if any(abs(l - s) > 0.05 * s for l, s in zip(lams_gates, scales)):
            notes.append("Achieved λ = %s for requested %s (folds add whole gates)." % (
                ", ".join("%.3g" % l for l in lams_gates), ", ".join("%g" % s for s in scales)))
        if lams_fault is not None:
            notes.append("Effective noise scale (light-cone fault rate ÷ original): λ_eff = %s; %s." % (
                ", ".join("%.3f" % l for l in lams_fault),
                "used for the fit" if lam_est == "fault-rate" else "the fit uses the nominal λ = %s" % ", ".join(
                    "%.3g" % l for l in lams_gates)))

        def sampler(N, B, rng):
            # exactly N target shots (Σ N_j = N, N_j ≥ 1): the variance-optimal integer allocation, or the
            # largest-remainder rounding of the shares when the weights are degenerate. Fewer shots than circuits
            # cannot give an estimate (every folded circuit needs a shot): NaN, never more shots than N.
            N = int(N)
            if N < len(runs):
                return np.full(int(B), np.nan)
            if np.isfinite(w).all() and w.sum() > 0:
                Nj = optimal_integer_split(w, N)[1]
            else:
                Nj = largest_remainder_split(alloc, N)
            Ek_ = np.stack([ctx.sample_expvals(rr, nj, B, rng) for rr, nj in zip(runs, Nj)], 1)
            return np.array([extrapolate(rule, lams, per_scale(e)) for e in Ek_])

        shown = specs[min(1, len(specs) - 1)]
        fs_meas = [ctx.measured(r) for r in runs]
        o_vec = ctx.obs.o
        return MethodResult(
            mean=est, var1=var1, circuits=len(flat),
            **cal_fields(ctx, lambda mit: float(extrapolate(rule, lams, per_scale([o_vec @ mit(f) for f in fs_meas])))),
            gamma=float(np.nansum(np.abs(eta))) * ctx.readout_gamma(),
            implemented=[("Folded, λ = %.3g" % lams[min(1, len(lams) - 1)], display_circuit(ctx, shown))]
            + [("Folded, λ = %.3g" % l, display_circuit(ctx, sp)) for l, sp in zip(lams, specs) if sp is not shown],
            details=dict(lams=lams, E=E, sigma=sig, eta=eta, alloc=per_scale(alloc * K_of), rule=rule, folding=folding, lam_est=lam_est,
                         subset=subset if folding == "local" else None, draws=[len(g) for g in groups],
                         a_circuits=[float(x) for x in w],
                         E_draws=[[float(Ek[i]) for i, j in enumerate(where) if j == jj] for jj in range(len(groups))],
                         lams_gates=lams_gates, lams_fault=lams_fault, depths=[int(sp.depth()) for sp in specs],
                         twoq=[int(sp.count("2q")) for sp in specs]), notes=notes, sampler=sampler)
