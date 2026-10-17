r"""Computations behind the analysis pages (no Streamlit): cost vs precision, the PEC exponential wall,
bias–variance darts, model mismatch and head-to-head sweeps.

Every estimator in the app reports an exact mean and a per-shot variance v; with training shots F, calibration
(c circuits, coefficient w) and the training-noise floor u, a total budget B gives (qemx/budget.py)

.. math::  \mathrm{RMSE}(B)=\sqrt{b^2+v/N_t(B)+w/s(B)+u},\qquad B=N_t+F+cs,

which reduces to :math:`\sqrt{b^2+v/N}` and :math:`N_\varepsilon=v/(\varepsilon^2-b^2)`, :math:`N^*=(v_m-v_r)/(b_r^2-b_m^2)`
without training and calibration. That is what makes the shot axis free: one simulation per setting gives the whole
curve.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .compile import Compiled, CompileOptions, build, with_io
from .methods import REGISTRY, Context, Recipe, Runner
from .noise import NoiseSpec
from .recipes import compare, family
from .resources import check


# ------------------------------------------------------------------ cost vs precision
def per_shot_variance(row: dict, B: int) -> float:
    """The per-shot variance v of a comparison-table row, consistent with its displayed sd at budget B: the analytic
    v (rows since v1.2 book the excess spread of learning repeats as the constant u_train), or for older rows whose
    sd came from repeats v = (sd² − calibration term)·N_t."""
    if "u_train" in row or row.get("sd from") != "repeats" or not np.isfinite(row.get("sd", np.nan)):
        return float(row.get("v", float(row["sd"]) ** 2 * B))     # rows with u_train carry the full cost model
    from . import budget as BG
    m = BG.CostModel.of_row(row)
    N, s = BG.split(m, B)
    cal = m.w / s if s > 0 else 0.0
    return max(float(row["sd"]) ** 2 - cal, 0.0) * max(N, 1.0)


def cost_fields(r) -> dict:
    """The equal-budget cost model of a MethodResult as plain row fields (see qemx/budget.py)."""
    from .budget import CostModel
    m = CostModel.of(r)
    return dict(v=m.v, fixed_shots=m.fixed, cal_circuits=m.c, w_cal=m.w, u_train=m.u)


def rmse_row(row: dict, budgets) -> np.ndarray:
    """RMSE of a row at total budgets B (target, training and calibration shots included)."""
    from . import budget as BG
    return BG.rmse_curve(float(row["bias"]), BG.CostModel.of_row(row), budgets)


def budget_row(row: dict, eps: float) -> float:
    from . import budget as BG
    return BG.budget_for(float(row["bias"]), BG.CostModel.of_row(row), eps)


def crossover_rows(row: dict, ref: dict):
    from . import budget as BG
    return BG.crossover(float(row["bias"]), BG.CostModel.of_row(row), float(ref["bias"]), BG.CostModel.of_row(ref))


def rmse_curve(bias: float, v: float, shots) -> np.ndarray:
    shots = np.asarray(shots, float)
    return np.sqrt(bias ** 2 + v / shots)


def shots_for(bias: float, v: float, eps: float) -> float:
    """Target shots for RMSE ≤ eps (inf when the bias alone exceeds eps)."""
    if eps <= abs(bias):
        return math.inf
    return v / (eps ** 2 - bias ** 2)


def crossover(b_m: float, v_m: float, b_r: float, v_r: float) -> float | None:
    """Shots above which method m beats reference r (None if never / always)."""
    den = b_r ** 2 - b_m ** 2
    num = v_m - v_r
    if abs(den) < 1e-18:
        return None
    x = num / den
    return x if x > 0 else None


def depol_gamma(p: float, k: int, construction: str = "inverse") -> float:
    """PEC sampling overhead γ of one gate with depolarizing probability p on k qubits."""
    from .methods.pec import depol_quasi, gamma_of
    if float(p) >= 1.0 - 1e-12:                  # ρ → I/d: the channel is not invertible, PEC is impossible
        return float("inf")
    return gamma_of(depol_quasi(float(p), k, construction))


def pec_wall(layers, p2_values, n_qubits: int, p1: float, eps: float, t_shot_us: float, construction="inverse",
             twoq_per_layer: float | None = None, oneq_per_layer: float | None = None, sigma: float = 1.0) -> dict:
    """Shots and QPU time PEC needs for precision eps on a brickwork circuit of `layers` layers.

    Each layer has ⌊n/2⌋ two-qubit gates and n one-qubit gates (by default), so
    Γ(L) = γ₁^{n L} γ₂^{⌊n/2⌋ L} and N(L) = Γ(L)² σ² / ε² — exponential in L."""
    layers = np.asarray(layers, float)
    n2 = (n_qubits // 2) if twoq_per_layer is None else twoq_per_layer
    n1 = n_qubits if oneq_per_layer is None else oneq_per_layer
    g1 = depol_gamma(p1, 1, construction) if p1 > 0 else 1.0
    out = {}
    for p2 in p2_values:
        g2 = depol_gamma(p2, 2, construction) if p2 > 0 else 1.0
        per_layer = n1 * math.log(g1) + n2 * math.log(g2)
        with np.errstate(invalid="ignore"):
            logG = np.where(layers > 0, layers * per_layer, 0.0)       # 0 layers cost nothing, even at γ = ∞
        N = np.exp(2 * logG) * sigma ** 2 / eps ** 2
        out[float(p2)] = dict(gamma=np.exp(logG), shots=N, seconds=N * t_shot_us * 1e-6)
    return out


def mirror_family(spec, k: int):
    """U (U† U)^k: a circuit k·2 times deeper with the same ideal answer (mirror / echo benchmarking)."""
    from .transforms import fold_global
    return fold_global(spec, 2 * k + 1)


def depth_sweep(make_compiled, depths, noise: NoiseSpec, observable_fn, recipes: list, backend: str, shots: int,
                symmetry=None, progress=None) -> list:
    """Simulate every recipe at every depth: [{depth, method, family, estimate, ideal, bias, v, gamma}]."""
    out = []
    for j, d in enumerate(depths):
        check()
        compiled, obs = make_compiled(d), observable_fn(d)
        ideal, outs, _ = compare(compiled, noise, obs, recipes, backend, shots, symmetry, check_budget=False)
        for o in outs:
            if o.result is None:
                continue
            r = o.result
            out.append(dict(depth=int(d), method=o.title, family=family(o.recipe), estimate=r.mean, ideal=ideal,
                            bias=r.mean - ideal, gamma=r.gamma, twoq=compiled.implemented.count("2q"),
                            **cost_fields(r)))
        if progress:
            progress((j + 1) / len(depths))
    return out


# ------------------------------------------------------------------ bias–variance darts
def darts(outs_a: list, outs_b: list, ideal_a: float, ideal_b: float, N: int, B: int, seed: int = 3) -> list:
    """For each method: B repeated experiments of N shots each, as (error in O_A, error in O_B) points.
    The two coordinates come from independent experiments (one per observable)."""
    rng = np.random.default_rng(seed)
    by_b = {o.title: o for o in outs_b}
    res = []
    for oa in outs_a:
        ob = by_b.get(oa.title)
        if oa.result is None or ob is None or ob.result is None:
            continue
        xa = _samples(oa.result, N, B, rng) - ideal_a
        xb = _samples(ob.result, N, B, rng) - ideal_b
        ok = np.isfinite(xa) & np.isfinite(xb)
        from .budget import variance as _var, CostModel as _CM
        res.append(dict(method=oa.title, family=family(oa.recipe), x=xa[ok], y=xb[ok],
                        bias=(oa.result.mean - ideal_a, ob.result.mean - ideal_b),
                        var=(_var(_CM.of(oa.result), N), _var(_CM.of(ob.result), N)), failed=int((~ok).sum())))
    return res


def _samples(r, N, B, rng):
    """B experiments at total budget N (budget.draw); a Gaussian with the budget variance as fallback."""
    from . import budget as BG
    try:
        x = BG.draw(r, int(N), int(B), rng)
        if x is not None:
            return x
    except Exception:                                              # noqa: BLE001 - fall back below
        pass
    var = BG.variance(BG.CostModel.of(r), int(N))
    if not math.isfinite(var):
        return np.full(int(B), np.nan)
    return rng.normal(r.mean, math.sqrt(var), size=int(B))


def mse_parts(d: dict, N: int) -> dict:
    """Squared bias and variance at the budget the darts were thrown with (``var`` is per experiment)."""
    bx, by = d["bias"]
    vx, vy = d["var"]
    return dict(bias2=bx ** 2 + by ** 2, var=vx + vy)


# ------------------------------------------------------------------ model mismatch
def pec_mismatch(compiled: Compiled, true_noise: NoiseSpec, observable, ratios, construction: str = "inverse",
                 scope: str = "all", backend: str = "numpy", shots: int = 4096, layout=None, progress=None) -> list:
    """PEC told p_assumed = ratio × p_true (both p1 and p2), run against the true noise.
    Returns [{ratio, p1, p2, estimate, bias, sd, gamma}] plus the unmitigated reference (ratio = None).
    On a device backend `true_noise` is the device's median-rate model and the device model is the truth."""
    from .hamiltonian import ideal_of, run_recipe
    runner = Runner(backend, layout=layout)
    rows = []
    base = Recipe.make("raw")
    ideal = ideal_of(compiled, observable, runner, true_noise)
    raw = run_recipe(compiled, true_noise, observable, base, runner, shots)
    rows.append(dict(ratio=None, p1=None, p2=None, estimate=raw.mean, bias=raw.mean - ideal,
                     sd=math.sqrt(raw.var1 / shots), gamma=1.0, ideal=ideal))
    for r in ratios:
        check()
        p1, p2 = true_noise.p1 * r, true_noise.p2 * r
        rc = Recipe.make("pec", construction=construction, model="depolarizing", p1=p1, p2=p2, scope=scope)
        res = run_recipe(compiled, true_noise, observable, rc, runner, shots)
        rows.append(dict(ratio=float(r), p1=p1, p2=p2, estimate=res.mean, bias=res.mean - ideal,
                         sd=math.sqrt(max(res.var1, 0) / shots), gamma=res.gamma, ideal=ideal))
        if progress:
            progress(len(rows) / (len(ratios) + 1))
    return rows


def pec_mismatch_map(spec_logical, inputs, basis, opts: CompileOptions, relabel, noise: NoiseSpec, observable,
                     assumed_p2, true_p2, keep_ratio_p1: bool = True, construction: str = "inverse",
                     progress=None) -> np.ndarray:
    """Bias of PEC over a grid (true p2 rows × assumed p2 columns); p1 scales with p2 when keep_ratio_p1."""
    compiled = build(spec_logical, inputs, basis, opts, relabel)
    B = np.full((len(true_p2), len(assumed_p2)), np.nan)
    r1 = (noise.p1 / noise.p2) if (keep_ratio_p1 and noise.p2 > 0) else 0.0
    for i, tp in enumerate(true_p2):
        check()
        tn = dataclasses.replace(noise, p2=float(tp), p1=float(tp) * r1 if keep_ratio_p1 else noise.p1)
        from .hamiltonian import ideal_of, run_recipe
        runner = Runner("numpy")
        ideal = ideal_of(compiled, observable, runner, tn)
        for j, ap in enumerate(assumed_p2):
            rc = Recipe.make("pec", construction=construction, model="depolarizing",
                             p1=float(ap) * r1 if keep_ratio_p1 else tn.p1, p2=float(ap))
            B[i, j] = run_recipe(compiled, tn, observable, rc, runner).mean - ideal
        if progress:
            progress((i + 1) / len(true_p2))
    return B


def readout_mismatch(compiled: Compiled, noise: NoiseSpec, observable, drifts) -> list:
    """Tensored readout mitigation calibrated with rates × (1 + drift) while the device has the true rates.
    A Pauli-sum Hamiltonian is handled setting by setting (each with its basis change) and summed."""
    from .hamiltonian import PauliSum, expand, ideal_of
    runner = Runner("numpy")
    ideal = ideal_of(compiled, observable, runner)
    parts = ([(c, o) for c, o, _, _ in expand(observable, compiled)] if isinstance(observable, PauliSum)
             else [(compiled, observable)])
    const = observable.constant if isinstance(observable, PauliSum) else 0.0
    runs = []
    for c, o in parts:
        rr = Context(c, noise, o, Recipe.make("raw"), runner).run()
        runs.append((rr.probs_meas, o.o, c.implemented.n))
    out = []
    for dr in drifts:
        e01, e10 = noise.ro01 * (1 + dr), noise.ro10 * (1 + dr)
        A = np.array([[1 - e01, e10], [e01, 1 - e10]])
        Ai = np.linalg.inv(A)
        est, raw = const, const
        for f, o, n in runs:
            t = f.reshape([2] * n)
            for ax in range(n):
                t = np.moveaxis(np.tensordot(Ai, t, axes=([1], [ax])), 0, ax)
            est += float(o @ t.reshape(-1))
            raw += float(o @ f)
        out.append(dict(drift=float(dr), estimate=est, bias=est - ideal, ideal=ideal, raw=raw))
    return out


# ------------------------------------------------------------------ head to head
SWEEPS = {
    "p2": ("two-qubit depolarizing p₂", lambda n, v: dataclasses.replace(n, p2=v)),
    "p1": ("one-qubit depolarizing p₁", lambda n, v: dataclasses.replace(n, p1=v)),
    "scale": ("all gate and idle noise × s", None),
    "over_rot": ("coherent over-rotation ε", lambda n, v: dataclasses.replace(n, over_rot=v)),
    "zz_phase": ("ZZ phase θ_ZZ", lambda n, v: dataclasses.replace(n, zz_phase=v)),
    "amp_damp": ("amplitude damping γ", lambda n, v: dataclasses.replace(n, amp_damp=v)),
    "field_sigma": ("quasi-static field σ", lambda n, v: dataclasses.replace(n, field_sigma=v)),
    "readout": ("readout error ε₀₁ = ε₁₀/2", lambda n, v: dataclasses.replace(n, ro01=v, ro10=2 * v)),
}


def apply_sweep(noise: NoiseSpec, key: str, v: float) -> NoiseSpec:
    if key == "scale":
        return noise.scaled(v) if hasattr(noise, "scaled") else noise
    return SWEEPS[key][1](noise, float(v))


def sweep(compiled: Compiled, noise: NoiseSpec, observable, recipes: list, key: str, values, backend: str,
          shots: int, symmetry=None, progress=None) -> list:
    """Simulate each recipe at each value of one noise parameter: rows with bias and per-shot variance."""
    out = []
    for j, v in enumerate(values):
        check()
        nz = apply_sweep(noise, key, v)
        ideal, outs, _ = compare(compiled, nz, observable, recipes, backend, shots, symmetry,
                                 check_budget=False)       # the winner map costs every budget from the cost model
        for o in outs:
            if o.result is None:
                out.append(dict(value=float(v), method=o.title, family=family(o.recipe), estimate=np.nan,
                                bias=np.nan, v=np.nan, gamma=np.nan, ideal=ideal, error=o.error, fixed_shots=0,
                                cal_circuits=0, w_cal=0.0))
                continue
            r = o.result
            out.append(dict(value=float(v), method=o.title, family=family(o.recipe), estimate=r.mean,
                            bias=r.mean - ideal, gamma=r.gamma, ideal=ideal, circuits=r.circuits, error="",
                            **cost_fields(r)))
        if progress:
            progress((j + 1) / len(values))
    return out


def winner_map(rows: list, shots_grid) -> tuple:
    """(values, methods, index of the lowest-RMSE method [value, shots], its RMSE, runner-up RMSE)."""
    vals = sorted({r["value"] for r in rows})
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    shots_grid = np.asarray(shots_grid, float)
    R = np.full((len(vals), len(methods), len(shots_grid)), np.inf)
    for r in rows:
        if not np.isfinite(r["bias"]):
            continue
        i, m = vals.index(r["value"]), methods.index(r["method"])
        R[i, m] = rmse_row(r, shots_grid)
    win = np.argmin(R, axis=1)
    srt = np.sort(R, axis=1)
    return vals, methods, win, srt[:, 0, :], srt[:, 1, :] if R.shape[1] > 1 else srt[:, 0, :]
