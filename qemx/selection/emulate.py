r"""Trial-level emulation of complete protocols from cached exact distributions (1.6.0).

One *trial* is one complete deployment of a protocol at a total budget B: a finite readout calibration (θ̂ drawn
from the calibration circuits' exact output law), finite training data where the protocol trains, integer target
shots split by the protocol's declared rule, the protocol's own post-processing, and its declared failure handling.
Repeating trials with independent randomness gives the protocol's full error law at B, so

    R_a(𝒯, B) = E[(μ̂_a − μ)²] = Bias² + Var,     Var = E_D[Var(μ̂|D)] + Var_D[E(μ̂|D)]   (Eqs. 1.5, 1.6)

includes the training and calibration randomness, and nonlinear estimators (exponential ZNE, refitted learning
models) are never evaluated on exact probabilities in place of finite data.

Why sampling from cached distributions is exact here. Every circuit of these protocols is a fixed circuit executed
shot by shot with i.i.d. outcomes under a stationary execution law, so its N-shot counts are multinomial in its exact
output distribution — which the exact density-matrix engine computes once per circuit (cached by the method's
runner). Correlations are kept where they exist: one θ̂ per trial is shared by every circuit of that trial; ZNE's
folds and NIL's neighbors share θ̂ and the fitted coefficients. PEC with one shot per sampled circuit draws each shot
from the exact sign-resolved law: P(+, x) = (p_abs(x) + p_net(x))/2, P(−, x) = (p_abs(x) − p_net(x))/2, where p_net
is the quasi-probability-weighted and p_abs the |q|-weighted output distribution over the sampled instances (a shot
is an instance draw followed by a measurement, so this two-stage law is exact). Not covered by a static law:
temporal drift within a run and shot-to-shot correlations; drift scenarios are emulated as blocks with different
static laws (``qemx.selection.mismatch``). The emulator is validated against direct execution (``validate``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .protocols import PORTFOLIO, Allocation, allocate

FLAGS = ("cal_fallback", "fit_fallback", "learn_fallback", "nonresponse")
NO_DESIGN = -1                      # design index of a trial of a protocol without a training design


@dataclass(frozen=True)
class DesignPlan:
    """How learning protocols' training designs are drawn (1.7.0; review finding R1).

    A training design is the random part of a learning protocol that is fixed before any shot: CDR's choice of
    near-Clifford training circuits, NIL's training circuits and sampled Pauli neighbors. Two estimands differ:

    * ``fixed-shared`` (1.6.0): K designs drawn once per task and shared by both label halves and every budget,
      trials spread over them round-robin. The labels estimate the risk *conditional on these K designs*; the two
      halves are independent only conditionally on them, and the between-design variance enters with K, not r.
    * ``fresh`` (1.7.0 default): the risk of a protocol that draws a *fresh* design at every deployment. Each label
      half has its own pool of ``per_half`` independent designs (selection pool ≠ evaluation pool: no design is in
      both halves), and the trials of a half are spread over its pool (m = r/K repetitions per design). The same
      pool serves every budget of a half — a declared pairing across budgets (common random numbers), not shared
      between halves. A protocol's allocation is computed from a separate *allocation design* (the task's base
      seed), so it never depends on a design that is scored.

    For K independent designs D with m repetitions each, Var(R̂) = Var_D(E[L|D])/K + E_D[Var(L|D)]/(K m): more
    repetitions do not shrink the first term. Labels therefore report a design-level standard error (the spread of
    the per-design mean losses, cluster-robust for unequal m) beside the trial-level one."""
    mode: str = "fixed-shared"
    count: int = 1                  # fixed-shared: designs shared by both halves (1.6.0 ``learning_designs``)
    per_half: int = 0               # fresh: independent designs per label half

    @staticmethod
    def from_config(v) -> "DesignPlan":
        if v is None:
            return DesignPlan()
        if isinstance(v, DesignPlan):
            return v
        if isinstance(v, bool):
            raise ValueError("learning_designs must be an integer or a mapping, not %r" % (v,))
        if isinstance(v, int):
            if v < 1:
                raise ValueError("learning_designs must be ≥ 1, got %d" % v)
            return DesignPlan("fixed-shared", int(v), 0)
        if isinstance(v, dict):
            extra = set(v) - {"mode", "per_half", "count"}
            if extra:
                raise ValueError("unknown learning_designs keys: %s" % ", ".join(sorted(extra)))
            mode = v.get("mode", "fresh")
            if mode == "fresh":
                k = int(v.get("per_half", 0))
                if k < 1:
                    raise ValueError("learning_designs.per_half must be ≥ 1 for fresh designs")
                return DesignPlan("fresh", 1, k)
            if mode == "fixed-shared":
                return DesignPlan("fixed-shared", int(v.get("count", 1)), 0)
            raise ValueError("learning_designs.mode must be 'fresh' or 'fixed-shared', got %r" % (mode,))
        raise ValueError("learning_designs must be an integer or a mapping, not %r" % (v,))

    def to_json(self):
        """The configuration value (an integer for the 1.6.0 form, so its configuration digests are unchanged)."""
        if self.mode == "fixed-shared":
            return int(self.count)
        return dict(mode="fresh", per_half=int(self.per_half))

    @property
    def estimand(self) -> str:
        return ("risk conditional on %d fixed designs shared by both halves" % self.count
                if self.mode == "fixed-shared" else "fresh-design risk (independent design pools per half)")


def pool_seed(seed: int, half: int, k: int) -> int:
    """Method seed of design k of a half's fresh pool: a SeedSequence of (seed, 'DESG', half, k), 31 bits."""
    ss = np.random.SeedSequence([int(seed) % (2 ** 63), 0x44455347, int(half), int(k)])
    return int(ss.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def structural_fallback(res) -> str:
    """'' when a learning method result applies its learned correction; otherwise why it returned its declared
    fallback (the readout-corrected target value) for every trial: 'clifford' (nothing to learn: the circuit is
    Clifford) or 'degenerate-training' (every training circuit has the same ideal value)."""
    if res is None or isinstance(res, Exception):
        return ""
    d = getattr(res, "details", None) or {}
    if d.get("degenerate"):
        return "degenerate-training"
    if not d or ("n_features" not in d and "names" not in d):
        return "clifford"
    return ""


def design_record(key: str, res, ctx, *, seed: int, half: int, index: int, ideal: float | None = None) -> dict:
    """Identity and diagnostics of one training design (stored in the label record). ``ideal`` (oracle) adds the
    research-only diagnostics: the exact-feature fit's error and whether μ lies inside the training labels' range."""
    import hashlib
    import json as _json
    rec = dict(protocol=key, half=int(half), index=int(index), seed=int(seed),
               design_id=hashlib.sha256(("%s|%d" % (key, int(seed))).encode()).hexdigest()[:16])
    if isinstance(res, Exception):
        rec.update(status="error", error="%s: %s" % (type(res).__name__, str(res)[:200]))
        return rec
    d = dict(getattr(res, "details", None) or {})
    fb = structural_fallback(res)
    rec.update(status="ok", fallback=fb)

    def _arr(x):
        return None if x is None else np.round(np.asarray(x, float), 12).tolist()
    content = [d.get("names"), _arr(d.get("y_train")), _arr(d.get("X")), _arr(d.get("x_train"))]
    rec["content_digest"] = hashlib.sha256(_json.dumps(content, default=str).encode()).hexdigest()[:16]
    y = np.asarray(d.get("y_train") if d.get("y_train") is not None else [], float)
    X = d.get("X")
    diag = dict(n_train=int(len(y)), n_features=int(d.get("n_features") or 0), rank=d.get("rank"),
                spread=d.get("spread"), spread_needed=d.get("spread_needed"),
                l1=d.get("l1"), nonzero=d.get("nonzero"), solver_ok=d.get("solver_ok"),
                twins=len(d.get("twins") or []), y_min=float(y.min()) if y.size else None,
                y_max=float(y.max()) if y.size else None, warning=(d.get("warning") or "")[:200])
    if X is not None and np.size(X):
        sv = np.linalg.svd(np.asarray(X, float), compute_uv=False)
        diag.update(sv_max=float(sv[0]), sv_min=float(sv[-1]),
                    condition=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"))
    ref = getattr(res, "reference_mean", None)
    est = ref if ref is not None else getattr(res, "mean", None)
    diag["exact_feature_estimate"] = None if est is None else float(est)
    if ideal is not None and est is not None:          # oracle diagnostics (research only)
        diag["oracle_error"] = float(est - ideal)
        diag["oracle_ideal_in_label_range"] = bool(y.size and y.min() - 1e-12 <= ideal <= y.max() + 1e-12)
    rec["diagnostics"] = {k: (None if isinstance(v, float) and not np.isfinite(v) and v != float("inf") else v)
                          for k, v in diag.items()}
    return rec


@dataclass
class Prepared:
    """The method results of every protocol on one task, under the execution law (truth) and under the supplied
    model (allocations). Holding them keeps the runners' cached exact distributions."""
    built: object
    truth: dict = field(default_factory=dict)          # protocol -> MethodResult | Exception
    supplied: dict = field(default_factory=dict)
    ctx_truth: dict = field(default_factory=dict)      # protocol -> Context
    seconds: dict = field(default_factory=dict)
    same_law: bool = True
    designs: dict = field(default_factory=dict)        # fixed-shared: learning protocol -> [(MethodResult, Context)]
    pools: dict = field(default_factory=dict)          # fresh: learning protocol -> {half: [(MethodResult, Context)]}
    design_records: dict = field(default_factory=dict)  # learning protocol -> [design_record] (both halves)
    plan: DesignPlan = field(default_factory=DesignPlan)


def _context(built, recipe, runner, noise, seed: int, moments: bool = True):
    from ..methods import Context
    ctx = Context(built.compiled, noise, built.observable, recipe, runner, shots=4096, seed=int(seed))
    ctx.protocol_moments = bool(moments)               # learning methods: Monte-Carlo moments not needed by trials
    return ctx


def design_seed(seed: int, k: int) -> int:
    return int((int(seed) + 7919 * int(k)) % (2 ** 31))


def prepare(built, protocols=None, seed: int = 0, designs=1, halves=(0, 1)) -> Prepared:
    """Run every protocol's method once on the truth runner (exact distributions, PEC's signed laws, learning's
    training sets) and, if the supplied model differs from the truth, once on the supplied runner (allocations).
    PEC's assumed model is always the *supplied* device's median-rate model, never the truth's.

    Learning protocols draw their training design at random (CDR: which non-Clifford slots become Clifford; NIL:
    the 2-design training circuits and the sampled neighbors). A deployment draws a fresh design, so that randomness
    is part of the protocol's risk: with ``designs`` = K > 1 the method is run for K design seeds and the trials are
    spread over them (``trials``), so the label contains the design-to-design variance (from K draws).

    ``designs`` is a ``DesignPlan`` or its configuration value (an integer: 1.6.0's fixed designs shared by both
    halves; ``{mode: fresh, per_half: K}``: independent pools per half, see ``DesignPlan``). Extra designs are
    prepared without the methods' Monte-Carlo moments (``Context.protocol_moments``), which trials never use.
    ``halves``: the label halves whose fresh pools are drawn (both for labels; one for a study that uses one half)."""
    from ..methods import REGISTRY, Runner
    from .. import device
    spec = built.spec
    phys = tuple(built.compiled.info["phys"])
    from .mismatch import truth_runner
    same = not spec.truth
    r_sup = Runner(backend=spec.supplied_backend, layout=phys)
    r_truth = r_sup if same else truth_runner(spec, phys)
    sup_noise = device.get_model(spec.supplied_backend, phys).equivalent_noise()
    plan = DesignPlan.from_config(designs)
    out = Prepared(built, same_law=same, plan=plan)
    for key in (protocols or list(PORTFOLIO)):
        pr = PORTFOLIO[key]
        rc = pr.recipe()
        meth = REGISTRY[pr.estimator]
        t0 = time.perf_counter()
        ctx_t = _context(built, rc, r_truth, sup_noise, seed)
        try:
            out.truth[key] = meth.run(ctx_t, **rc.p)
            out.ctx_truth[key] = ctx_t
        except Exception as e:                       # noqa: BLE001 - recorded: the protocol is unavailable
            out.truth[key] = e
        if same:
            out.supplied[key] = out.truth[key]
        else:
            try:
                out.supplied[key] = meth.run(_context(built, rc, r_sup, sup_noise, seed), **rc.p)
            except Exception as e:                   # noqa: BLE001
                out.supplied[key] = e
        ideal = float(built.ideal) if built.ideal is not None else None    # deployment: unknown (no oracle)

        def draw(sd, half, idx):
            ctx_k = _context(built, rc, r_truth, sup_noise, sd, moments=False)
            try:
                res_k = meth.run(ctx_k, **rc.p)
            except Exception as e:                   # noqa: BLE001 - a design that cannot be built is recorded
                res_k = e
            out.design_records.setdefault(key, []).append(design_record(key, res_k, ctx_k, seed=sd, half=half,
                                                                        index=idx, ideal=ideal))
            return res_k, ctx_k
        if pr.learning and not isinstance(out.truth[key], Exception):
            if plan.mode == "fixed-shared" and plan.count > 1:
                out.designs[key] = [(out.truth[key], out.ctx_truth[key])]
                out.design_records.setdefault(key, []).append(design_record(
                    key, out.truth[key], out.ctx_truth[key], seed=seed, half=-1, index=0, ideal=ideal))
                for k in range(1, plan.count):
                    out.designs[key].append(draw(design_seed(seed, k), -1, k))
            elif plan.mode == "fresh":
                used = {int(seed) % (2 ** 31)}
                out.pools[key] = {}
                for half in halves:
                    pool, k = [], 0
                    while len(pool) < plan.per_half:
                        sd = pool_seed(seed, half, k)
                        k += 1
                        if sd in used:                   # never reuse a seed (allocation design or the other half)
                            continue
                        used.add(sd)
                        pool.append(draw(sd, half, len(pool)))
                    out.pools[key][half] = pool
        out.seconds[key] = time.perf_counter() - t0
    return out


@dataclass
class Trials:
    protocol: str
    B: int
    allocation: dict
    estimates: np.ndarray                  # (R,) unclipped estimates (NaN: nonresponse)
    flags: dict                            # name -> (R,) bool
    seconds: float = 0.0
    design: np.ndarray | None = None       # (R,) index of the trial's design in its pool (NO_DESIGN: none)
    status: str = "ok"                     # ok | infeasible | method-error

    def errors(self, mu: float) -> np.ndarray:
        """Signed errors with the declared nonresponse estimate 0 (loss μ²) substituted for missing estimates."""
        e = np.where(np.isfinite(self.estimates), self.estimates, 0.0) - float(mu)
        return e


def trials(prep: Prepared, key: str, B: int, R: int, rng, *, diagnostics: int = 0, selection: int = 0,
           half: int = 0) -> Trials:
    """R independent complete trials of protocol ``key`` at total budget B (Allocation from the supplied model).
    ``half`` (0 selection, 1 evaluation) chooses the design pool of a learning protocol under fresh designs.

    Status: ``method-error`` when the method could not be run at all on this task (a software or modeling failure,
    with its message — never mixed with infeasibility), ``infeasible`` when the allocation cannot meet the
    protocol's minima at B (the reason is recorded), else ``ok``."""
    pr = PORTFOLIO[key]
    res_t, res_s = prep.truth.get(key), prep.supplied.get(key)
    R = int(R)
    nan = np.full(R, np.nan)
    none = np.full(R, NO_DESIGN, np.int16)
    if isinstance(res_t, Exception) or isinstance(res_s, Exception) or res_t is None or res_s is None:
        why = repr(res_t if isinstance(res_t, Exception) else res_s)[:200]
        a = Allocation(key, int(B), False, "method failed: " + why).to_json()
        a["status"] = "method-error"
        return Trials(key, int(B), a, nan, {f: np.zeros(R, bool) for f in FLAGS}, design=none,
                      status="method-error")
    alloc = allocate(pr, res_s, int(B), diagnostics=diagnostics, selection=selection)
    if not alloc.feasible:
        a = alloc.to_json()
        a["status"] = "infeasible"
        return Trials(key, int(B), a, nan, {f: np.zeros(R, bool) for f in FLAGS}, design=none, status="infeasible")
    t0 = time.perf_counter()
    a_json = alloc.to_json()
    a_json["status"] = "ok"
    pool = None
    if key in prep.pools:                              # fresh designs: this half's own pool
        pool = prep.pools[key][int(half)]
    elif key in prep.designs:                          # fixed designs shared by both halves (1.6.0)
        pool = prep.designs[key]
    if pool is not None:                               # learning: trials spread over the pool, round-robin
        est = np.full(R, np.nan)
        flags = {f: np.zeros(R, bool) for f in FLAGS}
        design = np.arange(R, dtype=np.int16) % len(pool)
        for k, (res_k, ctx_k) in enumerate(pool):
            rows = np.arange(k, R, len(pool))
            if not len(rows):
                continue
            if isinstance(res_k, Exception):
                flags["nonresponse"][rows] = True
                continue
            sub = _run(pr, res_k, ctx_k, alloc, len(rows), rng)
            est[rows] = sub[0]
            for f in FLAGS:
                flags[f][rows] = sub[1][f]
        flags["nonresponse"] = ~np.isfinite(est)
        return Trials(key, int(B), a_json, est, flags, time.perf_counter() - t0, design=design)
    est, flags = _run(pr, res_t, prep.ctx_truth[key], alloc, R, rng)
    design = np.zeros(R, np.int16) if pr.learning else none
    return Trials(key, int(B), a_json, est, flags, time.perf_counter() - t0, design=design)


def _run(pr, res_t, ctx, alloc, R: int, rng) -> tuple:
    """(estimates, flags) of R trials of one method result with a given allocation."""
    flags = {f: np.zeros(R, bool) for f in FLAGS}
    theta = None
    if pr.readout != "none":
        theta = ctx.sample_cal_theta(int(alloc.s), R, rng)
        bad = np.array([not ctx.theta_invertible(t) for t in theta])
        if bad.any():                                  # declared fallback: no readout correction in that trial
            theta = theta.copy()
            theta[bad] = 0.0
            flags["cal_fallback"] = bad
    kw = dict(theta=theta)
    if pr.estimator == "zne":
        est, failed = res_t.sampler(int(alloc.N), R, rng, theta=theta, split=list(alloc.split), with_failures=True)
        if dict(pr.params).get("rule") == "exponential":
            flags["fit_fallback"] = np.asarray(failed, bool) & np.isfinite(est)
    elif pr.estimator == "learning":
        est = res_t.sampler(int(alloc.N), R, rng, theta=theta, deployable=True, s_tr=int(alloc.s_tr))
        fb = getattr(res_t.sampler, "fallbacks", None)
        if fb is not None:
            flags["learn_fallback"] = np.asarray(fb, bool)
    elif pr.estimator == "nil":
        est = res_t.sampler(int(alloc.N), R, rng, theta=theta, s_tr=int(alloc.s_tr))
    else:
        est = res_t.sampler(int(alloc.N), R, rng, **kw)
    est = np.asarray(est, float)
    if pr.learning and structural_fallback(res_t):      # declared fallback for every trial (1.7.0: was unflagged
        flags["learn_fallback"] = np.ones(R, bool)      # for NIL, and for CDR on Clifford/degenerate designs)
    flags["nonresponse"] = ~np.isfinite(est)
    return est, flags


# ------------------------------------------------------------------ validation against direct execution
def _shots(p: np.ndarray, N: int, rng) -> np.ndarray:
    """Counts of N single-shot draws (one categorical draw per shot, not a multinomial call)."""
    p = np.clip(np.asarray(p, float), 0, None)
    p = p / p.sum()
    x = rng.choice(len(p), size=int(N), p=p)
    return np.bincount(x, minlength=len(p)).astype(float)


def _theta_direct(ctx, s: int, rng):
    """One finite calibration by single-shot draws on the two calibration circuits; the singular-θ̂ fallback."""
    from ..circuit import CircuitSpec, Op
    n = ctx.n
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
    zero = CircuitSpec(n, (), "cal-0")
    one = CircuitSpec(n, tuple(Op("x", (q,), (), "gate", "prep") for q in range(n)), "cal-1")
    c0 = _shots(ctx.runner.run(zero, ctx.noise, ctx.mm, role="calibration").probs_meas, s, rng)
    c1 = _shots(ctx.runner.run(one, ctx.noise, ctx.mm, role="calibration").probs_meas, s, rng)
    th = np.empty(2 * n)
    th[0::2] = c0 @ bits / float(s)
    th[1::2] = c1 @ (1 - bits) / float(s)
    return th if ctx.theta_invertible(th) else np.zeros_like(th)


def direct_trials(prep: Prepared, key: str, B: int, R: int, rng) -> np.ndarray:
    """R trials of the same protocol and allocation by *direct execution*, re-implemented independently of the
    methods' samplers: every shot is its own categorical draw from the executed circuit's exact distribution; the
    calibration is drawn shot by shot; ZNE fits its own polynomial (numpy.polyfit); PEC samples an explicit instance
    circuit (one correction per gate location with probability |q|/γ), simulates it under the execution law and
    draws one outcome per shot. Slow — for validation at small B. Supported: RAW, RO, ZNE-L, ZNE-R, PEC."""
    pr = PORTFOLIO[key]
    res_s, res_t = prep.supplied[key], prep.truth[key]
    alloc = allocate(pr, res_s, int(B))
    if not alloc.feasible:
        raise ValueError("infeasible at B = %d: %s" % (B, alloc.reason))
    ctx = prep.ctx_truth[key]
    o = ctx.obs.o
    out = np.empty(int(R))
    if pr.estimator == "raw":
        p = ctx.run().probs_meas
        for r in range(int(R)):
            oe = ctx.o_eff_theta(_theta_direct(ctx, alloc.s, rng)) if pr.readout != "none" else o
            out[r] = _shots(p, alloc.N, rng) @ oe / alloc.N
        return out
    if pr.estimator == "zne":
        d = res_t.sampler.direct
        if d["rule"] not in ("linear", "richardson"):
            raise NotImplementedError("direct ZNE validation covers the linear and Richardson rules")
        ps = [r_.probs_meas for r_ in d["runs"]]
        lams, P = np.asarray(d["lams"]), d["P"]
        deg = 1 if d["rule"] == "linear" else len(lams) - 1
        for r in range(int(R)):
            oe = ctx.o_eff_theta(_theta_direct(ctx, alloc.s, rng))
            E = np.array([_shots(p_, nj, rng) @ oe / nj for p_, nj in zip(ps, alloc.split)]) @ P
            out[r] = float(np.polyval(np.polyfit(lams, E, deg), 0.0))
        return out
    if pr.estimator == "pec":
        sample_instances = res_t.sampler.sample_instances
        for r in range(int(R)):
            oe = ctx.o_eff_theta(_theta_direct(ctx, alloc.s, rng))
            keys = []
            inst = sample_instances(int(alloc.N), int(rng.integers(2 ** 62)), keys)
            acc = 0.0
            for (w_sign, rr) in inst:
                x = int(rng.choice(len(rr.probs_meas), p=np.clip(rr.probs_meas, 0, None) /
                                   np.clip(rr.probs_meas, 0, None).sum()))
                acc += w_sign * oe[x]
            out[r] = acc / int(alloc.N)
        return out
    raise NotImplementedError("direct execution is implemented for RAW, RO, ZNE-L, ZNE-R and PEC")


def compare_laws(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sample comparison of emulated and direct trial estimates: difference of means with its z score and the
    variance ratio with an F-test p-value (normal theory)."""
    from scipy import stats as SS
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = math.sqrt(va / len(a) + vb / len(b))
    z = (ma - mb) / se if se > 0 else 0.0
    f = va / vb if vb > 0 else math.inf
    p_f = 2 * min(SS.f.cdf(f, len(a) - 1, len(b) - 1), SS.f.sf(f, len(a) - 1, len(b) - 1)) if math.isfinite(f) \
        else 0.0
    return dict(n_emulated=int(len(a)), n_direct=int(len(b)), mean_emulated=ma, mean_direct=mb, z_mean=float(z),
                var_emulated=va, var_direct=vb, var_ratio=float(f), p_var=float(p_f))
