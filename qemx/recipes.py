"""Ready-made recipes and the comparison runner shared by every page."""
from __future__ import annotations

import dataclasses
import math

from dataclasses import dataclass, field

import numpy as np

from .methods import REGISTRY, Context, MethodResult, Recipe, Runner
from .compile import Compiled
from .noise import NoiseSpec
from .observables import Observable

PRESETS = {
    "Unmitigated": dict(estimator="raw"),
    "ZNE · Richardson": dict(estimator="zne", rule="richardson"),
    "ZNE · linear": dict(estimator="zne", rule="linear"),
    "ZNE · exponential": dict(estimator="zne", rule="exponential"),
    "ZNE · local folding": dict(estimator="zne", folding="local"),
    "PEC · inverse": dict(estimator="pec", construction="inverse"),
    "PEC · compensation": dict(estimator="pec", construction="compensation"),
    "PEC · learned model + RC": dict(estimator="pec", model="learned", twirl="all"),
    "CDR · OLS": dict(estimator="learning", features="cdr", model="ols"),
    "FR · Ridge": dict(estimator="learning", features="fr", model="ridge", alpha=1e-3),
    "FR · Lasso": dict(estimator="learning", features="fr", model="lasso", alpha=1e-4),
    "NIL · ℓ1 ≤ 2 (Chen et al.)": dict(estimator="nil", model="l1", l1_bound=2.0),
    "NIL · Ridge": dict(estimator="nil", model="ridge", alpha=1e-3),
    # budget-scaled for teaching (T=6 training circuits × 4 circuits each × 128 shots = 3,072 training shots, so it
    # runs within the default 4,096-shot budget). Not the configuration of the NIL paper; expect a noisy fit.
    "NIL · small budget": dict(estimator="nil", model="l1", l1_bound=2.0, n_train=6, n_neighbors=3, train_shots=128),
    "Readout · tensored": dict(estimator="raw", readout="tensored"),
    "Readout · M3": dict(estimator="raw", readout="m3"),
    "Readout · TREX": dict(estimator="raw", readout="trex"),
    "Symmetry verification": dict(estimator="symmetry"),
    "Virtual distillation (M=2)": dict(estimator="vd", M=2),
    "Pauli twirling (RC) only": dict(estimator="raw", twirl="all"),
    "Dynamical decoupling (XY4)": dict(estimator="raw", dd="XY4"),
    "ZNE + RC + TREX": dict(estimator="zne", rule="richardson", twirl="2q", readout="trex"),
}
DEFAULT_SET = ["Unmitigated", "ZNE · Richardson", "ZNE · linear", "PEC · inverse", "CDR · OLS", "Readout · TREX"]


def preset(name: str, symmetry=None) -> Recipe:
    d = dict(PRESETS[name])
    est = d.pop("estimator")
    tw, dd, ro = d.pop("twirl", "none"), d.pop("dd", "none"), d.pop("readout", "none")
    if est == "symmetry" and symmetry:
        d.setdefault("qubits", list(symmetry[0]))
        d.setdefault("eigenvalue", symmetry[1])
        if len(symmetry) > 2 and symmetry[2]:          # further checks that must hold too (all Z checks of a code)
            d.setdefault("more", str(symmetry[2]))
    return Recipe.make(est, tw, dd, ro, **d)


def family(recipe: Recipe) -> str:
    if recipe.estimator == "raw" and recipe.readout != "none":
        return "readout"
    if recipe.estimator == "chain":                   # a chain is drawn in its outer method's family
        return "zne" if recipe.p.get("outer") == "zne" else "learning"
    return REGISTRY[recipe.estimator].family


@dataclass
class Outcome:
    recipe: Recipe
    result: MethodResult | None
    error: str = ""
    label: str = ""                                   # set by compare() so that two identical contenders stay apart
    status: str = ""                                  # "" (ran) | INSUFFICIENT_BUDGET (not run) | "error"
    preflight: dict | None = None                     # the budget preflight that decided it (budget.preflight)

    @property
    def title(self) -> str:
        return self.label or self.recipe.title()


def preflight(rc: Recipe, make_ctx, compiled, observable, budget: int) -> dict | None:
    """Cheap budget check before a recipe runs (no simulation, no fit): {feasible, min_budget, training, note} or
    None when the method cannot tell its cost in advance. min_budget is inclusive: the smallest total budget that can
    yield an estimate (training + one shot per calibration circuit + one shot per target circuit)."""
    from . import hamiltonian as HM
    if isinstance(observable, HM.PauliSum):            # several settings: each pays its own training; not preflighted
        return None
    try:
        ctx = make_ctx(compiled, observable)
        mc = REGISTRY[rc.estimator].min_cost(ctx, **rc.p)
    except Exception:                                  # noqa: BLE001 - the run itself reports invalid settings
        return None
    if mc is None:
        return None
    cal = int(ctx.cal_circuits())
    mb = float(mc["training"]) + cal + max(int(mc.get("target_circuits", 1)), 1)
    return dict(feasible=bool(mb <= int(budget)), min_budget=mb, training=int(mc["training"]), calibration_circuits=cal,
                note=mc.get("note", ""))


def compare(compiled: Compiled, noise: NoiseSpec, observable: Observable, recipes: list, backend: str = "numpy",
            shots: int = 4096, symmetry=None, seed: int = 0, check_budget: bool = False) -> tuple:
    """Run every recipe; returns (ideal value, list[Outcome], runner). ``shots`` is the total budget B per estimate.
    With ``check_budget`` (the Playground) a recipe whose fixed cost (training, calibration) alone exceeds B is not run
    at all — no fit, no Monte Carlo — and is returned with ``Outcome.status`` = INSUFFICIENT_BUDGET and its preflight.
    Without it (studies that cost every recipe at many budgets from its cost model) every recipe runs."""
    from .resources import RunTimeout
    from . import device
    runner = Runner(backend, layout=compiled.info.get("phys") if device.is_device(backend) else None)
    if runner.device_model is not None:
        noise = runner.device_model.equivalent_noise()      # the *model* methods may assume (PEC rates, TREX)
    from . import hamiltonian as HM
    out = []
    ideal = None
    for rc in recipes:
        def make_ctx(c, o, rc=rc):
            return Context(c, noise, o, rc, runner, shots=shots, symmetry=symmetry, seed=seed)
        if ideal is None:
            ideal = HM.ideal_value(make_ctx, compiled, observable)
        pf = preflight(rc, make_ctx, compiled, observable, shots) if check_budget else None
        if pf is not None and not pf["feasible"]:
            out.append(Outcome(rc, None, "%s: %s (budget %s shots; %s)" % (
                INSUFFICIENT_BUDGET, budget_needed_text(pf["min_budget"]), f"{int(shots):,}", pf["note"]),
                status=INSUFFICIENT_BUDGET, preflight=pf))
            continue
        try:
            # a Pauli-sum Hamiltonian runs once per measurement setting and is combined; a diagonal one directly
            res = HM.evaluate(make_ctx, lambda ctx, rc=rc: REGISTRY[rc.estimator].run(ctx, **rc.p), compiled,
                              observable, shots)
            out.append(Outcome(rc, res))
        except (RunTimeout, MemoryError, KeyboardInterrupt):
            raise                                    # stop the whole comparison; the UI explains why
        except Exception as e:                       # one failing recipe must not hide the others
            out.append(Outcome(rc, None, "%s: %s" % (type(e).__name__, e)))
    seen = {}
    for o in out:                                    # results, samples and figures are keyed by title
        t = o.recipe.title()
        seen[t] = seen.get(t, 0) + 1
        o.label = t if seen[t] == 1 else "%s (#%d)" % (t, seen[t])
    return ideal, out, runner


HIDDEN = ("family", "v", "fixed_shots", "cal_circuits", "w_cal", "u_train", "n_circuits", "per_circuit",  # cost model
          "target moments", "reference bias")             # budget-dependent moment grid (JSON) of nonlinear rows

# column names of a single measured experiment (hardware replay): one signed error and its absolute value, not a
# statistical bias or RMSE (qemx/stats.py). Rows keep "bias"/"RMSE" as machine keys for ranking code; tables and
# exports show these names for measured rows (measured_view).
MEASURED_LABELS = {"bias": "signed error to reference", "RMSE": "absolute error", "sd": "bootstrap sd"}


def measured_view(df):
    """A comparison table (pandas DataFrame) with measured-experiment columns named for what they are: when every
    row that ran is a single measured experiment, "bias" → "signed error to reference", "RMSE" → "absolute error",
    "sd" → "bootstrap sd"; mixed tables keep the statistical names and gain the two measured columns."""
    if "estimate kind" not in df:
        return df
    ran = df["estimate kind"] != "not run"
    meas = df["estimate kind"] == "measured"
    if meas.any() and bool((meas | ~ran).all()):
        drop = [k for k, v in MEASURED_LABELS.items() if k in df and v in df]      # explicit columns exist
        df = df.drop(columns=drop)
        return df.rename(columns={k: v for k, v in MEASURED_LABELS.items() if k in df})
    return df

INSUFFICIENT_BUDGET = "insufficient_budget"     # status of a protocol that was not run: its minimum cost exceeds B


def budget_needed_text(min_budget: float) -> str:
    """'needs at least 212,997 shots' (min_budget is the smallest total budget that yields an estimate, inclusive)."""
    if not math.isfinite(min_budget):
        return "no estimate at any budget"
    return "needs at least %s shots" % f"{int(math.ceil(min_budget)):,}"


def nonlinear_exact(recipe: Recipe) -> bool:
    """Estimators that are nonlinear in the sampled data (their first-order sd can be optimistic at small budgets)
    and whose samplers draw the actual shots, so the spread of the repeats is the real one: ZNE with the exponential
    rule (a log-fit) and symmetry verification (a ratio with a random number of accepted shots). Virtual distillation
    is a ratio too, but its sampler is Gaussian, so its repeats carry no extra information."""
    if recipe.estimator == "symmetry":
        return True
    return recipe.estimator == "zne" and recipe.p.get("rule") == "exponential"


def summary(o: Outcome, ideal: float, B: int, t_shot_us: float, samples=None, alloc: dict | None = None) -> dict:
    """One row of the comparison table at the total shot budget B, identical for every method (qemx/budget.py).

    ``sd`` is the analytic standard error at that budget, sqrt(v/N_t + w/s + u): v/N_t from the target circuits, w/s
    from a finitely sampled readout calibration and u from the training circuits' shot noise (learning). The learning
    fit is refitted on noisy training data in every repeat; when that spread exceeds the first-order value (a
    non-linear fit), ``sd`` is the spread of the repeats and the excess is booked as training noise u. The same
    fallback applies to the nonlinear estimators with exact samplers (``nonlinear_exact``: exponential ZNE, symmetry
    verification), with the same 25 % threshold; their excess comes from too few target shots and shrinks with the
    budget, so it is *not* booked as a constant: the cost model (v, u) stays analytic and only this budget's ``sd``
    is the repeats'. ``sd (repeats)`` is always the empirical spread (NaN-aware; failed repeats
    are counted, not averaged). A NaN estimate or variance gives an infinite RMSE.

    Since 1.5.1, a result with a ``target_sampler`` (estimators nonlinear in their target data: exponential ZNE,
    symmetry verification) is summarized by its **budget-dependent protocol moments** at this budget's target shots
    (qemx.stats.target_moments, Monte Carlo to the declared tolerance): ``estimate`` is the protocol's expected value,
    ``bias`` = estimate − ideal with its MC error, ``sd`` the protocol's spread plus the calibration (and training)
    terms, ``estimate kind`` "protocol", and the exact-probability fit is the ``exact-fit value``. The columns
    ``fit failures`` (share of experiments whose nonlinear fit failed) and ``failure policy`` say how failures were
    handled; with "abstain" the moments are conditional on success and the status says how often it abstained.
    A measured row (one hardware experiment) also carries ``signed error to reference`` and ``absolute error``
    (the latter equals its ``RMSE`` slot); tables name them so through :func:`measured_view`.

    ``alloc`` (hardware) is the allocation actually run — {N, s, training, calibration, circuits, shots}: the shot
    columns, the circuits and the analytic sd then describe that experiment, and QPU s its shots."""
    from . import budget as BG
    r = o.result
    from .hardware import MEAS_TWIRLS
    acc = BG.account(r, int(B), randomized_target=o.recipe.twirl != "none", alloc=alloc,
                     meas_twirls=MEAS_TWIRLS if o.recipe.readout == "trex" else 1)
    sd_an = acc["sd"]
    emp, failed = float("nan"), 0
    if samples is not None:
        s = np.asarray(samples, float)
        ok = s[np.isfinite(s)]
        failed = int(s.size - ok.size)
        if ok.size >= 10:
            emp = float(np.std(ok, ddof=1))
    fam = family(o.recipe)
    m = BG.CostModel.of(r)
    # 120 repeats estimate a spread to ±6.5 %: only a clear excess (> 25 %) is treated as non-linearity
    excess = np.isfinite(emp) and acc["feasible"] and emp > 1.25 * sd_an
    use_emp = bool(excess and (fam == "learning" or nonlinear_exact(o.recipe)))
    if use_emp:
        sd = emp
        if fam == "learning":
            # the refit on noisy training data is non-linear (FR/NIL: many near-collinear features), so the
            # delta-method floor u can be optimistic; the repeats measure it. Their excess over v/N_t + w/s is
            # training noise, which does not shrink with the budget: it becomes the cost model's constant u.
            m = dataclasses.replace(m, u=max(emp * emp - (acc["var_target"] + acc["var_cal"]), m.u))
    else:
        sd = sd_an
    kind = getattr(r, "estimate_kind", "exact") or "exact"
    mom = getattr(r, "moments", None) or {}
    mc = float(mom.get("mean_mcse", 0.0) or 0.0) if kind == "protocol" else 0.0
    est_row = r.mean
    b = r.mean - ideal
    st_ = status(r)
    flags = list(getattr(r, "flags", ()) or ())
    sd_from = "repeats" if use_emp else "analytic"
    extra = {}
    tm = target_protocol(r, acc) if kind != "measured" else None
    if tm is not None:
        # an estimator nonlinear in its target data (exponential ZNE, symmetry verification): the row describes the
        # complete finite-shot protocol at this budget (Monte Carlo), not the fit to exact probabilities
        kind, est_row, mc = "protocol", tm.mean, tm.mean_mcse
        b = tm.mean - ideal
        sd = math.sqrt(tm.var + acc["var_cal"] + acc["var_train"]) if math.isfinite(tm.var) else math.inf
        sd_from = "protocol MC"
        if "budget-dependent-moments" not in flags:
            flags.append("budget-dependent-moments")
        if tm.conditional:
            flags.append("conditional-on-success")
            st_ = (st_ + "; " if st_ else "") + ("abstains in %.1f %% of experiments: estimate, sd and RMSE are "
                                                  "conditional on success" % (100 * tm.abstention_rate))
        extra = {"fit failures": tm.fit_failure_rate, "failure policy": tm.policy, "MC experiments": tm.repeats}
    if not acc["feasible"]:
        st_ = INSUFFICIENT_BUDGET + ": " + budget_needed_text(acc["min_budget"])
    if kind == "measured":
        # one measured run: e = estimate − ideal is a single signed error to the reference; e² is an unbiased
        # observation of the MSE, and |e| is shown in the "RMSE" slot (named "absolute error" for measured rows,
        # measured_view) — not √(e² + sd²), which would count the shot noise twice
        rmse_ = abs(float(b)) if np.isfinite(b) else math.inf
    else:
        rmse_ = float(np.sqrt(b * b + sd * sd)) if (np.isfinite(sd) and np.isfinite(b)) else math.inf
    ref = getattr(r, "reference_mean", None)
    grid = (r.details or {}).get("target_moments") if getattr(r, "target_sampler", None) is not None else None
    if grid and kind != "measured":
        import json
        extra.update({"target moments": json.dumps(grid, default=float),
                      "reference bias": float(ref) - ideal if ref is not None else math.nan})
    if kind == "measured":
        extra.update({"signed error to reference": b, "absolute error": rmse_})
    return {"method": o.title, "family": fam, "estimate": est_row, "estimate kind": kind,
            "exact-fit value": float(ref) if ref is not None else math.nan, "bias": b, "bias MC ±": mc, "sd": sd,
            "sd from": sd_from, "sd (repeats)": emp, "failed repeats": failed,
            "RMSE": rmse_, "γ": r.gamma, "flags": ", ".join(dict.fromkeys(flags)), **extra,
            "circuits": acc["circuits"], "extra qubits": r.extra_qubits, "accepted": r.accept,
            "target shots": int(round(acc["target_shots"])), "training shots": int(acc["training_shots"]),
            "calibration shots": int(round(alloc["calibration"] if alloc else acc["calibration_shots"])),
            "QPU s": (alloc["shots"] if alloc else int(B)) * t_shot_us * 1e-6, "status": st_,
            "v": m.v, "fixed_shots": m.fixed, "cal_circuits": m.c, "w_cal": m.w, "u_train": m.u,
            "n_circuits": m.circuits, "per_circuit": int(getattr(m, "per_circuit", 1) or 1)}


def target_protocol(r: MethodResult, acc: dict):
    """qemx.stats.TargetMoments of a target-nonlinear result at the target shots of a budget (``acc`` = budget.account),
    or None for estimators whose finite-shot mean does not depend on the target shots. Seeded by the target-shot
    count, so a table is reproducible. Calibration and training variances are added analytically by the caller."""
    ts = getattr(r, "target_sampler", None)
    if ts is None or not acc.get("feasible"):
        return None
    from .stats import target_moments
    N = int(max(round(float(acc["target_shots"])), 1))
    return target_moments(ts, N, np.random.default_rng([0x51A7, N]), getattr(r, "failure_policy", None) or "abstain")


def table_rows(outs, ideal: float, B: int, t_shot_us: float, repeats: int = 120, seed: int = 1,
               allocs: dict | None = None) -> tuple:
    """(samples, rows) for a comparison at total budget B: every method simulated ``repeats`` times at that
    budget (budget.draw) and summarized. ``allocs`` {title: allocation run} for hardware results."""
    from . import budget as BG
    rng = np.random.default_rng(seed)
    samples, rows = {}, []
    allocs = allocs or {}
    for o in outs:
        if o.result is None:
            if o.status == INSUFFICIENT_BUDGET and o.preflight:
                rows.append(unaffordable_row(o, int(B), t_shot_us))      # shown, explained, never run
            continue
        a = allocs.get(o.title)
        if a is None and not BG.account(o.result, int(B))["feasible"]:
            samples[o.title] = None              # budget preflight: an unaffordable protocol is not run at all
        else:
            try:
                samples[o.title] = BG.draw(o.result, int(B), int(repeats), rng, alloc=a)
            except Exception:                                    # noqa: BLE001 - the table still shows the row
                samples[o.title] = None
        rows.append(summary(o, ideal, int(B), t_shot_us, samples.get(o.title), alloc=a))
    return samples, rows


def unaffordable_row(o: Outcome, B: int, t_shot_us: float) -> dict:
    """The table row of a recipe the budget preflight did not run: no estimate, no repeats (not "failed repeats"),
    the training cost and the smallest budget that would run it."""
    pf = o.preflight or {}
    nan = math.nan
    return {"method": o.title, "family": family(o.recipe), "estimate": nan, "estimate kind": "not run",
            "exact-fit value": nan, "bias": nan, "bias MC ±": nan, "sd": math.inf, "sd from": "—",
            "sd (repeats)": nan, "failed repeats": 0, "RMSE": math.inf, "γ": nan, "flags": "",
            "circuits": 0, "extra qubits": 0, "accepted": 1.0, "target shots": 0,
            "training shots": int(pf.get("training", 0)), "calibration shots": 0,
            "QPU s": nan,                                    # not run: it cannot spend this budget
            "status": "%s: %s (%s)" % (INSUFFICIENT_BUDGET, budget_needed_text(pf.get("min_budget", math.inf)),
                                       pf.get("note", "")),
            "v": math.inf, "fixed_shots": float(pf.get("training", 0)), "cal_circuits": int(pf.get("calibration_circuits", 0)),
            "w_cal": 0.0, "u_train": 0.0, "n_circuits": 1}


def status(r: MethodResult) -> str:
    """A short flag for results that should not be read at face value."""
    d = r.details or {}
    if d.get("degenerate"):
        return "not applied: training data degenerate (unmitigated value shown)"
    if d.get("warning"):
        return "weak training signal"
    return ""
