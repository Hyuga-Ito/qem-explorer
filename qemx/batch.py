r"""Batch runs: many algorithms × noise models × noise scales × simulators × mitigation methods in one go.

Every *scenario* (algorithm, noise, scale λ, simulator) is one ``compare()`` call that runs all chosen methods on
the same compiled circuit, exactly as the Playground's comparison does. Scenarios run one after the other; a
scenario that fails (too large, out of memory, over the time limit, a method that does not apply) becomes rows
with a status message and the batch carries on.

A *sweep* (``qemx/sweep.py``) adds variants of each algorithm: register inputs such as ``ramp:1``, measurement
bases, template parameters and noise parameters, every combination as its own scenario.

Rows carry the usual metrics (estimate, bias, sd, RMSE, γ, circuits, QPU s) plus the per-shot variance ``v``, so
RMSE can be re-evaluated at any total shot budget N (qemx/budget.py): RMSE(N) = sqrt(b² + v/N_t + w/s + u).
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass

import numpy as np

from . import compile as CP
from . import device
from . import sweep as SW
from . import templates as T
from .noise import PRESETS
from .recipes import compare, family, preset, summary

CURRENT = "current"
DEVICE_NOISE = "device calibration"


@dataclass(frozen=True)
class Scenario:
    source: str               # "current" or a template key
    noise: str                # a noise preset name, or DEVICE_NOISE for a fake device
    scale: float              # noise scale λ (1 = the preset as defined)
    backend: str              # numpy | aer | a fake device name
    variant: tuple = ()       # swept values: ((name, value), …)

    def label(self, names: dict) -> str:
        sim = device.fake_class(self.backend) if device.is_device(self.backend) else self.backend
        nz = self.noise if self.noise == DEVICE_NOISE else "%s ×%g" % (self.noise, self.scale)
        return "%s · %s · %s" % (algorithm_label(names.get(self.source, self.source), self.variant), nz, sim)


def algorithm_label(name: str, variant) -> str:
    return name + (" · " + SW.label(variant) if variant else "")


def algorithm_names(current_name: str = "current circuit") -> dict:
    out = {CURRENT: "Playground: %s" % current_name}
    out.update({k: v[0] for k, v in T.REGISTRY.items()})
    return out


def plan(sources, noises, scales, backends, axes=()) -> list:
    """Every combination; a fake device brings its own noise, so it runs once per algorithm (and swept variant).
    Each algorithm gets the swept quantities that apply to it (sweep.applicable)."""
    out = []
    for a in sources:
        for b in backends:
            dev = device.is_device(b)
            for v in SW.variants(SW.applicable(axes, a, dev)):
                if dev:
                    out.append(Scenario(a, DEVICE_NOISE, 1.0, b, v))
                    continue
                for nz in noises:
                    for s in scales:
                        out.append(Scenario(a, nz, float(s), b, v))
    return out


def _algorithm(source: str, current: dict | None, cache: dict, variant=(), kinds=None) -> dict:
    params, inp, basis, _ = SW.split(variant, kinds or {})
    key = (source, tuple(sorted(params.items())), inp, basis)
    if key in cache:
        return cache[key]
    if source == CURRENT:
        if current is None:
            raise ValueError("no current circuit")
        a = dict(current)
    else:
        tpl = T.build(source, T.clamp_params(source, dict(T.defaults(source), **params)))
        n = tpl.spec.n
        a = dict(spec=tpl.spec, inputs=["0"] * n, basis="Z" * n, observable=tpl.observable, symmetry=tpl.symmetry,
                 relabel=tpl.relabel, name=tpl.spec.name)
    n = a["spec"].n
    if inp is not None:
        a["inputs"] = SW.input_tokens(inp, n)
    if basis is not None:
        a["basis"] = SW.basis_string(basis, n)
    cache[key] = a
    return a


def _opts(opts: CP.CompileOptions, backend: str) -> CP.CompileOptions:
    if device.is_device(backend):
        return CP.CompileOptions(mode="isa", device=backend, opt_level=opts.opt_level, seed=opts.seed,
                                 layout=opts.layout if opts.layout in ("sabre", "dense", "trivial") else "sabre")
    return dataclasses.replace(opts, device=None, initial_layout=None)


def _recipes_for(recipes, symmetry) -> list:
    """Preset names become recipes with the algorithm's own symmetry (symmetry verification checks the parity that
    this algorithm conserves); Recipe objects (custom contenders) are used as built."""
    return [preset(r, symmetry) if isinstance(r, str) else r for r in recipes]


def run(scenarios, recipes, *, current: dict | None, opts: CP.CompileOptions, shots: int, t_shot_us: float,
        time_limit: float | None = None, progress=None, names: dict | None = None, axes=()) -> list:
    """Rows for every (scenario, method). ``recipes``: preset names and/or Recipe objects. ``axes``: the sweep the
    scenarios' variants come from. ``progress(done, total, label)`` is called after each scenario."""
    from .resources import RunTimeout, deadline
    names = names or algorithm_names()
    kinds = {a.name: a.kind for a in axes}
    algos, compiled, rows = {}, {}, []
    for i, sc in enumerate(scenarios):
        name = names.get(sc.source, sc.source)
        base = dict(algorithm=algorithm_label(name, sc.variant), noise=sc.noise, λ=sc.scale,
                    simulator=device.fake_class(sc.backend) if device.is_device(sc.backend) else sc.backend,
                    algorithm_base=name, sweep=SW.label(sc.variant), **{k: v for k, v in sc.variant})
        t0 = time.time()
        try:
            with deadline(time_limit):
                a = _algorithm(sc.source, current, algos, sc.variant, kinds)
                o = _opts(opts, sc.backend)
                params, inp, basis, overrides = SW.split(sc.variant, kinds)
                key = (sc.source, tuple(sorted(params.items())), inp, basis, dataclasses.astuple(o))
                if key not in compiled:
                    compiled[key] = CP.build(a["spec"], a["inputs"], a["basis"], o, a["relabel"])
                comp = compiled[key]
                noise = PRESETS[sc.noise].scaled(sc.scale) if sc.noise in PRESETS else PRESETS["Depolarizing"]
                if overrides:
                    noise = noise.replace(**overrides)
                ideal, outs, _ = compare(comp, noise, a["observable"], _recipes_for(recipes, a["symmetry"]),
                                         sc.backend, int(shots), a["symmetry"],
                                         check_budget=False)       # rows are re-costed at any budget later
            secs = time.time() - t0
            for out in outs:
                row = dict(base, method=out.title, family=family(out.recipe), ideal=ideal,
                           observable=a["observable"].label, qubits=a["spec"].n,
                           **{"2q gates": comp.implemented.count("2q")})
                if out.result is None:
                    row.update(status="failed: %s" % (out.error or "unknown")[:160], estimate=math.nan,
                               bias=math.nan, sd=math.nan, RMSE=math.nan, v=math.nan)
                else:
                    s = summary(out, ideal, int(shots), t_shot_us)
                    s.pop("method", None)
                    row.update(s)
                row["seconds"] = secs
                rows.append(row)
        except (MemoryError, RunTimeout, Exception) as e:            # noqa: BLE001 - one failure never stops a batch
            why = ("out of memory" if isinstance(e, MemoryError) else "time limit" if isinstance(e, RunTimeout)
                   else "%s: %s" % (type(e).__name__, str(e)[:160]))
            from .export import recipe_labels                    # the labels compare() would give (#k for repeats)
            recs = _recipes_for(recipes, None)
            for r, lab in zip(recs, recipe_labels(recs)):
                rows.append(dict(base, method=lab, family=family(r), ideal=math.nan, estimate=math.nan,
                                 bias=math.nan, sd=math.nan, RMSE=math.nan, v=math.nan, status="failed: " + why,
                                 seconds=time.time() - t0))
        if progress is not None:
            progress(i + 1, len(scenarios), sc.label(names))
    return rows


def rmse_at(rows, N: int) -> np.ndarray:
    """RMSE of every row at a total budget of N shots (target, training and calibration together)."""
    from . import budget as BG
    out = []
    for r in rows:
        b = r.get("bias", math.nan)
        if not np.isfinite(b) or not np.isfinite(r.get("v", math.nan)):
            out.append(math.nan)
            continue
        out.append(BG.rmse(float(b), BG.CostModel.of_row(r), max(int(N), 1)))
    return np.array(out, float)


def scenario_key(r: dict) -> str:
    nz = r["noise"] if r["noise"] == DEVICE_NOISE else "%s ×%g" % (r["noise"], r["λ"])
    return "%s · %s · %s" % (r["algorithm"], nz, r["simulator"])


def winners(rows, N: int, metric: str = "RMSE") -> list:
    """Best and runner-up method per scenario (lowest RMSE at N, or lowest |bias|)."""
    vals = rmse_at(rows, N) if metric == "RMSE" else np.abs(np.array([r.get("bias", math.nan) for r in rows], float))
    by = {}
    for r, v in zip(rows, vals):
        if np.isfinite(v):
            by.setdefault(scenario_key(r), []).append((float(v), r))
    out = []
    if not by:
        return out
    for k, lst in by.items():
        lst.sort(key=lambda t: t[0])
        raw = next((v for v, r in lst if r["method"] == "Unmitigated"), math.nan)
        out.append({"scenario": k, "best method": lst[0][1]["method"], metric: lst[0][0],
                    "runner-up": lst[1][1]["method"] if len(lst) > 1 else "–",
                    "gain vs unmitigated": raw / lst[0][0] if np.isfinite(raw) and lst[0][0] > 0 else math.nan})
    return out
