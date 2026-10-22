r"""The hardware transfer test of the main study: repeated, interleaved blocks of complete QEM protocols (1.5.1).

Question. Does the simulated ranking of complete mitigation protocols at a fixed total shot budget per estimate hold
on a device, for a few circuits whose predicted regimes are clearly separated?

Design (``DESIGN``). Each **block** runs, back to back, every protocol on every circuit once — a complete estimate
each, with its own calibration circuits, its own PEC samples and its own twirl/mask draws (seeded by block and
circuit). K blocks give K independent repetitions per (circuit, protocol); contrasts are paired *within* blocks, so
slow drift affects both sides alike. The protocols are the hardware planner's **effective** protocols (requested and
effective parameters recorded; PEC's sampled circuits are capped, which raises its shots per sampled circuit), and
the predictions are simulations of exactly those effective protocols on the device model.

Endpoint. The signed error e = estimate − exact ideal value of one complete estimate (the main study's scalar
endpoint). Per protocol: MSE = mean(e²) over blocks (a bootstrap over blocks gives its interval). Per primary contrast
(the predicted best protocol against readout correction only): d_b = e²_worse,b − e²_better,b, one-sided paired
test of mean(d) > 0, Holm-adjusted over the primary contrasts. Each protocol's mean error is also compared with its
predicted bias (z with the across-block SE): this is the per-protocol discrepancy check on the same circuits.

Power. Under the prediction (one estimate ≈ N(b, σ²)), Var(e²) = 2σ⁴ + 4b²σ²; the one-sided paired z test at 0.05
has power Φ(√K·|E d|/SD d − 1.645). K is the smallest number of blocks giving ≥ 0.8 for every primary contrast,
before usage is checked against the ledger.

Usage. One Sampler job per (circuit, shot count) of a block (``hardware.submit`` splits further by the circuit
limit); each job is estimated with ``qemx.campaign.estimate_usage`` and reserved by the ledger at the larger of its
high estimate and its per-job ``max_execution_time``. The plan states what it reserves; the ledger enforces it.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

SCHEMA = "qemx.transfer/1"
DESIGN = dict(
    version="transfer-v1",
    device="fake_fez", target_backend="ibm_fez",
    budget=4000, pec_max_circuits=64, power_target=0.8, alpha=0.05, max_blocks=40,
    compile=dict(mode="isa", opt_level=1, seed=11, layout="sabre"),
    circuits=[
        dict(name="qft4", role="primary", circuit=dict(template="qft", params=dict(n=4, s=5, mode="swap",
                                                                                   prep="fourier"))),
        dict(name="bv4", role="primary", circuit=dict(template="bv", params=dict(n=4, s=11))),
    ],
    # not in the default design: a predicted near-tie (Ising-4, 2 Trotter steps: RO 0.0154 vs ZNE-L 0.0171 at this
    # budget) would need about 30 blocks for 0.8 power; it would cost a third of every block for a power of ~0.1.
    methods=[
        dict(id="RAW", estimator="raw"),
        dict(id="RO", estimator="raw", readout="tensored"),
        dict(id="ZNE-L", estimator="zne", readout="tensored", params=dict(rule="linear", scales=[1.0, 3.0])),
        dict(id="PEC", estimator="pec", readout="tensored", params=dict(construction="inverse", shots_per_circuit=1)),
    ],
    reference="RO",
)


def _doc(design: dict, c: dict, backend: str) -> dict:
    return dict(schema="qemx.experiment/1", name="transfer-" + c["name"], circuit=c["circuit"], backend=backend,
                compile=dict(design["compile"]),
                methods=[{k: v for k, v in m.items() if k != "id"} for m in design["methods"]],
                budget=dict(shots=int(design["budget"]), t_shot_us=570), run=dict(seed=0, repeats=200))


def resolve(design: dict = DESIGN, backend: str | None = None) -> list:
    """[(circuit entry, qemx.experiment.Resolved)] compiled for ``backend`` (default: the design's device model)."""
    from . import experiment as EX
    return [(c, EX.resolve(_doc(design, c, backend or design["device"]))) for c in design["circuits"]]


def resolve_on(design: dict, backend) -> list:
    """[(circuit entry, namespace(compiled, observable, recipes, symmetry))] compiled for a backend *object* — the
    target device on the day of the run, with its current calibration (layout on its good qubits)."""
    import types
    from . import device as D
    from .compile import Compiled, with_io
    from .methods import Recipe
    cp = design["compile"]
    out = []
    for c in design["circuits"]:
        from .io_semantics import template_inputs                 # the same input model as qemx.experiment
        tokens, body, tpl = template_inputs(c["circuit"]["template"], c["circuit"].get("params") or {})
        logical = with_io(body, tokens)
        spec, mm, phys, info = D.compile_for_target(logical, backend, D.good_target_for(backend), backend.name,
                                                    opt_level=int(cp.get("opt_level", 1)), seed=int(cp.get("seed", 11)))
        recipes = [Recipe.make(m["estimator"], twirl=m.get("twirl", "none"), dd=m.get("dd", "none"),
                               readout=m.get("readout", "none"), **(m.get("params") or {})) for m in design["methods"]]
        out.append((c, types.SimpleNamespace(compiled=Compiled(logical, spec, mm, info), observable=tpl.observable,
                                             recipes=recipes, symmetry=tpl.symmetry)))
    return out


def calibration_hash(dm) -> str | None:
    """A digest of the device model (gate errors, durations, T1/T2, readout) a plan was made with."""
    if dm is None:
        return None
    import dataclasses
    import hashlib
    blob = dict(errors=sorted((k[0], list(k[1]), v) for k, v in dm.errors.items()),
                durations=sorted((k[0], list(k[1]), v) for k, v in dm.durations.items()),
                qubits=[dataclasses.asdict(q) for q in dm.qubits], phys=list(dm.phys))
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=float).encode()).hexdigest()[:16]


def predictions_from_plans(design: dict, plans: list, ideals: dict) -> dict:
    """Diagnostics only: the planner's own recording-model prediction of each executed protocol
    (``alloc[i]["predicted"]``: the device's median-rate equivalent noise, no idle relaxation, PEC averaged over its
    sampled instances). It is *not* the model of the design and power analysis; the prediction frozen before a
    hardware run is :func:`predictions_on`."""
    out = {}
    for name, plan in plans:
        rows = {}
        for k, m in enumerate(design["methods"]):
            a = plan.alloc.get(k) or {}
            pr = a.get("predicted") or {}
            mean = pr.get("mean", math.nan)
            bias = float(mean) - float(ideals[name]) if mean is not None and math.isfinite(mean) else math.nan
            sd = float(pr.get("sd", math.nan))
            rows[m["id"]] = dict(bias=bias, sd=sd, rmse=math.hypot(bias, sd) if math.isfinite(bias) and
                                 math.isfinite(sd) else math.nan, effective=plan.labels()[k],
                                 effective_budget=a.get("effective_budget"))
        out[name] = dict(ideal=float(ideals[name]), role=next(c["role"] for c in design["circuits"]
                                                               if c["name"] == name),
                         methods=rows, calibration_hash=calibration_hash(plan.dm))
    return out


def block_seed(block: int, ci: int) -> int:
    return 1000 * (int(block) + 1) + int(ci)


def block_plan(design: dict, r, backend, block: int, ci: int):
    from . import hardware as H
    return H.make_plan(r.compiled, r.observable, r.recipes, int(design["budget"]), r.symmetry,
                       block_seed(block, ci), backend, pec_max_circuits=design["pec_max_circuits"])


def _predict(design: dict, items) -> dict:
    """The design's prediction: each circuit's effective protocols of block 0 simulated on a device model with the
    full device path (``recipes.compare`` on that model: per-gate errors, idle relaxation, asymmetric readout).
    ``items``: [(circuit entry, namespace with compiled, observable, symmetry), plan, simulator name]."""
    from .noise import NoiseSpec
    from .recipes import compare, summary
    out = {}
    for c, r, plan, sim in items:
        eff = plan.effective_recipes()
        ideal, outs, _ = compare(r.compiled, NoiseSpec(), r.observable, eff, sim, int(design["budget"]),
                                 r.symmetry, seed=0)
        prot, labels = plan.protocol_table(), plan.labels()
        rows = {}
        for k, (m, o) in enumerate(zip(design["methods"], outs)):
            s_ = summary(o, ideal, int(design["budget"]), 570.0) if o.result is not None else {}
            al = plan.alloc.get(k) or {}
            rows[m["id"]] = dict(bias=float(s_.get("bias", math.nan)), sd=float(s_.get("sd", math.nan)),
                                 rmse=float(s_.get("RMSE", math.nan)), effective=o.title,
                                 effective_budget=al.get("effective_budget"), budget_note=al.get("budget_note"),
                                 protocol=[x for x in prot if x["contender"] == labels[k]])
        out[c["name"]] = dict(ideal=float(ideal), role=c["role"], methods=rows, simulator=sim,
                              calibration_hash=calibration_hash(plan.dm))
    return out


def predictions(design: dict = DESIGN) -> dict:
    """{circuit: {method: dict(bias, sd, rmse, effective, protocol)}} — simulations of the effective protocols of
    block 0 on the design's fake device model (the same compilation as the plan): the plan and power analysis."""
    from . import device as D
    backend = D.fake_backend(design["device"])
    return _predict(design, [(c, r, block_plan(design, r, backend, 0, ci), design["device"])
                             for ci, (c, r) in enumerate(resolve(design))])


def predictions_on(design: dict, backend, rs: list | None = None) -> dict:
    """The prediction to freeze before a hardware run: :func:`predictions`' model and simulation path, made with the
    target backend's calibration of the day (``device.model_for`` on the qubits each circuit is compiled to,
    registered as a calibration snapshot) and the compilation ``resolve_on`` makes for it — so the hardware errors
    are compared with the same kind of prediction the design and its power analysis used."""
    from . import device as D
    rs = rs if rs is not None else resolve_on(design, backend)
    items = []
    for ci, (c, r) in enumerate(rs):
        plan = block_plan(design, r, backend, 0, ci)
        dm = D.model_for(backend, tuple(r.compiled.info["phys"]))
        items.append((c, r, plan, D.register_model(dm, "%s-%s-%s" % (getattr(backend, "name", "backend"), c["name"],
                                                                     calibration_hash(dm)))))
    return _predict(design, items)


def _var_e2(m: dict) -> float:
    return 2 * m["sd"] ** 4 + 4 * m["bias"] ** 2 * m["sd"] ** 2


def power(dmean: float, dsd: float, K: int) -> float:
    from scipy.stats import norm
    if not (dsd > 0):
        return 1.0 if dmean > 0 else 0.0
    return float(norm.cdf(math.sqrt(K) * dmean / dsd - 1.645))


def contrasts(pred: dict, design: dict = DESIGN) -> list:
    """Primary contrast per circuit: the predicted best mitigated protocol against the reference (readout correction
    only); if the reference is predicted best, it is contrasted with the runner-up. Secondary: best vs runner-up."""
    out = []
    ref = design["reference"]
    for name, c in pred.items():
        ms = {k: v for k, v in c["methods"].items() if k != "RAW" and math.isfinite(v["rmse"])}
        order = sorted(ms, key=lambda k: ms[k]["rmse"])
        if len(order) < 2:                               # nothing to contrast (one mitigated protocol)
            continue
        best = order[0]
        pairs = [(best, ref if best != ref else order[1], "primary" if c["role"] == "primary" else "exploratory")]
        if len(order) > 1 and order[1] != pairs[0][1]:
            pairs.append((best, order[1], "secondary"))
        for b, w, kind in pairs:
            d = ms[w]["rmse"] ** 2 - ms[b]["rmse"] ** 2
            out.append(dict(circuit=name, better=b, worse=w, kind=kind, rmse_better=ms[b]["rmse"],
                            rmse_worse=ms[w]["rmse"], d_mean=d, d_sd=math.sqrt(_var_e2(ms[b]) + _var_e2(ms[w]))))
    return out


def preflight(design: dict = DESIGN, model=None, blocks: int | None = None, pred: dict | None = None) -> dict:
    """Blocks needed for the power target, the usage of one block (jobs = Sampler calls) and the ledger total, with
    a usage-calibration pilot (Stage 1). Nothing is submitted."""
    from . import campaign as CG
    from . import device as D
    model = model or CG.UsageModel()
    pred = pred or predictions(design)
    C = contrasts(pred, design)
    prim = [x for x in C if x["kind"] == "primary"]
    K = blocks or next((k for k in range(2, int(design["max_blocks"]) + 1)
                        if all(power(x["d_mean"], x["d_sd"], k) >= design["power_target"] for x in prim)),
                       int(design["max_blocks"]))
    for x in C:
        x["power"] = power(x["d_mean"], x["d_sd"], K)
    from . import hardware as H
    backend = D.fake_backend(design["device"])
    limit = H.job_circuit_limit(backend)
    jobs = []
    for ci, (c, r) in enumerate(resolve(design)):
        plan = block_plan(design, r, backend, 0, ci)
        # exactly the Sampler calls hardware.submit makes: built pubs grouped by shots, split by the circuit limit
        for _opts, pubs, _group in H.build_jobs(plan, backend, native_twirl=True, native_dd=True,
                                                seed=block_seed(0, ci)):
            for sh, grp in H._by_shots(pubs):
                for lo in range(0, len(grp), limit):
                    part = grp[lo:lo + limit]
                    est = CG.estimate_usage(int(sh) * len(part), len(part), 1, model)
                    jobs.append(dict(circuit=c["name"], shots_per_circuit=int(sh), circuits=len(part),
                                     executions=int(sh) * len(part), high_s=est.high_s, central_s=est.central_s,
                                     low_s=est.low_s, max_execution_time_s=CG.default_max_execution_time(est),
                                     reserved_s=CG.reservation_s(est)))
    block_reserved = sum(j["reserved_s"] for j in jobs)
    pilot = CG.estimate_usage(10_000, 4, 1, model)
    total = CG.reservation_s(pilot) + K * block_reserved
    admissible = 600.0 - 60.0
    return dict(schema=SCHEMA, design=design["version"], blocks=K, contrasts=C, jobs_per_block=jobs,
                block=dict(jobs=len(jobs), executions=sum(j["executions"] for j in jobs),
                           circuits=sum(j["circuits"] for j in jobs), central_s=sum(j["central_s"] for j in jobs),
                           high_s=sum(j["high_s"] for j in jobs), reserved_s=block_reserved),
                stage1=dict(executions=pilot.executions, high_s=pilot.high_s, reserved_s=CG.reservation_s(pilot)),
                total_reserved_s=total, admissible_s=admissible, fits=bool(total <= admissible),
                usage_model=dict(per_shot_ms=1e3 * model.per_shot_s, margin=model.margin,
                                 circuits_per_subjob=model.circuits_per_subjob),
                max_blocks_that_fit=int(max(0, (admissible - CG.reservation_s(pilot)) // block_reserved)))


# ------------------------------------------------------------------ execution (emulated here; hardware via the CLI)
def block_errors(design: dict, plan, freqs: dict, block: int, circuit: str) -> list:
    """Per protocol of one block: the replayed estimate and its signed error against the exact ideal value."""
    from . import hardware as H
    res = H.analyze(plan, freqs, repeats=50, seed=block)
    out = []
    for m, o in zip(design["methods"], res["outs"]):
        est = float(o.result.mean) if o.result is not None else math.nan
        out.append(dict(block=int(block), circuit=circuit, method=m["id"], estimate=est,
                        error=est - float(res["ideal"]), ideal=float(res["ideal"]),
                        status="ok" if o.result is not None else (o.status or "failed")))
    return out


def rehearse(design: dict = DESIGN, blocks: int = 8, progress=None) -> list:
    """The complete block design on the fake device in local testing mode (EMULATION, no account, no QPU): every
    block plans, transpiles, samples and replays exactly as the hardware path would."""
    from . import device as D
    from . import hardware as H
    backend = D.fake_backend(design["device"])
    rows = []
    rs = resolve(design)
    for b in range(int(blocks)):
        for ci, (c, r) in enumerate(rs):
            plan = block_plan(design, r, backend, b, ci)
            freqs = H.run_local(plan, design["device"], seed=block_seed(b, ci))
            rows += block_errors(design, plan, freqs, b, c["name"])
        if progress:
            progress(b + 1, blocks)
    return rows


def analyze(rows: list, pred: dict, design: dict = DESIGN, reps: int = 4000, seed: int = 0) -> dict:
    """Per (circuit, protocol): MSE over blocks with a bootstrap interval, mean error vs predicted bias; per contrast:
    the paired block-level test (Holm over the primary contrasts)."""
    import pandas as pd
    from scipy.stats import t as tdist
    from .discrepancy import adjust_pvalues
    df = pd.DataFrame(rows)
    df = df[df["status"] == "ok"]
    rng = np.random.default_rng(seed)
    per = []
    for (c, m), g in df.groupby(["circuit", "method"]):
        e = g.sort_values("block")["error"].to_numpy(float)
        K = len(e)
        bs = [float(np.mean(rng.choice(e, K) ** 2)) for _ in range(int(reps))] if K > 1 else [math.nan]
        p = pred.get(c, {}).get("methods", {}).get(m, {})
        se = float(np.std(e, ddof=1) / math.sqrt(K)) if K > 1 else math.nan
        z = (float(np.mean(e)) - p.get("bias", math.nan)) / se if se and se > 0 else math.nan
        sd_obs = float(np.std(e, ddof=1)) if K > 1 else math.nan
        sd_pred = p.get("sd", math.nan)
        from scipy.stats import chi2
        ratio_ci = ((math.sqrt((K - 1) / chi2.ppf(0.975, K - 1)) * sd_obs / sd_pred,
                     math.sqrt((K - 1) / chi2.ppf(0.025, K - 1)) * sd_obs / sd_pred) if K > 1 and sd_pred and
                    sd_pred > 0 else (math.nan, math.nan))
        per.append(dict(circuit=c, method=m, blocks=K, mse=float(np.mean(e ** 2)), rmse=float(math.sqrt(np.mean(e ** 2))),
                        sd_obs=sd_obs, sd_ratio=sd_obs / sd_pred if sd_pred and sd_pred > 0 else math.nan,
                        sd_ratio_lo=ratio_ci[0], sd_ratio_hi=ratio_ci[1],
                        mse_lo=float(np.quantile(bs, 0.025)), mse_hi=float(np.quantile(bs, 0.975)),
                        mean_error=float(np.mean(e)), mean_error_se=se, predicted_bias=p.get("bias"),
                        predicted_rmse=p.get("rmse"), predicted_sd=p.get("sd"), bias_z=z))
    C = []
    for x in contrasts(pred, design):
        a = df[(df["circuit"] == x["circuit"]) & (df["method"] == x["worse"])].set_index("block")["error"]
        b = df[(df["circuit"] == x["circuit"]) & (df["method"] == x["better"])].set_index("block")["error"]
        common = sorted(set(a.index) & set(b.index))
        d = np.array([a[k] ** 2 - b[k] ** 2 for k in common], float)
        K = len(d)
        if K > 1 and np.std(d, ddof=1) > 0:
            tstat = float(np.mean(d) / (np.std(d, ddof=1) / math.sqrt(K)))
            pval = float(1 - tdist.cdf(tstat, K - 1))
        else:
            tstat, pval = math.nan, math.nan
        C.append(dict(x, blocks=K, d_obs=float(np.mean(d)) if K else math.nan, t=tstat, p_one_sided=pval,
                      predicted_direction_holds=bool(K and np.mean(d) > 0)))
    # Holm over the whole pre-declared primary family: a contrast without a finite p (too few blocks, no spread)
    # counts as p = 1 and still takes its place in the family, so the others are not adjusted less than declared
    prim = [i for i, x in enumerate(C) if x["kind"] == "primary"]
    if prim:
        adj = adjust_pvalues([C[i]["p_one_sided"] if math.isfinite(C[i]["p_one_sided"]) else 1.0 for i in prim],
                             "holm")
        for i, pa in zip(prim, adj):
            C[i]["p_holm"] = float(pa)
            C[i]["confirmed"] = bool(pa < design["alpha"])
    return dict(schema=SCHEMA, design=design["version"], per_method=per, contrasts=C)


def report_markdown(an: dict, pf: dict | None, label: str) -> str:
    def f(x, p=3):
        return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else "%.*g" % (p, x)
    L = ["# Hardware transfer test — %s" % label, "",
         "Design `%s`: repeated blocks of complete protocols, paired within blocks; endpoint = signed error of one "
         "complete estimate against the exact ideal value." % an["design"], "",
         "## Contrasts", "",
         "| circuit | kind | better (predicted) | worse | predicted RMSE | blocks | mean d = e²_worse − e²_better | "
         "one-sided p | Holm p | confirmed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in an["contrasts"]:
        L.append("| %s | %s | %s | %s | %s vs %s | %d | %s | %s | %s | %s |" % (
            x["circuit"], x["kind"], x["better"], x["worse"], f(x["rmse_better"]), f(x["rmse_worse"]), x["blocks"],
            f(x["d_obs"]), f(x["p_one_sided"], 2), f(x.get("p_holm"), 2),
            {True: "yes", False: "no"}.get(x.get("confirmed"), "—")))
    L += ["", "## Per protocol (discrepancy against the prediction on the same circuits)", "",
          "| circuit | protocol | blocks | RMSE [95 % CI of MSE] | predicted RMSE | mean error ± SE | predicted bias | "
          "z | SD observed / predicted [95 % CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for r in an["per_method"]:
        L.append("| %s | %s | %d | %s [%s, %s] | %s | %s ± %s | %s | %s | %s [%s, %s] |" % (
            r["circuit"], r["method"], r["blocks"], f(r["rmse"]), f(r["mse_lo"]), f(r["mse_hi"]),
            f(r["predicted_rmse"]), f(r["mean_error"]), f(r["mean_error_se"]), f(r["predicted_bias"]), f(r["bias_z"], 2),
            f(r.get("sd_ratio"), 2), f(r.get("sd_ratio_lo"), 2), f(r.get("sd_ratio_hi"), 2)))
    L += ["", "MSE intervals: bootstrap over blocks. z: (mean error − predicted bias) / SE. The SD ratio interval is "
          "the χ² interval of a normal sample's SD."]
    if pf:
        L += ["", "Plan: %d blocks; per block %d jobs, %s executions, %.1f s reserved; total reserved %.0f s of %.0f s "
              "admissible (Stage 1 included)." % (pf["blocks"], pf["block"]["jobs"], f"{pf['block']['executions']:,}",
                                                    pf["block"]["reserved_s"], pf["total_reserved_s"],
                                                    pf["admissible_s"])]
    L += ["", "A confirmed contrast supports the simulated ranking for that circuit at this budget on this device and "
          "time window only. An unconfirmed one is not evidence that the protocols are equivalent.", ""]
    return "\n".join(L)


def save(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
