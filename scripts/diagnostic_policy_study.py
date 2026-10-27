"""Supporting study (offline simulation): does a cheap discrepancy diagnostic, paid for from the same shot budget,
make a model-based choice of QEM protocol safer? (qemx.decision)

Design (frozen in this file before any outcome was looked at):

* Model: a supplied noise model (depolarizing gates + readout flips). Candidates: unmitigated (RAW), readout
  correction only (RO), linear and Richardson ZNE with readout correction, and PEC whose inverse uses the model's
  supplied rates (the only candidate that depends on the model being right).
* Truths: the model itself (null) and five unmodeled effects. Exploration uses the null and "gate error × 1.6";
  the held-out evaluation uses coherent ZZ, readout × 2, amplitude damping and ZZ crosstalk (mismatch types never
  seen when the diagnostic budget was chosen) on other base circuits (width 4 and an extra family).
* Diagnostic: D shots of the unmitigated target circuit, TVD against the model's prediction, level α = 0.01 with
  the matched-shot null; D ∈ {256, 1024, 4096} chosen on the exploration cases only. Fallback on a flag: RO.
* Budget: B = 20 000 total shots per estimate; the conditioned policy estimates with B − D.
* Sensitivity (added after the primary results were seen, and labeled post hoc): the same analysis with the
  model-free fallback ZNE-L (linear ZNE with readout correction) instead of RO.

Every number is a simulation (EMULATION), not hardware. Run:

    python scripts/diagnostic_policy_study.py --out reports/diagnostic_policy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qemx import compile as CP          # noqa: E402
from qemx import decision as DC          # noqa: E402
from qemx import insight as IN           # noqa: E402
from qemx import templates as T          # noqa: E402
from qemx.methods import Recipe          # noqa: E402
from qemx.noise import NoiseSpec         # noqa: E402
from qemx.recipes import compare, summary  # noqa: E402

B = 20_000
DS = (256, 1024, 4096)
ALPHA = 0.01
FALLBACK = "RO"                 # pre-declared
FALLBACK_POST_HOC = "ZNE-L"     # sensitivity analysis, chosen after the primary result was seen
MODEL = NoiseSpec(p1=0.002, p2=0.015, ro01=0.02, ro10=0.04)
SCENARIOS = {
    "null": (MODEL, False, "both"),
    "gate error x1.6": (MODEL.replace(p1=0.0032, p2=0.024), True, "explore"),
    "coherent ZZ 0.08 rad": (MODEL.replace(zz_phase=0.08), True, "held-out"),
    "readout x2": (MODEL.replace(ro01=0.04, ro10=0.08), True, "held-out"),
    "amplitude damping 0.006": (MODEL.replace(amp_damp=0.006), True, "held-out"),
    "ZZ crosstalk 0.06 rad": (MODEL.replace(xtalk_zz=0.06), True, "held-out"),
}
CIRCUITS = [  # (label, template, params, split)
    ("ghz-3", "ghz", {"n": 3}, "explore"),
    ("qft-3", "qft", {"n": 3, "s": 5}, "explore"),
    ("ising-3", "ising", {"n": 3, "steps": 2}, "explore"),
    ("bv-3", "bv", {"n": 3, "s": 5}, "explore"),
    ("ghz-4", "ghz", {"n": 4}, "held-out"),
    ("qft-4", "qft", {"n": 4, "s": 9}, "held-out"),
    ("ising-4", "ising", {"n": 4, "steps": 2}, "held-out"),
    ("bv-4", "bv", {"n": 4, "s": 11}, "held-out"),
    ("qaoa-4", "qaoa", {}, "held-out"),
]


def recipes():
    return {"RAW": Recipe.make("raw"), "RO": Recipe.make("raw", readout="tensored"),
            "ZNE-L": Recipe.make("zne", scales=(1.0, 3.0), rule="linear", readout="tensored"),
            "ZNE-R": Recipe.make("zne", scales=(1.0, 3.0, 5.0), rule="richardson", readout="tensored"),
            "PEC": Recipe.make("pec", model="depolarizing", p1=MODEL.p1, p2=MODEL.p2, readout="tensored")}


def rmse_table(comp, obs, noise, budgets, seed):
    keys = list(recipes())
    ideal, outs, runner = compare(comp, noise, obs, list(recipes().values()), "numpy", shots=B, seed=seed)
    tab = {}
    for b in budgets:
        tab[b] = {}
        for k, o in zip(keys, outs):
            if o.result is None:
                tab[b][k] = math.inf
                continue
            r = summary(o, ideal, int(b), 1.0)
            tab[b][k] = float(r["RMSE"]) if math.isfinite(r["RMSE"]) else math.inf
    p = np.asarray(IN.distributions(comp, noise, [], runner, "numpy")["series"]["noisy (unmitigated)"], float)
    return tab, p, ideal


def main(argv=None) -> int:
    from qemx.textio import utf8_stdio
    utf8_stdio()                       # printed text is UTF-8 on every platform
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    budgets = [B] + [B - D for D in DS]
    cases, notes = [], []
    for k, (label, key, params, split) in enumerate(CIRCUITS):
        try:
            tpl = T.build(key, params)
        except Exception as e:                           # noqa: BLE001 - recorded, never silently dropped
            notes.append("%s: template failed (%s: %s)" % (label, type(e).__name__, e))
            continue
        comp = CP.build(tpl.spec, opts=CP.CompileOptions())
        model_tab, p_model, _ = rmse_table(comp, tpl.observable, MODEL, budgets, a.seed + k)
        for j, (sname, (truth, mismatch, where)) in enumerate(SCENARIOS.items()):
            if where not in ("both", split):
                continue
            true_tab, p_truth, _ = rmse_table(comp, tpl.observable, truth, budgets, a.seed + k)
            pf = {D: DC.flag_probability(p_model, p_truth, D, ALPHA, rng=[a.seed, k, j, D]) for D in DS}
            cases.append((DC.Case(label, label, key, sname, mismatch, split, true_tab, model_tab,
                                  {D: v["p_flag"] for D, v in pf.items()}), pf))
    analyses = {}
    for fb, label in ((FALLBACK, "primary (pre-declared fallback)"), (FALLBACK_POST_HOC, "post hoc sensitivity")):
        rows = [DC.case_outcomes(c, B, D, list(recipes()), fb) for c, _ in cases for D in DS]
        explore = [r for r in rows if r["split"] == "explore"]
        D_star = DC.choose_D(explore, DS)
        held = [r for r in rows if r["split"] == "held-out" and r["D"] == D_star]
        analyses[fb] = dict(label=label, fallback=fb, chosen_D=D_star, rows=rows, held=held,
                            explore={D: DC.summarize([r for r in explore if r["D"] == D]) for D in DS},
                            held_out=DC.summarize(held, by=("mismatch",)),
                            held_out_by_scenario=DC.summarize(held, by=("scenario",)))
    prim = analyses[FALLBACK]
    rows, held, D_star = prim["rows"], prim["held"], prim["chosen_D"]
    flags = [dict(circuit=c.circuit, scenario=c.scenario, split=c.split, D=D, **v) for c, pf in cases
             for D, v in pf.items()]
    import pandas as pd
    for fb, an in analyses.items():
        pd.DataFrame(an["rows"]).assign(fallback=fb).to_csv(os.path.join(a.out, "cases_%s.csv" % fb), index=False, encoding="utf-8")
    pd.DataFrame(flags).to_csv(os.path.join(a.out, "flag_probabilities.csv"), index=False, encoding="utf-8")
    res = dict(label="EMULATION (simulation only)", budget=B, diagnostic_shots=list(DS), alpha=ALPHA,
               model=MODEL.__dict__, scenarios=list(SCENARIOS), circuits=[c[0] for c in CIRCUITS], notes=notes,
               seconds=time.time() - t0, seed=a.seed,
               analyses={fb: {k: v for k, v in an.items() if k not in ("rows", "held")} for fb, an in analyses.items()})
    with open(os.path.join(a.out, "diagnostic_policy.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    _figure(analyses, a.out)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(_report(res))
    summ_held = {fb: an["held_out"] for fb, an in analyses.items()}
    print(json.dumps(dict(chosen_D={fb: an["chosen_D"] for fb, an in analyses.items()}, held_out=summ_held,
                          seconds=round(time.time() - t0, 1)), indent=1, default=float))
    return 0


def _fmt(x, p=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else "%.*g" % (p, x)


def _figure(analyses, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.4, 1]))
    colors = {FALLBACK: "#0072B2", FALLBACK_POST_HOC: "#009E73"}
    df0 = pd.DataFrame(analyses[FALLBACK]["held"])
    scen = list(dict.fromkeys(df0["scenario"]))
    y = np.arange(len(scen))
    ext = [0.0, 0.0]
    for off, (fb, an) in zip((-0.15, 0.15), analyses.items()):
        df = pd.DataFrame(an["held"])
        for k, s_ in enumerate(scen):
            g = df[df["scenario"] == s_]
            d = (g["regret_diag"] - g["regret_model"]).to_numpy(float)
            lo, hi = DC._group_ci(g["group"].to_numpy(), d)
            ext = [min(ext[0], lo if np.isfinite(lo) else 0.0), max(ext[1], hi if np.isfinite(hi) else 0.0)]
            ax[0].plot([lo, hi], [k + off] * 2, color=colors[fb], linewidth=2)
            ax[0].plot(np.mean(d), k + off, "o", color=colors[fb],
                       label=("fallback %s (%s, D = %d)" % (fb, "pre-declared" if fb == FALLBACK else "post hoc",
                                                              an["chosen_D"])) if k == 0 else None)
    for k, s_ in enumerate(scen):
        g = df0[df0["scenario"] == s_]
        ax[1].barh(k, g["p_flag"].mean(), color="#D55E00" if g["mismatch"].iloc[0] else "#7A7A7A")
    ax[0].axvline(0, color="#333333", linewidth=0.8)
    ax[0].set_yticks(y, scen)
    ax[0].set_xscale("symlog", linthresh=0.1)
    ax[0].set_xlim(ext[0] * 1.6 - 0.02, ext[1] * 1.6 + 0.02)
    ax[0].set_xlabel("mean regret: diagnostic policy − model policy\n(right of 0: the diagnostic costs more than it saves)")
    ax[0].set_title("a  Effect on the decision (held out)")
    h, l_ = ax[0].get_legend_handles_labels()
    ax[1].set_yticks(y, [""] * len(scen))
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("P(diagnostic flags), D = %d" % analyses[FALLBACK]["chosen_D"])
    ax[1].set_title("b  Detection (gray: null)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(h, l_, loc="lower center", ncol=2, frameon=False, fontsize=7)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out, "fig_diagnostic_policy.%s" % e), dpi=200 if e == "png" else None)


def _report(res) -> str:
    L = ["# Diagnostic-conditioned QEM decisions — supporting study (EMULATION)", "",
         "Simulation only: no hardware data. Budget B = %s shots per estimate; a diagnostic of D shots of the "
         "unmitigated target circuit (D chosen on the exploration cases from %s), TVD check at α = %g; the "
         "conditioned policy estimates with B − D. Regret = RMSE / RMSE of the best protocol under the truth at B − 1; "
         "harmful = worse than unmitigated by more than 10 %%; expectations over the flag." % (
             f"{res['budget']:,}", ", ".join(map(str, res["diagnostic_shots"])), res["alpha"]), "",
         "![](fig_diagnostic_policy.png)", ""]
    for fb, an in res["analyses"].items():
        L += ["## Fallback %s — %s (D = %s)" % (fb, an["label"], an["chosen_D"]), "",
              "### Held-out circuits and mismatch types", "",
              "| scenario | cases (circuits) | P(flag) | model policy: regret / harmful | diagnostic policy: regret / "
              "harmful | fallback always: regret / harmful | diagnostic − model regret [95 % CI over circuits] |",
              "|---|---|---|---|---|---|---|"]
        for s_ in an["held_out_by_scenario"]:
            L.append("| %s | %d (%d) | %s | %s / %s | %s / %s | %s / %s | %s [%s, %s] |" % (
                s_["scenario"], s_["cases"], s_["circuits"], _fmt(s_["p_flag"], 2), _fmt(s_["regret_model"]),
                _fmt(s_["harm_model"], 2), _fmt(s_["regret_diag"]), _fmt(s_["harm_diag"], 2),
                _fmt(s_["regret_fallback"]), _fmt(s_["harm_fallback"], 2), _fmt(s_["diff_regret"]), _fmt(s_["diff_lo"]),
                _fmt(s_["diff_hi"])))
        L += ["", "### Exploration cases (every D)", "", "| D | mismatch | cases | P(flag) | model regret | "
              "diagnostic regret | diagnostic − model [95 % CI] |", "|---|---|---|---|---|---|---|"]
        for D, ss in an["explore"].items():
            for s_ in ss:
                L.append("| %s | %s | %d | %s | %s | %s | %s [%s, %s] |" % (
                    D, s_["mismatch"], s_["cases"], _fmt(s_["p_flag"], 2), _fmt(s_["regret_model"]),
                    _fmt(s_["regret_diag"]), _fmt(s_["diff_regret"]), _fmt(s_["diff_lo"]), _fmt(s_["diff_hi"])))
        L.append("")
    L += ["## Reading", "",
          "* Under the null the diagnostic can only cost: it spends D shots and flags at about the size α.",
          "* It can help only where a mismatch both makes the model's choice worse than the fallback *and* is "
          "visible in the unmitigated output distribution at D shots. A mismatch that the check does not see, or that "
          "does not change the decision, leaves only the cost — and a flag that switches to a worse fallback adds "
          "loss.",
          "* Decision relevance: the unmitigated-distribution check is most sensitive to readout errors, which the "
          "protocols' own measured calibration already corrects, and least sensitive to the gate-rate errors that "
          "bias the model-dependent PEC inverse. A diagnostic aimed at the inputs of the decision (for example the "
          "gate-error rate the PEC inverse assumes, estimated from readout-corrected mirror circuits) is the natural "
          "next design; it is a proposal here, not a tested result.",
          "* The second analysis changed the fallback after the first was seen. It is a sensitivity analysis, not a "
          "confirmed result; a confirmation needs new circuits and mismatch types.",
          "* The circuits, the mismatch strengths and the single statistic are a small designed set: these numbers "
          "are evidence about this design, not a general property of discrepancy diagnostics.", ""]
    if res["notes"]:
        L += ["Notes: " + "; ".join(res["notes"]), ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
