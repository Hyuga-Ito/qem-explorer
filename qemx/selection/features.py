r"""Features of a task for protocol selection, with their information class (1.6.0).

Every feature names its class (``task.INFORMATION``) and its group; a deployed policy may use ``available``
features freely, ``acquired`` ones only with their shots charged, ``simulated`` ones as a declared simulator-only
augmentation (their cost grows exponentially with width), and never ``oracle`` ones. Models are trained on feature
*sets* (``SETS``) so that each research question compares like with like.

Light-cone features (precise definitions; ``qemx.phase.lightcone.calculate_lambda_lc`` on the compiled circuit, the
device's calibration snapshot as the noise model). The *backward structural light cone* of the observable starts at
the physical qubits its support is measured on (through the measurement map) and walks the compiled circuit from
the last operation to the first; a two-qubit gate that touches the current set adds both its qubits. It is
conservative: algebraic cancellations can make the exact Heisenberg support smaller, and crosstalk can make the
relevant region larger.

* ``lc_qubits`` — qubits in the light cone at the start of the circuit;
* ``lc_gates``, ``lc_2q`` — compiled gates (two-qubit gates) inside it;
* ``lc_locations`` — fault locations inside it with a non-zero fault probability (gates and idle windows);
* ``lam_lc`` — Σ of Pauli-twirled fault probabilities over those locations (snapshot model), ``lam_lc_exact`` =
  −Σ ln(1 − p), ``lam_total`` the same over the whole circuit, ``lc_share`` = lam_lc / lam_total.

A sum of error indicators inside a light cone is a feature, not a fidelity or a failure probability (theory
companion §1.5.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class F:
    name: str
    info: str            # available | acquired | simulated | oracle
    group: str           # structure | observable | lightcone | noise | budget | protocol | state | oracle
    definition: str


DICTIONARY = (
    F("n", "available", "structure", "qubits of the logical circuit"),
    F("two_qubit_logical", "available", "structure", "two-qubit gates of the logical circuit as written"),
    F("non_clifford", "available", "structure", "non-Clifford gates of the logical circuit as written"),
    F("two_qubit", "available", "structure", "two-qubit gates of the compiled circuit"),
    F("depth", "available", "structure", "ASAP layers of the compiled circuit (virtual RZ included)"),
    F("pulses_1q", "available", "structure", "non-virtual one-qubit gates of the compiled circuit"),
    F("non_clifford_slots", "available", "structure", "compiled RZ angles that are not multiples of π/2 "
      "(the slots CDR/NIL training replaces)"),
    F("idle_fraction", "available", "structure", "share of (qubit, layer) slots after a qubit's first gate without "
      "a gate"),
    F("duration_us", "available", "structure", "scheduled duration of the compiled circuit (ALAP, µs)"),
    F("obs_weight", "available", "observable", "largest Pauli weight of the observable's Z-string terms"),
    F("obs_terms", "available", "observable", "non-identity terms of the observable"),
    F("obs_support", "available", "observable", "classical bits any term acts on"),
    F("lc_qubits", "available", "lightcone", "qubits in the backward structural light cone (see module text)"),
    F("lc_gates", "available", "lightcone", "compiled gates inside the light cone"),
    F("lc_2q", "available", "lightcone", "compiled two-qubit gates inside the light cone"),
    F("lc_locations", "available", "lightcone", "fault locations with p > 0 inside the light cone"),
    F("lam_lc", "available", "noise", "Σ Pauli-twirled fault probability inside the light cone (snapshot model)"),
    F("lam_lc_exact", "available", "noise", "−Σ ln(1 − p) over the same locations"),
    F("lam_lc_gate", "available", "noise", "lam_lc from gate errors"),
    F("lam_lc_idle", "available", "noise", "lam_lc from idle windows (relaxation, dephasing)"),
    F("lam_total", "available", "noise", "Σ fault probability over every location of the circuit"),
    F("lc_share", "available", "noise", "lam_lc / lam_total"),
    F("lam_ro", "available", "noise", "Σ over the measured support of the mean readout flip probability"),
    F("log10_budget", "available", "budget", "log10 of the total shot budget B"),
    # 1.8.0 noise-composition descriptors (calibration snapshot only; prompt section 4). Not part of the 1.7 sets.
    F("lam_lc_2q", "available", "noise", "lam_lc from the errors of two-qubit gates only (lam_lc_gate − lam_lc_2q is "
      "the one-qubit part)"),
    F("ro_asym", "available", "noise", "mean |P(1|0) − P(0|1)| of the readout of the observable's measured qubits "
      "(readout asymmetry)"),
    F("err2q_spread", "available", "noise", "max / median reported error of the two-qubit gates the compiled "
      "circuit uses (spatial heterogeneity; 1 with fewer than two)"),
    F("duration_t2", "available", "noise", "scheduled duration / median T2 of the used qubits (relaxation exposure "
      "scale)"),
    F("pec_log_gamma", "available", "protocol", "ln γ of PEC's quasi-probability decomposition of the snapshot's "
      "median-rate model (readout γ included): its standard deviation grows as γ"),
    F("zne_eta_l1_linear", "available", "protocol", "Σ|η_j| of linear extrapolation at the achieved fold λ (gate "
      "counts): ZNE-L's noise amplification of the per-scale means"),
    F("zne_eta_l1_richardson", "available", "protocol", "Σ|η_j| of Richardson extrapolation at λ = 1, 3, 5"),
    F("log10_v_raw_sup", "simulated", "protocol", "log10 per-shot variance of the readout-corrected estimator under the "
      "supplied model"),
    F("log10_v_zne_sup", "simulated", "protocol", "log10 per-shot variance of ZNE-R under the supplied model "
      "(optimal split)"),
    F("log10_v_pec_sup", "simulated", "protocol", "log10 per-shot variance of PEC under the supplied model"),
    F("abs_signal_sup", "simulated", "protocol", "|⟨O⟩| of the noisy circuit under the supplied model (raw, no "
      "readout error)"),
    F("entropy_half", "simulated", "state", "half-cut entanglement entropy (bits) of the ideal final state"),
    F("output_entropy_bits", "simulated", "state", "Shannon entropy of the ideal output distribution"),
    F("ideal", "oracle", "oracle", "exact ideal value μ"),
    F("raw_bias_true", "oracle", "oracle", "exact bias of the unmitigated estimator under the execution law"),
)
BY_NAME = {f.name: f for f in DICTIONARY}

SETS = {
    # research question 2: which descriptors are necessary
    "size_budget": ("n", "two_qubit", "depth", "log10_budget"),
    "structure": ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "pulses_1q", "non_clifford_slots",
                  "idle_fraction", "duration_us", "obs_weight", "obs_terms", "obs_support", "log10_budget"),
    "lightcone": ("n", "two_qubit", "depth", "obs_weight", "lc_qubits", "lc_gates", "lc_2q", "lc_locations",
                  "log10_budget"),
    # the 1.7.0 sets, frozen by name (1.8.0 added features must not change them: bundles and analyses refer to them)
    "deployable": ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "pulses_1q", "non_clifford_slots",
                   "idle_fraction", "duration_us", "obs_weight", "obs_terms", "obs_support", "lc_qubits", "lc_gates",
                   "lc_2q", "lc_locations", "lam_lc", "lam_lc_exact", "lam_lc_gate", "lam_lc_idle", "lam_total",
                   "lc_share", "lam_ro", "log10_budget", "pec_log_gamma", "zne_eta_l1_linear",
                   "zne_eta_l1_richardson"),
}
SETS["deployable+simulated"] = SETS["deployable"] + ("log10_v_raw_sup", "log10_v_zne_sup", "log10_v_pec_sup",
                                                     "abs_signal_sup", "entropy_half", "output_entropy_bits")
NOISE18 = ("lam_lc_2q", "ro_asym", "err2q_spread", "duration_t2")
SETS["deployable18"] = SETS["deployable"] + NOISE18                               # 1.8.0
SETS["deployable18+simulated"] = SETS["deployable+simulated"] + NOISE18
GROUPS = sorted({f.group for f in DICTIONARY if f.info != "oracle"})
# Every feature computed from the backward light cone (1.7.0, review of Figure 3): the structural light-cone group and
# the noise-weighted sums restricted to it. Removing the "lightcone" group alone leaves lam_lc and lc_share in the
# noise group; the semantic ablation removes all of these.
LIGHTCONE_DERIVED = ("lc_qubits", "lc_gates", "lc_2q", "lc_locations", "lam_lc", "lam_lc_exact", "lam_lc_gate",
                     "lam_lc_idle", "lc_share", "lam_lc_2q")
SEMANTIC_ABLATIONS = {"-lightcone (all derived)": LIGHTCONE_DERIVED,
                      "-noise (calibration)": tuple(f.name for f in DICTIONARY if f.group == "noise"),
                      "-simulated": tuple(f.name for f in DICTIONARY if f.info == "simulated")}


def _eta_l1(lams, deg) -> float:
    lams = np.asarray(lams, float)
    V = np.vander(lams, deg + 1, increasing=True)
    eta = np.linalg.pinv(V)[0]
    return float(np.abs(eta).sum())


def task_features(prep) -> dict:
    """Budget-independent features of a prepared task (``emulate.Prepared``). Oracle entries are included for the
    research evaluation only; ``matrix`` drops them unless asked."""
    from .. import device
    from .. import features as FT
    from ..context_noise import alap_schedule
    from ..phase.lightcone import calculate_lambda_lc
    built = prep.built
    comp, obs = built.compiled, built.observable
    sup = built.spec.supplied_backend
    phys = tuple(comp.info["phys"])
    dm = device.get_model(sup, phys)
    names = ("n", "two_qubit_logical", "non_clifford", "two_qubit", "depth", "pulses_1q", "non_clifford_slots",
             "idle_fraction", "obs_weight", "obs_terms", "obs_support", "entropy_half", "output_entropy_bits")
    vals, _miss = FT.compute(comp, obs, sup, None, names=names)
    out = {k: _num(vals.get(k)) for k in names}
    out["duration_us"] = float(alap_schedule(comp.implemented, dm.durations)[3]) * 1e6
    rep = calculate_lambda_lc(comp, sup, obs)
    out.update(lc_qubits=float(len(rep.lc_qubits)), lc_gates=float(rep.lc_gates), lc_2q=float(rep.lc_2q_gates),
               lc_locations=float(rep.n_loc_lc), lam_lc=float(rep.lam_lc), lam_lc_exact=float(rep.lam_lc_exact),
               lam_lc_gate=float(rep.lam_lc_gate), lam_lc_idle=float(rep.lam_lc_idle), lam_total=float(rep.lam_total),
               lc_share=float(rep.lam_lc / rep.lam_total) if rep.lam_total > 0 else 0.0, lam_ro=float(rep.lam_ro))
    out.update(noise18(comp, dm, obs, rep))
    s = prep.supplied
    pec = s.get("PEC")
    out["pec_log_gamma"] = float(math.log(pec.gamma)) if _ok(pec) and pec.gamma > 0 else math.nan
    zl, zr = s.get("ZNE-L"), s.get("ZNE-R")
    out["zne_eta_l1_linear"] = _eta_l1(zl.details["lams"], 1) if _ok(zl) else math.nan
    out["zne_eta_l1_richardson"] = _eta_l1(zr.details["lams"], 2) if _ok(zr) else math.nan
    ro = s.get("RO")
    out["log10_v_raw_sup"] = _log10(ro.var1) if _ok(ro) else math.nan
    out["log10_v_zne_sup"] = _log10(zr.var1) if _ok(zr) else math.nan
    out["log10_v_pec_sup"] = _log10(pec.var1) if _ok(pec) else math.nan
    raw_s = s.get("RAW")
    if _ok(raw_s):
        ctx = prep.ctx_truth["RAW"]
        r_sup = ctx.runner if prep.same_law else None
        from ..methods import Runner
        run = (r_sup or Runner(backend=sup, layout=phys)).run(comp.implemented, ctx.noise, comp.measure_map)
        out["abs_signal_sup"] = float(abs(obs.o @ run.probs))
    else:
        out["abs_signal_sup"] = math.nan
    out["ideal"] = float(built.ideal)
    raw_t = prep.truth.get("RAW")
    out["raw_bias_true"] = float(raw_t.mean - built.ideal) if _ok(raw_t) else math.nan
    return out


def noise18(comp, dm, obs=None, rep=None, support_bits=None) -> dict:
    """The 1.8.0 noise-composition descriptors (NOISE18) from the compiled circuit, the snapshot model and the light
    cone report; ``support_bits``: the observable's classical bits (else from ``obs``)."""
    from ..phase.lightcone import observable_support
    mm = comp.measure_map or {}
    if support_bits is None:
        sup = observable_support(obs, mm, comp.implemented.n)              # physical (compact) qubits
    else:
        sup = tuple(sorted({int(mm.get(b, b)) for b in support_bits}))
    asym = [abs(dm.qubits[q].p01 - dm.qubits[q].p10) for q in sup]
    e2 = sorted(v for (nm, qa), v in dm.errors.items() if len(qa) == 2 and v is not None
                and any(o.k == 2 and tuple(o.qubits) in (tuple(qa), tuple(qa)[::-1]) for o in comp.implemented.ops
                        if o.kind == "gate"))
    from ..context_noise import alap_schedule
    dur = float(alap_schedule(comp.implemented, dm.durations)[3])
    t2 = [q.t2 for q in dm.qubits if q.t2 and np.isfinite(q.t2)]
    return dict(lam_lc_2q=float(getattr(rep, "lam_lc_2q", math.nan)) if rep is not None else math.nan,
                ro_asym=float(np.mean(asym)) if asym else 0.0,
                err2q_spread=float(e2[-1] / np.median(e2)) if len(e2) >= 2 and np.median(e2) > 0 else 1.0,
                duration_t2=float(dur / np.median(t2)) if t2 else math.nan)


def _ok(r) -> bool:
    return r is not None and not isinstance(r, Exception)


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _log10(x) -> float:
    return float(math.log10(x)) if x is not None and x > 0 and math.isfinite(x) else math.nan


def with_budget(feats: dict, B: int) -> dict:
    d = dict(feats)
    d["log10_budget"] = float(math.log10(int(B)))
    return d


def matrix(rows: list, names) -> np.ndarray:
    """(rows, features) array of the named features; refuses oracle features."""
    bad = [n for n in names if BY_NAME[n].info == "oracle"]
    if bad:
        raise ValueError("oracle features cannot be model inputs: %s" % ", ".join(bad))
    return np.array([[float(r.get(n, math.nan)) for n in names] for r in rows], float)


def dictionary_markdown() -> str:
    lines = ["| feature | class | group | definition |", "|---|---|---|---|"]
    for f in DICTIONARY:
        lines.append("| `%s` | %s | %s | %s |" % (f.name, f.info, f.group, f.definition))
    return "\n".join(lines)
