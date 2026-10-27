"""Cited parameter presets for circuit templates (1.5).

Every preset fills a template's parameters with stated provenance: source and version, the locator inside it
(equation / table / figure / section), units, the gate and angle convention, the problem instance, the sizes it is
meant for, its purpose and its limitations. ``kind`` says how it relates to the source:

* ``exact reproduction`` — the setting and the quantity it checks are the source's, in the same convention (the
  test suite reproduces the quantity);
* ``adapted`` — follows the source but changes something (direction, index convention, gates); the change is
  listed under ``differences``;
* ``educational`` — a teaching example; any source is background, not an endorsement of these values.

Nothing here is "recommended by the literature": for example, there is no universal optimal QAOA angle or circuit
depth. Each source was read on the date in ``checked`` (arXiv versions as listed); see docs/literature_matrix.md.
Selecting a preset only fills the template's controls; every value stays editable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

KINDS = ("exact reproduction", "adapted", "educational")
CHECKED = "2026-10-17"


@dataclass(frozen=True)
class Source:
    authors: str
    title: str
    arxiv: str = ""                # id with version, e.g. 1411.4028v1
    journal: str = ""
    doi: str = ""

    def cite(self) -> str:
        bits = [self.authors, "“%s”" % self.title]
        if self.arxiv:
            bits.append("arXiv:%s" % self.arxiv)
        if self.journal:
            bits.append(self.journal)
        if self.doi:
            bits.append("doi:%s" % self.doi)
        return ", ".join(bits)


@dataclass(frozen=True)
class CitedPreset:
    key: str
    title: str
    template: str
    params: dict
    kind: str
    purpose: str
    instance: str
    convention: str
    units: str
    sizes: str
    limitations: str
    sources: tuple = ()            # ((Source, locator), …)
    differences: str = ""
    checked: str = CHECKED
    tags: tuple = field(default=())

    def search_text(self) -> str:
        return " ".join([self.key, self.title, self.template, self.kind, self.purpose, self.instance, " ".join(self.tags),
                         " ".join(s.cite() + " " + loc for s, loc in self.sources)]).lower()


FARHI = Source("E. Farhi, J. Goldstone, S. Gutmann", "A Quantum Approximate Optimization Algorithm", "1411.4028v1")
WANG = Source("Z. Wang, S. Hadfield, Z. Jiang, E. G. Rieffel",
              "Quantum Approximate Optimization Algorithm for MaxCut: A Fermionic View", "1706.02998v2",
              "Phys. Rev. A 97, 022304 (2018)", "10.1103/PhysRevA.97.022304")
LUBINSKI = Source("T. Lubinski et al.", "Application-Oriented Performance Benchmarks for Quantum Computing",
                  "2110.03137v3", "IEEE Trans. Quantum Eng. (2023)", "10.1109/TQE.2023.3253761")
COPPERSMITH = Source("D. Coppersmith", "An approximate Fourier transform useful in quantum factoring",
                     "quant-ph/0201067v1", "IBM Research Report RC 19642 (1994)")
KANDALA = Source("A. Kandala et al.", "Hardware-efficient variational quantum eigensolver for small molecules and "
                 "quantum magnets", "1704.05018v2", "Nature 549, 242 (2017)", "10.1038/nature23879")
CHILDS_SU = Source("A. M. Childs, Y. Su", "Nearly optimal lattice simulation by product formulas", "1901.00564v2",
                   "Phys. Rev. Lett. 123, 050503 (2019)", "10.1103/PhysRevLett.123.050503")
CHILDS_TROTTER = Source("A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, S. Zhu", "A Theory of Trotter Error",
                        "1912.08854v3", "Phys. Rev. X 11, 011020 (2021)", "10.1103/PhysRevX.11.011020")

QFT_CONVENTION = ("F|y⟩ = N^(−1/2) Σ_k e^(2πi·yk/N)|k⟩, N = 2^n, integers little-endian (q0 = 1s bit); the circuit is "
                  "H and controlled phases π/2^d followed by a SWAP network (or a bit-reversed load / relabeling).")
QAOA_CONVENTION = ("C = Σ_(i,j)∈E (1 − Z_i Z_j)/2, B = Σ_i X_i, |γ,β⟩ = e^(−iβB) e^(−iγC) |+⟩^n; the app applies "
                   "RZZ(−γ) on every edge and RX(2β) on every qubit, which equals this up to a global phase.")

PRESETS = (
    CitedPreset(
        key="qft-known-answer", title="QFT · known answer (Fourier-state input)", template="qft",
        params=dict(n=4, s=5, mode="swap", prep="fourier"), kind="adapted",
        purpose="A benchmark with a known ideal answer: the ideal output is |s⟩ with probability 1, so any deviation "
                "is error (useful for observable-level QEM checks with a trusted reference).",
        instance="prepare the product state F†|s⟩ (n Hadamards, ≤ n phases), apply F, measure P(s)",
        convention=QFT_CONVENTION, units="s: integer 0 … 2^n − 1", sizes="any n ≥ 2; n(n−1)/2 controlled phases",
        limitations="Tests one known output; the prepared state is a product state; the add-one variant QFT(1) of "
                    "the source is not implemented.",
        sources=((LUBINSKI, "App. A.2 and Fig. 26 (benchmark QFT(2))"),),
        differences="The source prepares the Fourier state of x and applies the inverse QFT; here F†|s⟩ is prepared "
                    "and F applied (the mirror image). The source's fidelity metric is not used here.",
        tags=("qft", "known answer", "benchmark", "qed-c")),
    CitedPreset(
        key="qft-basis-input", title="QFT · basis-state input (uniform output)", template="qft",
        params=dict(n=3, s=1, mode="swap", prep="basis"), kind="educational",
        purpose="Shows that an output is a distribution, not a bitstring: F|s⟩ gives every outcome with probability "
                "1/N; the input only sets phases, which a Z-basis measurement cannot see.",
        instance="prepare |s⟩ with X gates, apply F, observe P(0…0) = 1/N", convention=QFT_CONVENTION,
        units="s: integer 0 … 2^n − 1", sizes="any n ≥ 2",
        limitations="The observable P(0…0) is the same for every s; measure in X or Y to see the phases.",
        tags=("qft", "distribution", "phases")),
    CitedPreset(
        key="aqft-coppersmith", title="AQFT · Coppersmith truncation m_C = 3", template="aqft",
        params=dict(n=6, s=1, mode="swap", m=2, prep="fourier"), kind="adapted",
        purpose="Shows the accuracy/gate-count trade-off of dropping small controlled phases.",
        instance="known-answer QFT with controlled phases π/2^d kept only for d < m_C (app m = m_C − 1 = 2)",
        convention=QFT_CONVENTION + " App degree m keeps distances d ≤ m.",
        units="m: qubit distance (integer)", sizes="n ≥ 3; about n·m controlled phases",
        limitations="The source's bound on each entry's phase error, 2π·n·2^(−m_C), is far from tight for small n "
                    "(it exceeds π here); the app reports the actual success probability instead.",
        sources=((COPPERSMITH, "section “Approximate Fourier Transform” (rotations with K ≥ J + m deleted; error "
                               "factor exp(iε), |ε| ≤ 2πL·2^(−m))"),),
        differences="Index shift: the app's m is Coppersmith's m − 1 (his m = 1 is the Hadamard transform).",
        tags=("aqft", "truncation", "approximate")),
    CitedPreset(
        key="qaoa-ring-p1", title="QAOA · ring, p = 1, (γ, β) = (π/4, π/8)", template="qaoa",
        params=dict(n=6, graph="ring", p=1, angles="fixed", gamma=math.pi / 4, beta=math.pi / 8, seed=1),
        kind="exact reproduction",
        purpose="Reproduces the p = 1 approximation ratio 3/4 on rings (the 'ring of disagrees').",
        instance="MaxCut on an even ring (2-regular, n ≥ 4), one QAOA layer, observable ⟨C⟩/C_max",
        convention=QAOA_CONVENTION, units="γ, β in radians", sizes="even n ≥ 4 (odd rings have C_max = n − 1)",
        limitations="Optimal for p = 1 on rings in this convention only; no universal optimal QAOA angle or depth "
                    "exists. Ratio 3/4 is the noiseless value.",
        sources=((FARHI, "Sec. IV (ring of disagrees: p = 1 maximum 3/4; no angles stated)"),
                 (WANG, "Sec. III (ratio 0.75 at (β, γ) = (π/8, π/4))")),
        tags=("qaoa", "maxcut", "ring", "angles")),
    CitedPreset(
        key="qaoa-3reg-p1", title="QAOA · 3-regular (Möbius ladder), p = 1, optimized", template="qaoa",
        params=dict(n=8, graph="3-regular", p=1, angles="optimized", seed=1), kind="adapted",
        purpose="Compares the app's optimized p = 1 ratio with the worst-case p = 1 guarantee 0.6924 for 3-regular "
                "graphs.",
        instance="MaxCut on the Möbius ladder with n nodes, angles optimized on a noiseless statevector",
        convention=QAOA_CONVENTION, units="γ, β in radians (optimized)", sizes="even n ≥ 4",
        limitations="The guarantee is a worst case over all 3-regular graphs; one graph says nothing about the "
                    "worst case. Angles are optimized classically per instance (cost 2^n).",
        sources=((FARHI, "Sec. V (p = 1 worst-case ratio 0.6924 on 3-regular graphs)"),),
        differences="A single graph family (Möbius ladder) instead of all 3-regular graphs.",
        tags=("qaoa", "maxcut", "3-regular")),
    CitedPreset(
        key="hea-depth-2", title="VQE · hardware-efficient ansatz, depth 2", template="hea_vqe",
        params=dict(n=4, layers=2, entanglement="linear", angles="VQE-optimized", seed=7), kind="educational",
        purpose="A shallow layered ansatz: d entangling layers between d + 1 rotation layers.",
        instance="antiferromagnetic chain H = Σ Z_i Z_(i+1), angles optimized on a noiseless statevector",
        convention="rotation layers RY·RZ on every qubit, CX entanglers (linear/circular/full)",
        units="angles in radians", sizes="n ≥ 2; 2n(d + 1) angles",
        limitations="Not Kandala et al.'s circuit: they use Z·X·Z Euler rotations and cross-resonance drift "
                    "entanglers on molecules and magnets. The layer structure is the only shared feature.",
        sources=((KANDALA, "Eq. (2), Fig. 1c–d (d entanglers alternating with Euler-rotation layers)"),),
        tags=("vqe", "hea", "ansatz", "depth")),
    CitedPreset(
        key="ising-trotter-1st", title="Ising · first-order Trotter, 4 steps of dt = 0.25", template="ising",
        params=dict(n=4, steps=4, h=1.0, dt=0.25), kind="educational",
        purpose="Shows the Trotter error of the first-order product formula against the exact evolution.",
        instance="H = −J Σ Z_i Z_(i+1) − h Σ X_i (J = 1, open chain) from |0…0⟩; t = steps·dt",
        convention="one step = e^(−i dt H_ZZ) e^(−i dt H_X) (RZZ on bonds, then RX on sites); ħ = 1",
        units="dt and t in units of 1/J", sizes="any n ≥ 2",
        limitations="The commutator bound below is an upper bound obtained with the triangle inequality and is "
                    "loose; the actual error depends on the state.",
        sources=((CHILDS_SU, "Eq. (6): ‖S₁(t) − e^(−itH)‖ ≤ ∫₀ᵗ∫₀^τ₁ ‖[H_even, H_odd]‖ = (t²/2)‖[·,·]‖ per step"),
                 (CHILDS_TROTTER, "Sec. 3.1: first-order error O(‖[B, A]‖ t²)")),
        differences="The app splits H into the ZZ part and the X part, not even/odd bonds; the bound applies to any "
                    "two-term split.",
        tags=("trotter", "ising", "product formula", "time step")),
)
BY_KEY = {p.key: p for p in PRESETS}


def search(query: str = "", template: str | None = None) -> list:
    """Presets whose text contains every word of ``query`` (case-insensitive), optionally of one template."""
    words = [w for w in str(query or "").lower().split() if w]
    out = []
    for p in PRESETS:
        if template and p.template != template:
            continue
        t = p.search_text()
        if all(w in t for w in words):
            out.append(p)
    return out


def resolve(key: str) -> dict:
    """The preset's parameters made valid for its template (every value shown), plus derived quantities."""
    from . import templates as T
    p = BY_KEY[key]
    vals = T.clamp_params(p.template, dict(p.params))
    derived = {}
    if p.template == "aqft":
        mC = int(vals["m"]) + 1
        derived["Coppersmith m_C"] = mC
        derived["phase-error bound 2π·n·2^(−m_C) (rad)"] = 2 * math.pi * int(vals["n"]) * 2.0 ** (-mC)
    if p.template == "ising":
        n, h, dt, r = int(vals["n"]), float(vals["h"]), float(vals["dt"]), int(vals["steps"])
        J = 1.0
        # ‖[H_ZZ, H_X]‖ ≤ Σ_bonds Σ_(j in bond) |J h| ‖[Z_i Z_(i+1), X_j]‖ = 2(n−1)·|J h|·2
        comm = 4 * (n - 1) * abs(J * h)
        derived["total time t"] = r * dt
        derived["per-step bound (dt²/2)·‖[H_ZZ,H_X]‖ ≤"] = 0.5 * dt * dt * comm
        derived["total bound (r steps) ≤"] = r * 0.5 * dt * dt * comm
    if p.template == "qaoa" and vals.get("angles") == "fixed":
        derived["γ/π"] = float(vals["gamma"]) / math.pi
        derived["β/π"] = float(vals["beta"]) / math.pi
    return dict(params=vals, derived=derived)


def describe(key: str) -> str:
    """Markdown provenance card."""
    p = BY_KEY[key]
    r = resolve(key)
    lines = ["**%s** · *%s*" % (p.title, p.kind), "", p.purpose, "",
             "- **Instance:** %s" % p.instance, "- **Convention:** %s" % p.convention, "- **Units:** %s" % p.units,
             "- **Sizes:** %s" % p.sizes, "- **Limitations:** %s" % p.limitations]
    if p.differences:
        lines.append("- **Differences from the source:** %s" % p.differences)
    for s, loc in p.sources:
        lines.append("- **Source:** %s — %s" % (s.cite(), loc))
    if not p.sources:
        lines.append("- **Source:** none (a teaching example; the convention above is the definition used)")
    lines.append("- **Resolved values:** " + ", ".join("%s = %s" % (k, ("%.6g" % v) if isinstance(v, float) else v)
                                                    for k, v in r["params"].items()))
    for k, v in r["derived"].items():
        lines.append("- %s %s" % (k, ("%.4g" % v) if isinstance(v, float) else v))
    lines.append("- Sources read on %s." % p.checked)
    return "\n".join(lines)
