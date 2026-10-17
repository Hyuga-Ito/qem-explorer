r"""Probabilistic error cancellation (Temme, Bravyi & Gambetta, PRL 119, 180509 (2017);
Endo, Benjamin & Li, PRX 8, 031027 (2018); van den Berg et al., Nat. Phys. 19, 1116 (2023)).

After every noisy gate :math:`\mathcal O=\mathcal N\circ\mathcal U` apply a quasi-probability
correction :math:`\mathcal C=\sum_i q_i\mathcal B_i` with :math:`\mathcal C\circ\mathcal N=\mathrm{id}`
for the *assumed* noise model. Sampling :math:`\mathcal B_i` with probability :math:`|q_i|/\gamma`
and weighting each shot by :math:`\gamma\,\mathrm{sgn}(q_i)` is unbiased for that model, at the
cost :math:`\gamma_{\rm circ}=\prod_g\gamma_g` in standard deviation.

Constructions for a depolarizing model :math:`\mathcal D(\rho)=(1-p)\rho+p\,\mathrm{Tr}\rho\,I/d`
(Pauli eigenvalue :math:`f=1-p`):

* inverse       :math:`\mathcal D^{-1}=\sum_P q_P\,P\cdot P`, :math:`q_I=\frac{1+(d^2-1)/f}{d^2}`,
                :math:`q_{P\ne I}=\frac{1-1/f}{d^2}`, :math:`\gamma=\frac{2(d^2-1)/f-(d^2-2)}{d^2}`
                (:math:`\gamma=\frac{15/f-7}{8}` for two qubits)
* compensation  :math:`\mathcal U=\frac{1}{1-p}\mathcal O-\frac{p}{1-p}\mathcal T\circ\mathcal O`,
                :math:`\mathcal T` the completely depolarizing map (a uniformly random Pauli),
                :math:`\gamma=\frac{1+p}{1-p}` (larger than the inverse's, as Endo et al. note)

The *learned* model takes each gate's actual noise, Pauli-twirled, and inverts it exactly -- the idealized version
of learning a sparse Pauli–Lindblad model. It is exact for the gates the recipe twirls: Clifford gates (one-qubit
ones only with twirl = "all"). A non-Clifford gate cannot be twirled exactly, so the non-Pauli part of its noise
(coherent errors, amplitude damping) stays uncorrected; the result then carries a note.
"""
from __future__ import annotations

import math

import numpy as np

from .base import MethodResult, Param, QEMMethod, register
from .raw import cal_fields, display_circuit
from ..circuit import Op, is_clifford, layer_order
from ..noise import (Instruction, gate_noise_instructions, instr_superop, pauli_labels, pauli_matrix,
                     pauli_probs_from_eigs, ptm)


def depol_quasi(p: float, k: int, construction: str) -> list:
    """[(q, Pauli label)] of the correction after one k-qubit gate with assumed depolarizing p.

    Convention: the channel ρ ↦ (1−p)ρ + p·I/d applied as the Pauli mixture with weight p spread uniformly over all
    4ᵏ Paulis (identity included), so its Pauli-transfer eigenvalues are 1−p. Domain: finite 0 ≤ p < 1 (p = 1 maps
    every state to I/d and has no inverse). An RB or average gate infidelity r is *not* this p: for this channel
    r = p(d−1)/d."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        raise ValueError("the depolarizing probability must be a number, got %r" % (p,)) from None
    if not math.isfinite(p) or p < 0:
        raise ValueError("the depolarizing probability must be finite and ≥ 0, got %r" % p)
    if int(k) != k or int(k) < 1:
        raise ValueError("a gate acts on at least one qubit, got k = %r" % (k,))
    if construction not in ("inverse", "compensation"):
        raise ValueError("construction must be 'inverse' or 'compensation', got %r" % (construction,))
    k = int(k)
    d2 = 4 ** k
    labels = pauli_labels(k)
    if p == 0:
        return [(1.0, "I" * k)]
    if p >= 1:
        raise ValueError("a depolarizing probability p = %g ≥ 1 maps every state to the maximally mixed state: the "
                         "information the inverse would need is destroyed, so the channel cannot be inverted" % p)
    f = 1 - p
    if construction == "inverse":
        return [((1 + (d2 - 1) / f) / d2 if l == "I" * k else (1 - 1 / f) / d2, l) for l in labels]
    # Endo et al.: with weight a run the noisy gate as is; with weight b follow it by a uniformly random
    # Pauli (the completely depolarizing map T). The identity appears in both branches, which is what
    # makes gamma = a + |b| = (1+p)/(1-p) larger than the inverse's (the branches are sampled separately).
    a, b = 1 / (1 - p), -p / (1 - p)
    return [(a, "I" * k)] + [(b / d2, l) for l in labels]


LEARNED_SINGULAR_TOL = 1e-12        # |Pauli-transfer eigenvalue| at or below this: the channel is not invertible


def learned_quasi(noise, o: Op) -> list:
    """Exact inverse of the Pauli twirl of this gate's noise channel."""
    ins = gate_noise_instructions(noise, o, 0)
    k = o.k
    if not ins:
        return [(1.0, "I" * k)]
    S = np.eye(4 ** k, dtype=complex)
    for i in ins:
        S = instr_superop(i, o.qubits) @ S
    eig = np.real_if_close(np.diag(ptm(S)))
    # the twirled channel is invertible iff no Pauli-transfer eigenvalue vanishes; negative nonzero eigenvalues are
    # legitimate (they invert to negative weights) and are kept
    small = np.flatnonzero(np.abs(eig) <= LEARNED_SINGULAR_TOL)
    if len(small):
        labs = pauli_labels(k)
        raise ValueError("the noise of gate %s (on qubits %s) is not invertible: its Pauli-twirled channel destroys the "
                         "%s component%s (transfer eigenvalue %s), so no quasi-probability correction exists"
                         % (o.name, ",".join(map(str, o.qubits)), ", ".join(labs[i] for i in small[:4]),
                            "" if len(small) == 1 else "s", ", ".join("%.2g" % abs(eig[i]) for i in small[:4])))
    q = pauli_probs_from_eigs(1 / eig, k)
    return [(float(v), l) for v, l in zip(q, pauli_labels(k)) if abs(v) > 1e-15]


def gamma_of(qs) -> float:
    return float(sum(abs(v) for v, _ in qs))


@register
class PEC(QEMMethod):
    key = "pec"
    label = "Probabilistic error cancellation"
    family = "pec"
    tex = (r"\mathcal U=\sum_i q_i\,\mathcal B_i\circ\mathcal O,\qquad \gamma_g=\sum_i|q_i|,\qquad "
           r"\sigma\propto\frac{\prod_g\gamma_g}{\sqrt N}")
    why = (r"The inverse of a noise channel inflates the Bloch ball, which no physical operation can do, so it is "
           r"written as a mixture of physical operations with some **negative** weights. The estimator is the "
           r"difference of the positive and negative parts; their masses $(\Gamma\pm1)/2$ grow like "
           r"$\Gamma=\prod_g\gamma_g$, exponentially in the number of gates. PEC is unbiased only for the noise "
           r"model it is given.")
    params = (
        Param("construction", "construction", "choice", "inverse", ("inverse", "compensation")),
        Param("model", "noise model", "choice", "depolarizing", ("depolarizing", "learned"),
              help="depolarizing: told p₁, p₂ below; learned: the exact Pauli-twirled channel of every gate."),
        Param("p1", r"assumed $p_1$ (one-qubit gates)", "float", None, lo=0.0, hi=0.2, step=1e-4,
              help="Blank = the injector's true value."),
        Param("p2", r"assumed $p_2$ (two-qubit gates)", "float", None, lo=0.0, hi=0.3, step=1e-4,
              help="Blank = the injector's true value."),
        Param("scope", "gates corrected", "choice", "all", ("all", "2q")),
        Param("shots_per_circuit", "shots per sampled circuit", "int", 1, lo=1, hi=100000, step=1,
              help="The protocol samples one circuit (one Pauli correction per gate) for every this many shots. "
                   "1 is the textbook estimator (Temme et al.): every shot is an independent sample, Var = "
                   "(γ²⟨o²⟩ − ⟨O⟩²)/N. Larger values send fewer circuits and add the instance-to-instance spread."),
        Param("instances", "instances simulated (Aer)", "int", 64, lo=8, hi=1024, step=8,
              help="Aer cannot apply negative weights, so its mean is estimated from this many sampled instances "
                   "(the NumPy simulator computes the exact quasi-probability mean)."),
    )

    def short(self, p):
        return "PEC %s%s" % (p["construction"], "" if p["model"] == "depolarizing" else " (learned)")

    def quasi_for(self, ctx, o: Op, construction, model, p1, p2, scope):
        if o.kind != "gate" or o.is_virtual or (scope == "2q" and o.k == 1):
            return None
        if model == "learned":
            dm = ctx.runner.device_model
            if dm is not None:
                # device: per-gate depolarizing model from this gate's calibrated error r (p = r d/(d-1))
                r = dm.gate_error(o.name, tuple(o.qubits))
                if r is None:
                    return None
                d = 2 ** o.k
                return depol_quasi(min(float(r) * d / (d - 1), 0.999), o.k, "inverse")
            return learned_quasi(ctx.noise, o)
        p = (ctx.noise.p1 if p1 is None else p1) if o.k == 1 else (ctx.noise.p2 if p2 is None else p2)
        return depol_quasi(float(p), o.k, construction)

    def run(self, ctx, construction="inverse", model="depolarizing", p1=None, p2=None, scope="all",
            instances=64, shots_per_circuit=1) -> MethodResult:
        spc = max(int(shots_per_circuit or 1), 1)
        notes = []
        if model == "learned" and construction == "compensation":
            notes.append("The compensation construction is defined for a depolarizing model; the learned "
                         "model uses the inverse.")
        if model == "learned" and ctx.recipe.twirl == "none":
            notes.append("Without twirling, coherent parts of the noise are not Pauli; the learned Pauli model "
                         "leaves them uncorrected.")
        elif model == "learned":
            from ..circuit import is_clifford
            untw = sorted({o.name for o in ctx.spec.ops if o.kind == "gate" and not o.is_virtual
                           and (not is_clifford(o) or (o.k == 1 and ctx.recipe.twirl != "all"))})
            if untw:
                notes.append("Not twirled (%s): only Clifford gates can be twirled exactly%s, so non-Pauli parts of "
                             "these gates' noise stay uncorrected." % (", ".join(untw), "" if ctx.recipe.twirl == "all"
                                                                        else ", and twirl = 2q leaves one-qubit gates"))
        cache = {}

        def q_of(o):
            k = (o.name, o.qubits, o.params)
            if k not in cache:
                cache[k] = self.quasi_for(ctx, o, construction, model, p1, p2, scope)
            return cache[k]

        def corrections(absolute=False):
            def f(o, step):
                qs = q_of(o)
                if qs is None:
                    return []
                items = [(abs(v) if absolute else v, pauli_matrix(l)) for v, l in qs]
                return [Instruction("mixture", o.qubits, items, step, "pec", "PEC correction")]
            return f

        gates = [o for o in ctx.spec.ops if o.kind == "gate"]
        gam_path = [1.0]
        for o in gates:
            qs = q_of(o)
            gam_path.append(gam_path[-1] * (gamma_of(qs) if qs else 1.0))
        G = gam_path[-1]
        ckey = ("pec", construction, model, p1, p2, scope)
        from .. import device
        backend_neg = device.supports_negative(ctx.runner.backend)       # numpy, or a device with eng=exact
        exec_ops = [o for o in layer_order(ctx.spec)[0] if o.kind != "barrier"]   # the steps resolve() uses

        def sample_instances(count, seed):
            """count sampled circuits: (Γ·sign, run result) each."""
            rng_i = np.random.default_rng(seed)
            out = []
            for _ in range(int(count)):
                sign, draw = 1, {}
                for step, o in enumerate(exec_ops):
                    qs = q_of(o) if o.kind == "gate" else None
                    if not qs:
                        continue
                    w = np.array([abs(v) for v, _ in qs])
                    j = rng_i.choice(len(qs), p=w / w.sum())
                    sign *= int(np.sign(qs[j][0]))
                    if set(qs[j][1]) != {"I"}:
                        draw[step] = qs[j][1]

                def fix(o, step, _d=dict(draw)):
                    l = _d.get(step)
                    return [Instruction("unitary", o.qubits, pauli_matrix(l), step, "pec", "PEC Pauli")] if l else []
                out.append((G * sign, ctx.run(corrections=fix, corr_key=ckey + (tuple(sorted(draw.items())),))))
            return out
        inst = None
        if backend_neg:
            net = ctx.run(corrections=corrections(), corr_key=ckey)
            absr = ctx.run(corrections=corrections(True), corr_key=ckey + ("abs",))
            mean = ctx.expval(net)
            mc_err = 0.0
        else:
            # Aer cannot apply negative weights: sample instances (each a CPTP circuit), weight by Γ·sign
            inst = sample_instances(instances, ctx.seed + 17)
            vals = np.array([w * ctx.expval(rr) for w, rr in inst])
            mean = float(vals.mean())
            mc_err = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            notes.append("Aer: mean from %d sampled instances (± %.3g Monte Carlo)." % (len(vals), mc_err))
            net = absr = None
        g_ro = ctx.readout_gamma()
        o_eff = ctx.o_eff()
        if absr is not None:
            # each shot is one instance: value ±Γ o_eff(x) with probability p_abs(x)/Γ  =>  E[X²] = Γ Σ p_abs o_eff²
            m2 = G * float(np.clip(ctx.measured(absr), 0, None) @ o_eff ** 2)
        else:
            # each shot: instance i drawn with probability |q_i|/Γ, outcome x ~ p_i, value ±Γ·o_eff(x) (o_eff folds
            # the readout mitigation in): E[X²] = Γ² · E_i[Σ_x p_i(x) o_eff(x)²], estimated over the sampled instances
            # (the same quantity the exact path computes as Γ Σ_x p_abs(x) o_eff(x)²)
            m2 = G ** 2 * float(np.mean([np.clip(ctx.measured(rr), 0, None) @ o_eff ** 2 for _, rr in inst]))
        var1 = max(m2 - mean ** 2, 0.0)              # one shot per sampled circuit (spc = 1)
        var_inst = 0.0
        if spc > 1:
            # s shots per circuit, M = N/s circuits: Var = Var_inst/M + (E[X²|shot] − E[X_i²])/N
            #   = [Var1 + (s − 1) Var_inst] / N, with Var_inst = Var over instances of Γ·sign·⟨o⟩_i
            pool = inst if inst is not None else sample_instances(max(int(instances), 2), ctx.seed + 29)
            xi = np.array([w * ctx.expval(rr) for w, rr in pool])
            var_inst = float(np.var(xi, ddof=1)) if len(xi) > 1 else 0.0
            var1 = var1 + (spc - 1) * var_inst
            notes.append("%d shots per sampled circuit: the instance-to-instance variance %.3g (a Monte-Carlo estimate "
                         "from %d sampled instances) adds (s − 1)·Var_inst = %.3g to the per-shot variance." % (
                             spc, var_inst, len(xi), (spc - 1) * var_inst))

        def sampler(N, B, rng):
            if net is None or spc > 1:
                return rng.normal(mean, np.sqrt(var1 / max(N, 1)), size=B)
            # exact sampling: each shot is one instance, a (sign, outcome) pair
            pn, pa = ctx.measured(net), ctx.measured(absr)
            pos, neg = np.clip((pa + pn) / 2, 0, None), np.clip((pa - pn) / 2, 0, None)
            joint = np.concatenate([pos, neg])
            joint = joint / joint.sum()
            vals = np.concatenate([G * o_eff, -G * o_eff])
            N = max(int(N), 1)
            cnt = rng.multinomial(N, joint, size=B)
            return cnt @ vals / N

        # one sampled instance, for display
        rng = np.random.default_rng(ctx.seed + 3)
        ops, sign, nins = [], 1, 0
        for o in ctx.spec.ops:
            ops.append(o)
            qs = q_of(o) if o.kind == "gate" else None
            if not qs:
                continue
            w = np.array([abs(v) for v, _ in qs])
            j = rng.choice(len(qs), p=w / w.sum())
            sign *= int(np.sign(qs[j][0]))
            for ch, q in zip(qs[j][1], o.qubits):
                if ch != "I":
                    ops.append(Op(ch.lower(), (q,), (), "gate", "pec"))
                    nins += 1
        disp_inst = ctx.spec.replace(ops=ops, name="PEC instance (sign %+d)" % sign)
        example = next((q_of(o) for o in gates if o.k == 2 and q_of(o)), None) or \
            next((q_of(o) for o in gates if q_of(o)), [(1.0, "I")])
        if net is not None:
            f_net = ctx.measured(net)
            cal_fun = lambda mit: float(ctx.obs.o @ mit(f_net))
        else:
            f_inst = [(w, ctx.measured(rr)) for w, rr in inst]
            cal_fun = lambda mit: float(np.mean([w * (ctx.obs.o @ mit(f)) for w, f in f_inst]))
        return MethodResult(
            mean=mean, var1=var1, circuits=1, shots_per_circuit=spc, gamma=G * g_ro, **cal_fields(ctx, cal_fun),
            implemented=[("Sampled instance · sign %+d · %d Pauli insertions" % (sign, nins), display_circuit(ctx, disp_inst))],
            details=dict(gamma_path=np.array(gam_path), quasi=example, sign=sign, mc_err=mc_err, var_inst=var_inst,
                         shots_per_circuit=spc,
                         p_net=None if net is None else ctx.measured(net), p_abs=None if absr is None else ctx.measured(absr)),
            notes=notes, sampler=sampler)
