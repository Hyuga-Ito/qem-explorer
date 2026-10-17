"""Rich column tooltips: every metric the tables show, with its formula (rendered by KaTeX in the tooltip) and a
practical reading tip. ``column_config(df)`` returns formats and help for whatever columns a table has."""
from __future__ import annotations

import streamlit as st

from ui.tooltips import plain

# name -> (format, markdown help with $LaTeX$)
METRICS = {
    "estimate": ("%.4f", r"**Expected estimate** $\mathbb E[\hat O]$ of the protocol as specified — not one sampled run. "
                         r"For linear estimators (ZNE, PEC, readout mitigation, …) it is exact. For learning methods "
                         r"(CDR / FR / NIL) it is the mean over Monte-Carlo draws of the finite training data "
                         r"$\mathbb E_D[m(D)]$ (see *estimate kind* and *bias MC ±*)."
                         "\n\nCompare it with the ideal value: the difference is the bias. On hardware it is the "
                         "value computed from the measured counts — one experiment, including its shot noise."),
    "estimate kind": (None, "What **estimate** is: *exact* (exact probabilities), *protocol* (Monte-Carlo expectation "
                            "of the finite-training protocol), *measured* (one hardware experiment) or *not run* "
                            "(insufficient budget)."),
    "exact-fit value": ("%.4f", r"**Reference fit** of a learning method: the regression fitted to *exact* training "
                                r"features — an oracle quantity no experiment sees. With finite training shots the fit "
                                r"is a nonlinear function of noisy data, so the protocol's expected value (*estimate*) "
                                r"differs: $\mathbb E_D[\hat f_D(x)]\neq f_{\mathbb E[D]}(x)$."),
    "bias MC ±": ("%.4f", r"**Monte-Carlo standard error** of the protocol mean (and so of the bias) from the finite "
                          r"number of training-data draws: $\sqrt{u/R}$. It is not experimental shot noise."),
    "flags": (None, "Approximations and validity conditions of this result, e.g. *calibration-mean-first-order* (the "
                    "readout calibration enters the variance to first order; its O(1/s) effect on the mean is not "
                    "included), *conditional-on-success* (failed refits excluded), *measured*."),
    "simulated (Playground)": ("%.4f", "The same contender's estimate in the Playground's simulation (same circuit, "
                                       "observable and budget), for comparison with the device."),
    "bias": ("%+.4f", r"**Bias** $b=\mathbb E[\hat O]-\langle O\rangle_{\rm ideal}$."
                      "\n\nWhat no number of shots can remove. *Closer to 0 means the method undoes more of the noise.*"),
    "sd": ("%.4f", r"**Standard error of one experiment** at the shot budget $B$: $\sigma=\sqrt{v/N_t+w/s+u}$, with "
                   r"$v$ the per-shot variance on the $N_t$ target shots, $w/s$ the error of a readout calibration "
                   r"sampled with $s$ shots per calibration circuit and $u$ the training data's shot noise (CDR / FR / NIL)."
                   "\n\nMitigation usually multiplies it by the overhead $\\gamma$. *Lower means a more precise single run.*"),
    "sd from": (None, "Whether **sd** is the analytic $\\sqrt{v/N_t+w/s+u}$ or the spread of simulated repeats (a "
                      "learning method whose refit on noisy training data is clearly non-linear)."),
    "sd (repeats)": ("%.4f", r"**Empirical spread** $\sqrt{\tfrac{1}{B-1}\sum_{k=1}^{B}(\hat O_k-\bar O)^2}$ of "
                             r"$B=120$ simulated experiments with $N$ shots each."
                             "\n\nShould agree with **sd**; a large difference flags a non-linear estimator."),
    "RMSE": ("%.4f", r"**Root-mean-square error** $\mathrm{RMSE}=\sqrt{b^2+\sigma^2}$ at the common budget $B$ "
                     r"(simulation). For one *measured* hardware run it is $|\hat O-\langle O\rangle_{\rm ideal}|$, "
                     r"whose square is an unbiased estimate of the MSE."
                     "\n\nThe error you should expect from one run. *Lowest RMSE = best method at this shot budget*; "
                     "with more shots the variance shrinks and low-bias methods win."),
    "γ": ("%.3f", r"**Sampling overhead** $\gamma$: the factor by which mitigation inflates the standard deviation, "
                  r"so $\gamma^2$ times more shots give the same precision. For PEC $\gamma=\sum_i|q_i|$ "
                  r"(quasi-probabilities), for ZNE $\gamma=\sum_j|\eta_j|$ (extrapolation weights)."
                  "\n\n*Close to 1 means cheap mitigation; PEC's $\\gamma$ grows exponentially with circuit size.*"),
    "circuits": ("%d", r"**Circuits sent to the backend** per estimate: noise scales (ZNE), sampled circuits (PEC: "
                       r"$N_t/s_{\rm PEC}$; with twirling one random circuit per shot), training circuits (CDR / FR / NIL) "
                       r"and readout-calibration circuits; TREX sends each fixed circuit with 8 random measurement "
                       r"masks."),
    "extra qubits": ("%d", r"**Extra qubits** the method needs, e.g. $M$ copies of the register for virtual "
                           r"distillation ($(M-1)n$ extra)."),
    "accepted": ("%.3f", r"**Acceptance** $P_{\rm accept}$ of post-selection (symmetry verification): the fraction "
                         r"of shots kept. The effective shot count is $P_{\rm accept}\,N$."
                         "\n\n*Close to 1 means few errors were detected — or the check is blind to them.*"),
    "target shots": ("%d", r"**Target shots** $N_t$: the part of the budget spent on the circuits that carry the "
                           r"estimate, $N_t=B-F-cs$."),
    "training shots": ("%d", r"**Training shots** $F$ of CDR / FR / NIL (training circuits × shots each), counted in the "
                             r"budget."),
    "calibration shots": ("%d", r"**Readout-calibration shots** $cs$: $c$ calibration circuits with $s$ shots each, "
                                r"$s$ chosen to minimize $v/N_t+w/s$ within the budget."),
    "QPU s": ("%.2f", r"**QPU time** $t=B\,t_{\rm shot}$ — the same for every method, because every method spends "
                      r"the same budget $B$ (on hardware: the shots actually sent × $t_{\rm shot}$)."),
    "status": (None, "Flags a result that should not be read at face value, e.g. a learning fit that could not be "
                     "trained (**not applied**: the unmitigated value is shown) or a weak training signal."),
    "failed repeats": ("%d", "Simulated repeats in which the estimator failed (e.g. an exponential fit when "
                             r"$E(\lambda)$ changes sign). They are counted, not averaged."),
    "ideal": ("%.4f", r"**Ideal value** $\langle O\rangle_{\rm ideal}=\mathrm{Tr}(\rho_{\rm ideal}O)$ from a "
                      r"noiseless simulation."),
    "stderr": ("%.4f", r"**Standard error from the measured counts**, estimated by resampling (bootstrap) the counts "
                       r"$B$ times."),
    "gamma": ("%.3f", r"**Sampling overhead** $\gamma$ (see γ)."),
    "per-shot σ": ("%.3f", r"**Per-shot standard deviation** $\sqrt{v}$ including the overhead: "
                           r"$\sigma_N=\sqrt{v}/\sqrt N$."),
    "budget for ε": ("%.3g", r"**Total budget to reach the target**: the smallest $B$ with "
                             r"$b^2+v/N_t(B)+w/s(B)\le\varepsilon^2$, training and calibration included "
                             r"(∞ when $|b|\ge\varepsilon$: no number of shots is enough)."),
    "QPU time for ε (s)": ("%.3g", r"$t_\varepsilon=B_\varepsilon\,t_{\rm shot}$."),
    "beats unmitigated above B": ("%.3g", r"**Crossover** $B^*$: above this total budget the method's RMSE is "
                                          r"lower than the unmitigated one."),
    "|bias|": ("%.4f", r"$|b|=|\mathbb E[\hat O]-\langle O\rangle_{\rm ideal}|$."),
}


def help_for(name: str) -> str | None:
    m = METRICS.get(name)
    return m[1] if m else None


def column_config(columns, overrides: dict | None = None) -> dict:
    """Formats and rich (LaTeX) header tooltips for the columns present."""
    out = {}
    for c in columns:
        m = METRICS.get(c)
        if not m:
            continue
        fmt, hlp = m
        hlp = plain(hlp)                       # Unicode math: KaTeX is unreliable inside tooltips
        if fmt is None:
            out[c] = st.column_config.TextColumn(c, help=hlp)
        elif fmt == "%d":
            out[c] = st.column_config.NumberColumn(c, format="%d", help=hlp)
        else:
            out[c] = st.column_config.NumberColumn(c, format=fmt, help=hlp)
    out.update(overrides or {})
    return out


def legend(columns=("bias", "sd", "RMSE", "γ", "circuits", "QPU s")):
    """A strip of metric names; hovering each shows its formula and a tip (the same text as the headers)."""
    row = st.container(horizontal=True, gap="small")
    with row:
        st.caption("Hover a metric (or a column header) for its formula:")
        for c in columns:
            if c in METRICS:
                st.markdown("`%s`" % c, help=METRICS[c][1], width="content")


LAYOUT_HELP = {
    "F": r"**Expected success** $F=\prod_{\rm 2q}(1-e_{ab})\prod_{\sqrt X,X}(1-e_a)\prod_{\rm meas}(1-r_a)$ from the "
         r"device calibration: the probability that no gate or readout fails."
         "\n\n*Higher is better.* Idle decoherence is not included; the simulation after recompiling includes it.",
    "1 − F": r"**Cost** $1-F$ (mapomatic's ranking score). *Lower is better.*",
    "vs now": r"$F_{\rm layout}-F_{\rm now}$: how much this placement improves on the one the compiler chose.",
}
