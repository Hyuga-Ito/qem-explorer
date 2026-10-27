"""Everything shown for one comparison result: figures, the analysis views, the equal-budget cost table, each
method's internals and the "Download all" ZIP. Shared by the Playground (simulation) and Run on IBM Quantum (local
rehearsal or hardware), so both produce the same suite of figures, tables and files."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from qemx import export as EX
from qemx import insight as IN
from qemx import report as RP
from qemx.bundle import make_zip
from qemx.methods import REGISTRY
from qemx.recipes import HIDDEN, measured_view
from qemx.viz import charts as CH
from qemx.viz import composer as CV
from qemx.viz import theme as TH

from . import metrics as MX
from . import state as S
from . import widgets as W
from .common import dark
from .widgets import P

ss = st.session_state
FULL_PX = 1150


def _show(kp, fig, key, data=None):
    W.show_fig(fig, kp + key, data)


def _show_if(kp, fig, key, data=None):
    """A method's figure, or a short note when it could not be drawn (never a KeyError)."""
    if fig is None:
        st.caption("This figure could not be drawn for this result.")
    else:
        _show(kp, fig, key, data)


def _n(res) -> int:
    d = res.get("dist") or {}
    return int(d.get("n") or res.get("n") or ss.n)


def results_figures(res, dark_=False):
    rows, ideal, N = res["rows"], res["ideal"], res["shots"]
    raw = next((r for r in rows if r["method"] == "Unmitigated"), None)
    est_rows = [dict(title=r["method"], family=r["family"], mean=r["estimate"], sd=r["sd"]) for r in rows]
    figs = {"estimates": CH.estimates(est_rows, ideal, raw["estimate"] if raw else None, N, dark=dark_)}
    samples = {k: v for k, v in res["samples"].items() if v is not None}
    if samples:
        figs["distribution"] = CH.distribution_map(samples, {r["method"]: r["family"] for r in rows}, ideal, N,
                                                   dark=dark_)
    for o in res["outs"]:
        if o.result is None or not o.result.details:
            continue
        d = o.result.details
        try:
            if o.recipe.estimator == "zne":
                figs["zne_fit " + o.title] = CH.zne_fit(d, ideal, dark=dark_)
            elif o.recipe.estimator == "pec":
                figs["pec_gamma " + o.title] = CH.gamma_growth(d["gamma_path"], dark=dark_)
                figs["pec_quasi " + o.title] = CH.quasi_bars(d["quasi"], dark=dark_)
            elif o.recipe.estimator == "learning" and not d.get("degenerate"):
                figs["learning " + o.title] = CH.learning_scatter(d, ideal, o.result.mean, dark=dark_)
            elif o.recipe.estimator == "nil" and not d.get("degenerate"):
                figs["nil_fit " + o.title] = CH.nil_fit(d, ideal, o.result.mean, dark=dark_)
                figs["nil_coef " + o.title] = CH.nil_coefficients(d, dark=dark_)
            elif o.recipe.estimator == "vd":
                figs["vd_spectrum " + o.title] = CH.spectrum(d["eigs"], d["M"], dark=dark_)
            elif o.recipe.estimator == "chain" and not d.get("degenerate"):
                if d.get("outer") == "zne":
                    figs["zne_fit " + o.title] = CH.zne_fit(d, ideal, dark=dark_)
                elif "coef" in d:
                    figs["nil_fit " + o.title] = CH.nil_fit(d, ideal, o.result.mean, dark=dark_)
                    figs["nil_coef " + o.title] = CH.nil_coefficients(d, dark=dark_)
        except Exception:
            pass
    try:                                              # the analysis views' figures travel with the ZIP too
        figs.update(analysis_figures(res, dark_))
    except Exception:
        pass
    return figs


def dist_payload(res):
    """(distribution series marginalized onto the bits the observable reads, bits, support) or None."""
    d = res.get("dist")
    if not d or not d.get("series"):
        return None
    n = d["n"]
    bits = list(range(n))
    support = None
    if d.get("values") is not None:
        rb = IN.relevant_bits(d["values"], n)
        if rb and len(rb) < n:
            bits = rb
        v = np.asarray(d["values"])
        rep = np.zeros(2 ** len(bits), dtype=int)                 # one full outcome per marginal outcome
        for k, b in enumerate(bits):
            rep |= ((np.arange(2 ** len(bits)) >> k) & 1) << b
        vm = v[rep]
        if IN.value_kind(v) == "projector":
            support = np.nonzero(vm > 0.5)[0]
    series = {k: IN.marginal(p, bits, n) if len(bits) < n else np.asarray(p) for k, p in d["series"].items()}
    return series, bits, support


def qec_rows(res):
    d = res.get("dist") or {}
    ref = d.get("intended") if d.get("intended") is not None else res["ideal"]
    return [dict(method=r["method"], family=r["family"], pL=IN.logical_error(r["estimate"], ref)) for r in res["rows"]]


def analysis_figures(res, dark_=False) -> dict:
    figs = {}
    pl = dist_payload(res)
    if pl is not None:
        series, bits, support = pl
        figs["output_distribution"] = CH.output_histogram(series, len(bits), support, dark=dark_, bits=bits)
    d = res.get("dist") or {}
    if d.get("checks") and d.get("series"):
        ch = d["checks"]
        syn = {k: IN.syndrome_distribution(p, ch, d["n"]) for k, p in d["series"].items()
               if k in ("ideal", "noisy (unmitigated)")}
        figs["syndromes"] = CH.syndrome_histogram(syn, [c[0] for c in ch], dark=dark_)
        figs["logical_error"] = CH.logical_error_bars(qec_rows(res), dark=dark_)
    if d.get("readout"):
        figs["readout_assignment"] = CH.readout_bars(d["readout"], dark=dark_)
    return figs


def estimators_of(outs) -> list:
    out = []
    for o in outs:
        if o.result is None:
            continue
        out.append(o.recipe.estimator)
        if o.recipe.readout != "none":
            out.append("readout")
    return out


def analysis_panel(res, figs, kp=""):
    """One analysis view, chosen automatically from the circuit, the observable and the methods — or by hand."""
    if not res["rows"]:
        st.error("Every method failed on this circuit, so there is nothing to analyze; the reasons are listed under "
                 "*Inside each method* below.", icon=":material/error:")
        return
    snap = (res.get("dist") or {}).get("snapshot")             # read the result as it was run, not the live state
    if snap:
        prof = IN.profile(snap["template"], snap["params"], snap["custom"], res.get("observable", ss.observable),
                          estimators_of(res["outs"]), int(res["dist"]["n"]), snap["logical"])
    else:
        prof = IN.profile(ss.get("template"), ss.get("tpl_params"), S.is_custom(), res.get("observable", ss.observable),
                          estimators_of(res["outs"]), _n(res))
    st.subheader("Analysis")
    options = ["auto"] + prof.order
    vk = kp + "w_an_view"                         # one view choice per result view (Playground, local run, hardware)
    S.wdefault(vk, "auto")
    if ss[vk] not in options:
        ss[vk] = "auto"
    c = st.columns([1.25, 2.4], vertical_alignment="center")
    c[0].selectbox("View", options, key=vk, persist_state=P,
                   format_func=lambda k: ("Auto · %s" % IN.VIEWS[prof.primary]) if k == "auto" else IN.VIEWS[k],
                   help="Auto picks the picture that matches the result's mathematics: a histogram for "
                        "probabilities, estimates ± σ for expectation values, logical error and syndromes for codes. "
                        "Choose another view to override it.")
    view = prof.primary if ss[vk] == "auto" else ss[vk]
    if ss[vk] == "auto":
        why_txt = "; ".join(prof.reasons)
        c[1].caption(":material/auto_awesome: " + W.md_escape(why_txt[:1].upper() + why_txt[1:]) + ".")
    else:
        c[1].caption(":material/tune: Manual view. Auto would show *%s*." % IN.VIEWS[prof.primary])
    if view == "distribution":
        view_distribution(res, figs, kp)
    elif view == "qec":
        view_qec(res, figs, kp)
    elif view == "diagnostics":
        view_diagnostics(res, figs, kp)
    else:
        view_expectation(res, figs, kp)


def view_expectation(res, figs, kp=""):
    rows = res["rows"]
    _show(kp, figs["estimates"], "estimates", pd.DataFrame(rows))
    W.why("estimates")
    samples = {k: v for k, v in res["samples"].items() if v is not None}
    if "distribution" in figs:
        st.markdown("**Spread of repeated experiments**")
        _show(kp, figs["distribution"], "distribution", pd.DataFrame({k: pd.Series(v) for k, v in samples.items()}))
        bad = {k: int(np.sum(~np.isfinite(np.asarray(v, float)))) for k, v in samples.items()}
        bad = {k: v for k, v in bad.items() if v}
        if bad:
            st.caption("Left out of the plot: " + "; ".join(
                "%s failed in %d of %d repeats (e.g. an exponential fit when E(λ) changes sign)"
                % (k, v, len(samples[k])) for k, v in bad.items()))
        W.why("distribution")


def view_distribution(res, figs, kp=""):
    d = res.get("dist") or {}
    pl = dist_payload(res)
    if pl is None:
        st.info(" ".join(d.get("notes", [])) or "Press **Run simulation** to compute the output distributions.",
                icon=":material/bar_chart:")
        return
    series, bits, support = pl
    ideal = series["ideal"]
    from qemx import discrepancy as DS
    m = st.columns(4)
    p_noisy = np.asarray(series["noisy (unmitigated)"], float)
    p_ideal = np.clip(np.asarray(ideal, float), 0, None)
    p_ideal = p_ideal / p_ideal.sum()                    # the exact ideal distribution (round-off only)
    try:
        tv, fc = DS.tvd(p_noisy, p_ideal), DS.classical_fidelity(p_noisy, p_ideal)
    except ValueError:                                   # not a probability distribution: say so, never clip
        tv = fc = float("nan")
    m[0].metric("TVD noisy ↔ ideal", "%.4f" % tv, border=True,
                help=r"Total variation distance $\tfrac12\sum_x|p(x)-q(x)|$ between two probability distributions: "
                     r"0 = identical, 1 = disjoint. Exact distributions here (no shot noise).")
    m[1].metric("Classical fidelity", "%.4f" % fc, border=True,
                help=r"$F_c=\bigl(\sum_x\sqrt{p(x)\,q(x)}\bigr)^2$ between noisy and ideal (Chen et al. Eq. 1; no "
                     r"normalization to the uniform distribution). A distribution overlap, not the quantum state "
                     r"fidelity.")
    best = [k for k in ("ZNE-extrapolated", "readout-corrected") if k in series]
    if support is not None:
        pc = {k: float(np.sum(v[support])) for k, v in series.items()}
        m[2].metric("P(counted) noisy", "%.4f" % pc["noisy (unmitigated)"], "ideal %.4f" % pc["ideal"],
                    delta_color="off", delta_arrow="off", border=True,
                    help="Probability of the outcomes the observable counts (shaded).")
        if best:
            m[3].metric("P(counted) %s" % best[0].split("-")[0], "%.4f" % pc[best[0]], border=True,
                        help="The same sum over the mitigated distribution: equal to that method's estimate.")
    elif best:
        q_ = np.asarray(series[best[0]], float)
        signed = bool((q_ < -DS.PROB_TOL).any() or abs(q_.sum() - 1) > DS.PROB_TOL * max(1, q_.size))
        m[2].metric(("ℓ1 %s ↔ ideal" if signed else "TVD %s ↔ ideal") % best[0].split("-")[0],
                    "%.4f" % (DS.l1_signed(q_, p_ideal) if signed else DS.tvd(q_, p_ideal)), border=True,
                    help="The mitigated vector has negative entries: it is a quasi-probability, not a distribution, so "
                         "this is the ℓ1 distance ½Σ|q − p| (it can exceed 1), not a TVD. Nothing was clipped."
                    if signed else "Total variation distance of the mitigated distribution to the ideal one.")
    _show(kp, figs.get("output_distribution") or CH.output_histogram(series, len(bits), support, dark=dark(),
                                                                       bits=bits), "output_distribution",
               pd.DataFrame({"outcome": [format(i, "0%db" % len(bits)) for i in range(2 ** len(bits))],
                             **{k: v for k, v in series.items()}}))
    notes = list(d.get("notes", []))
    if len(bits) < d["n"]:
        notes.insert(0, "Marginal over the bits the observable reads (%s); the other qubits are summed over." %
                     " ".join("q%d" % q for q in reversed(bits)))
    if d.get("hamiltonian"):
        notes.insert(0, "The observable is a Hamiltonian: this is the computational-basis (Z) output; each measurement "
                        "setting rotates some qubits first.")
    st.caption("  \n".join(W.md_escape(t) for t in notes))
    W.why("distribution_view")


def _short(name: str, k: int = 30) -> str:
    """A recipe name for a metric chip: whole if it fits, otherwise cut at a word with an ellipsis."""
    return name if len(name) <= k else name[:k - 1].rsplit(" ", 1)[0] + " …"


def view_qec(res, figs, kp=""):
    d = res.get("dist") or {}
    rows = qec_rows(res)
    raw = next((r for r in rows if r["method"] == "Unmitigated"), None)
    fin = [r for r in rows if np.isfinite(r["pL"])]
    m = st.columns(3)
    ref = d.get("intended")
    if ref is not None:
        m[0].metric("⟨O_L⟩ intended", "%.4f" % ref, border=True,
                    help="The error-free value of the decoded logical observable (cos θ for an encoded RY(θ)|0⟩): the "
                         "value the code is meant to protect. p_L is measured against it.")
    else:
        m[0].metric("⟨Z_L⟩ ideal", "%.4f" % res["ideal"], border=True)
    if raw and np.isfinite(raw["pL"]):
        m[1].metric("p_L unmitigated", "%.4f" % raw["pL"], border=True,
                    help=r"$p_L=\tfrac12(1-\langle Z_L\rangle/\langle Z_L\rangle_{\rm ideal})$")
    if fin:
        b = min(fin, key=lambda r: abs(r["pL"]))
        m[2].metric("closest to p_L = 0", "%.4f" % b["pL"], _short(b["method"]), delta_color="off", delta_arrow="off",
                    border=True, help="Mitigated estimates can overshoot (negative p_L): the metric is |p_L|.")
    else:
        st.caption("The intended ⟨Z_L⟩ is close to 0 (θ ≈ π/2), so the logical error cannot be read from it; choose "
                   "another θ in the template.")
    if ref is not None and abs(res["ideal"] - ref) > 1e-6:
        st.caption("With the injected error the noiseless circuit gives ⟨Z_L⟩ = %.4f: the code %s it." % (
            res["ideal"], "did not undo" if abs(res["ideal"] - ref) > 1e-3 else "almost undid"))
    left, right = st.columns([1, 1.15], gap="medium")
    with left:
        st.markdown("**Effective logical error per method**")
        if fin:
            _show(kp, figs.get("logical_error") or CH.logical_error_bars(rows, dark=dark()), "logical_error",
                       pd.DataFrame(rows))
    with right:
        st.markdown("**Syndromes**")
        if "syndromes" in figs:
            _show(kp, figs["syndromes"], "syndromes")
            ch = d["checks"]
            p0 = float(IN.syndrome_distribution(d["series"]["noisy (unmitigated)"], ch, d["n"])[0])
            st.caption("No check fired in %.1f %% of the noisy shots — the shots a pure error-detection scheme "
                       "would keep." % (100 * p0))
        else:
            st.info("Syndromes need the distributions of this run; press **Run simulation**.")
    W.why("qec_view")


def view_diagnostics(res, figs, kp=""):
    """A card per method with its characteristic curve (or number) and one line on how to read it."""
    cards, none, ro_done = [], [], False
    for o in res["outs"]:
        if o.result is None:
            continue
        e, t = o.recipe.estimator, o.title
        if e == "zne" and ("zne_fit " + t) in figs:
            cards.append(dict(title=t, figs=[figs["zne_fit " + t]], tip="Measured E(λ) and the fit continued to "
                                                                        "λ = 0; the estimate is where it lands."))
        elif e == "pec" and ("pec_quasi " + t) in figs:
            cards.append(dict(title=t, figs=[figs["pec_quasi " + t], figs["pec_gamma " + t]],
                              tip="Signed quasi-probabilities and γ along the circuit: γ² is the shot cost."))
        elif e == "learning" and ("learning " + t) in figs:
            cards.append(dict(title=t, figs=[figs["learning " + t]],
                              tip="Training circuits: noisy against ideal values, and the fitted map."))
        elif e == "nil" and ("nil_fit " + t) in figs:
            cards.append(dict(title=t, figs=[figs["nil_fit " + t], figs["nil_coef " + t]],
                              tip="Training circuits: fitted against ideal values; the coefficient of every "
                                  "neighbor circuit (γ = Σ|c| is the shot-cost factor)."))
        elif e == "vd" and ("vd_spectrum " + t) in figs:
            cards.append(dict(title=t, figs=[figs["vd_spectrum " + t]],
                              tip="Eigenvalues of ρ: distillation weights the dominant one."))
        elif e == "symmetry":
            a = float(o.result.accept)
            cards.append(dict(title=t, figs=[], accept=a, tip="Accepted fraction P_accept = %.3f: shots cost "
                                                               "×%.2f." % (a, 1 / max(a, 1e-12))))
        elif o.recipe.readout != "none" and "readout_assignment" in figs and not ro_done:
            ro_done = True
            cards.append(dict(title=t, figs=[figs["readout_assignment"]],
                              tip="Calibrated assignment errors per bit: what the readout correction inverts."))
        else:
            none.append(t)
    if not cards:
        st.info("None of these methods has a diagnostic curve: unmitigated runs and pure noise tailoring (twirling, "
                "DD) change only the circuit — see the Circuits tab.", icon=":material/insights:")
    for k in range(0, len(cards), 2):
        cols = st.columns(2, gap="medium")
        for j, (col, card) in enumerate(zip(cols, cards[k:k + 2])):
            with col.container(border=True):
                st.markdown("**%s**" % W.md_escape(card["title"]))
                if "accept" in card:
                    st.progress(min(max(card["accept"], 0.0), 1.0))
                for i, f in enumerate(card["figs"]):
                    _show(kp, f, "diag_%d_%d" % (k + j, i))
                st.caption(W.md_escape(card["tip"]))
    if none:
        st.caption("No diagnostic curve: " + ", ".join(W.md_escape(x) for x in none) + ".")
    W.why("diagnostics_view")


def render(res, kp: str = "", stale: bool = False, bundle=None, zip_name: str = "qem_explorer.zip"):
    """The full result view of a comparison ``res`` (as stored by the Playground or the IBM Quantum page)."""
    if stale:
        st.warning("Inputs changed since these results were computed; press Run simulation to refresh.")
    rows, ideal, N = res["rows"], res["ideal"], res["shots"]
    raw = next((r for r in rows if r["method"] == "Unmitigated"), None)
    best = min((r for r in rows if np.isfinite(r["RMSE"])), key=lambda r: r["RMSE"], default=None)
    m = st.columns(4)
    m[0].metric("Ideal ⟨O⟩", "%.4f" % ideal if ideal is not None else "–", border=True)
    if raw:
        m[1].metric("Unmitigated", "%.4f" % raw["estimate"], "bias %+.4f" % raw["bias"], delta_color="off",
                    delta_arrow="off", border=True)
    if best:
        m[2].metric("Lowest RMSE at %s shots" % f"{N:,}", best["method"], border=True,
                    help=MX.help_for("RMSE"))
        m[3].metric("its RMSE", "%.4f" % best["RMSE"], "vs %.4f unmitigated" % raw["RMSE"] if raw else None,
                    delta_color="off", delta_arrow="off", border=True,
                    help=r"$\mathrm{RMSE}=\sqrt{b^2+\sigma^2}$ at the sidebar's shot budget.")
    if bundle is not None:
        st.download_button("Download all (ZIP)", bundle, file_name=zip_name, mime="application/zip",
                           icon=":material/folder_zip:", key=kp + "dl_all", type="primary",
                           help="Every table and figure this result supports, whichever page shows it: the comparison "
                                "and its analysis views, cost and precision (RMSE against budget, budget for ε, "
                                "crossovers), the PEC wall, bias–variance, plus the repeated-experiment samples, the "
                                "state and the Qiskit script. Figures are PNG (3×) and SVG.")
    figs = results_figures(res, dark())
    flagged = [(o.title, o.result.details) for o in res["outs"] if o.result is not None and o.result.details
               and o.result.details.get("warning")]
    for t_, d_ in flagged:
        st.warning("**%s:** %s" % (W.md_escape(t_), W.md_escape(d_["warning"])),
                   icon=":material/warning:" if not d_.get("degenerate") else ":material/block:")
    analysis_panel(res, figs, kp)
    if rows:
        st.subheader("Cost and accuracy")
        df = pd.DataFrame(rows).drop(columns=[c for c in HIDDEN if c in pd.DataFrame(rows).columns])
        if not res.get("hardware"):
            ran = df["estimate kind"] != "not run" if "estimate kind" in df else True
            df["QPU s"] = np.where(ran, N * float(ss.t_shot_us) * 1e-6, np.nan)   # the same budget N for every method run
        for c_ in ("training shots", "calibration shots"):
            if c_ in df and not df[c_].any():
                df = df.drop(columns=[c_])
        if not df["failed repeats"].any():
            df = df.drop(columns=["failed repeats"])
        # contract columns only where they say something: the exact-fit reference and its MC error (learning methods),
        # the estimate kind when not every row is exact, and approximation flags
        if "exact-fit value" in df and not df["exact-fit value"].notna().any():
            df = df.drop(columns=["exact-fit value"])
        if "bias MC ±" in df and not (df["bias MC ±"].fillna(0) > 0).any():
            df = df.drop(columns=["bias MC ±"])
        if "estimate kind" in df and set(df["estimate kind"]) <= {"exact"}:
            df = df.drop(columns=["estimate kind"])
        if "flags" in df and not df["flags"].astype(bool).any():
            df = df.drop(columns=["flags"])
        for c_ in ("fit failures", "failure policy", "MC experiments", "signed error to reference", "absolute error"):
            if c_ in df and not df[c_].notna().any():
                df = df.drop(columns=[c_])
        df = measured_view(df)              # one hardware experiment: signed / absolute error, not bias / RMSE
        if "status" in df:
            if not df["status"].astype(bool).any():
                df = df.drop(columns=["status"])
            else:                                # flags next to the method name, where they are seen
                df = df[["method", "status"] + [c for c in df.columns if c not in ("method", "status")]]
        st.caption(("Every method spends the same total budget of **%s shots** per estimate — target, training "
                    "and calibration circuits together — so the table is an equal-cost comparison." % f"{N:,}")
                   + (" On hardware the columns *target / training / calibration shots* and *circuits* are what was "
                      "actually sent; *sd* is the standard error of that allocation." if res.get("hardware") else ""))
        MX.legend(("signed error to reference", "bootstrap sd", "absolute error", "γ", "circuits", "target shots",
                   "QPU s") if "absolute error" in df else ("bias", "sd", "RMSE", "γ", "circuits", "accepted",
                                                            "target shots", "QPU s"))
        st.dataframe(df, hide_index=True, width="stretch", column_config=MX.column_config(df.columns))
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Table (CSV)", df.to_csv(index=False), "qem_comparison.csv", "text/csv", key=kp + "tbl_csv")
        d2.download_button("Table (LaTeX)", df.to_latex(index=False, float_format="%.4f"), "qem_comparison.tex",
                           "text/plain", key=kp + "tbl_tex")
        W.why("table")
    st.subheader("Inside each method")
    for i_o, o in enumerate(res["outs"]):
        if o.result is None and getattr(o, "status", "") == "insufficient_budget":
            st.info("**%s was not run.** %s Raise the shot budget in the sidebar, or reduce its training circuits, "
                    "feature circuits or training shots (e.g. the *NIL · small budget* preset)."
                    % (W.md_escape(o.title), W.md_escape(o.error)), icon=":material/savings:")
            continue
        if o.result is None:
            st.error("%s failed: %s" % (W.md_escape(o.title), W.md_escape(o.error)))
            continue
        r = o.result
        with st.expander(W.md_escape(o.title), icon=":material/biotech:", key=kp + "exp_in_%d" % i_o):
            st.latex(REGISTRY[o.recipe.estimator].tex)
            for note in r.notes:
                st.caption(W.md_escape(note))
            d = r.details
            if d and d.get("hamiltonian"):
                st.dataframe(pd.DataFrame(d["hamiltonian"]).rename(columns={
                    "basis": "setting (q%d…q0)" % (_n(res) - 1), "terms": "terms", "mean": "Σ c⟨P⟩",
                    "var1": "per-shot variance", "shots": "shots (of %s)" % f"{int(N):,}"}),
                    hide_index=True, width="stretch")
            if o.recipe.estimator == "zne" and d:
                _show_if(kp, figs.get("zne_fit " + o.title), "zf_%d" % i_o)
                st.caption("Weights η = %s; optimal shot split %s." % (
                    ", ".join("%+.3f" % e for e in d["eta"]), ", ".join("%.0f %%" % (100 * a) for a in d["alloc"])))
                W.why("zne_fit")
            elif o.recipe.estimator == "pec" and d:
                a, b = st.columns(2)
                with a:
                    _show_if(kp, figs.get("pec_gamma " + o.title), "pg_%d" % i_o)
                with b:
                    _show_if(kp, figs.get("pec_quasi " + o.title), "pq_%d" % i_o)
                W.why("pec_gamma")
                W.why("pec_quasi")
            elif o.recipe.estimator == "learning" and d:
                if d.get("warning"):
                    st.warning(W.md_escape(d["warning"]), icon=":material/warning:")
                st.caption("Training set: %s · spread of the ideal training values %.3g (needed %.3g) · "
                           "feature rank %d of %d." % (W.md_escape(d.get("sampling", "–")), d.get("spread", 0),
                                                       d.get("spread_needed", 0), d.get("rank", 0),
                                                       d.get("n_features", 1)))
                if ("learning " + o.title) in figs:
                    _show(kp, figs["learning " + o.title], "ls_%d" % i_o)
                    st.caption("Coefficients: " + ", ".join("%s %+.3f" % (nm, c) for nm, c in
                                                            zip(d["names"], d["coef"]))
                               + "; intercept %+.4f." % d["intercept"])
                else:
                    st.info("No regression plot: the training data are degenerate, so there is no fit to show.")
                W.why("learning")
            elif o.recipe.estimator == "nil" and d:
                if d.get("warning"):
                    st.warning(W.md_escape(d["warning"]), icon=":material/warning:")
                if not d.get("degenerate"):
                    st.caption("%d neighbor circuits × %d training circuits (%s angles) · %s · %d coefficients ≠ 0 · "
                               "γ = Σ|c| = %.3f · feature directions resolved above shot noise: %d of %d%s." % (
                                   d.get("n_features", 0), d.get("n_train", 0), W.md_escape(d.get("ensemble", "–")),
                                   W.md_escape(d.get("model", "–")), d.get("nonzero", 0), d.get("l1", 0.0),
                                   d.get("rank", 0), d.get("n_features", 0),
                                   "" if d.get("solver_ok", True) else " · optimality not certified"))
                    a, b = st.columns(2)
                    with a:
                        _show_if(kp, figs.get("nil_fit " + o.title), "nf_%d" % i_o)
                    with b:
                        _show_if(kp, figs.get("nil_coef " + o.title), "nc_%d" % i_o)
                else:
                    st.info("No fit to show: the training data are degenerate (unmitigated value shown).")
                W.why("nil")
            elif o.recipe.estimator == "vd" and d:
                _show_if(kp, figs.get("vd_spectrum " + o.title), "vs_%d" % i_o)
                st.caption("Dominant eigenvector gives ⟨O⟩ = %.4f: the limit M → ∞." % d["dominant_value"])
                W.why("spectrum")
            elif o.recipe.estimator == "symmetry" and d:
                st.markdown("Accepted fraction $P_{\\rm accept}$ = **%.3f**" % d["accept"])
                W.why("symmetry")
            elif o.recipe.estimator == "chain" and d:
                st.caption("Outer %s over inner %s on every circuit it runs (simulation only)." % (
                    W.md_escape(str(d.get("outer"))), W.md_escape(str(d.get("inner")))))
                if d.get("outer") == "zne":
                    _show_if(kp, figs.get("zne_fit " + o.title), "czf_%d" % i_o)
                elif not d.get("degenerate"):
                    a, b = st.columns(2)
                    with a:
                        _show_if(kp, figs.get("nil_fit " + o.title), "cnf_%d" % i_o)
                    with b:
                        _show_if(kp, figs.get("nil_coef " + o.title), "cnc_%d" % i_o)
                W.why("chain")
            for title, sp in r.implemented[:1]:
                if sp is not None:
                    st.markdown("**%s**" % title)
                    fig = CV.draw(sp, dark=dark(), width_px=FULL_PX - 40)
                    fig.update_layout(uirevision=kp + "ic_%d" % i_o)
                    st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key=kp + "ic_%d" % i_o)


def zip_params() -> dict:
    """The Cost-vs-precision page's settings (or their defaults), captured for a download thread."""
    g = ss.get
    return dict(eps=g("w_cvp_eps"), budget=g("w_cvp_budget"), nmin=g("w_cvp_nmin"), nmax=g("w_cvp_nmax"),
                nq=g("w_wall_n"), p1=g("w_wall_p1"), p2s=g("w_wall_p2s"), eps_w=g("w_wall_eps"), L=g("w_wall_L"),
                con=g("w_wall_con"), bud=g("w_wall_budget"), t=float(ss.t_shot_us))


def page_extras_inputs() -> dict:
    """Other pages' stored results and their settings, captured by reference in the script thread (cheap); the
    tables and figures are built from them only when a ZIP is downloaded (page_extras)."""
    g = ss.get
    return dict(batch=g("batch_result"), batch_N=g("w_b_N", 10000), ds=g("ds_result"), ds_N=g("w_ds_N", 10000),
                bb=g("bb_result"), bb_N=g("w_bb_N_num", ss.shots), bb_B=g("w_bb_B", 150), shots=ss.shots,
                hh=g("hh_result"), mm=g("mm_result"), mm_un=g("mm_un"))


def page_extras(inp: dict | None = None) -> dict:
    """{part: (figures, tables)} from other pages' stored results (``inp`` from page_extras_inputs)."""
    from qemx import analysis as A
    from qemx import batch as B
    from qemx.viz import analysis_charts as AC
    inp = page_extras_inputs() if inp is None else inp
    out = {}
    try:
        b = inp.get("batch")
        if b:
            N = int(inp.get("batch_N") or 10000)
            out["batch"] = ({}, {"batch_results": pd.DataFrame(b["rows"]),
                                 "batch_winners": pd.DataFrame(B.winners(list(b["rows"]), N))})
    except Exception:                                       # noqa: BLE001
        pass
    try:
        ds = inp.get("ds")
        if ds:
            N = int(inp.get("ds_N") or 10000)
            out["depth_sweep"] = ({"depth_sweep": AC.depth_chart(ds["rows"], N)},
                                  {"depth_sweep": pd.DataFrame(ds["rows"])})
    except Exception:                                       # noqa: BLE001
        pass
    try:
        bb = inp.get("bb")
        if bb:
            N, Bn = int(inp.get("bb_N") or inp["shots"]), int(inp.get("bb_B") or 150)
            d = A.darts(bb["a"][1], bb["b"][1], bb["a"][0], bb["b"][0], N, Bn)
            out["bullseye"] = ({"bullseye": AC.bullseye(d, N, labels=bb["labels"]), "mse_bars": AC.mse_bars(d, N)},
                               {"darts": pd.concat([pd.DataFrame({"method": x["method"], "error_A": x["x"],
                                                                  "error_B": x["y"]}) for x in d], ignore_index=True)
                                if d else pd.DataFrame()})
    except Exception:                                       # noqa: BLE001
        pass
    for key, name in (("hh", "head_to_head"), ("mm", "model_mismatch"), ("mm_un", "unmodelled_noise")):
        r = inp.get(key)
        rows = r.get("rows") if isinstance(r, dict) else r
        if isinstance(rows, list) and rows:
            try:
                out[name] = ({}, {name: pd.DataFrame(rows)})
            except Exception:                               # noqa: BLE001
                pass
    return out


def script_values(res, state) -> list | None:
    """The estimates the exported script should reproduce, one per contender of ``state`` — only when the result
    is a simulation of exactly those contenders (a hardware result is not reproducible by a simulation)."""
    from qemx.methods import Recipe
    if res.get("hardware"):
        return None
    recs = state.get("recipes") or []
    outs = res.get("outs") or []
    if len(outs) != len(recs) or any(o.recipe.key() != Recipe.from_json(d).key() for o, d in zip(outs, recs)):
        return None
    return [o.result.mean if o.result is not None else None for o in outs]


def bundle_factory(res, state, source_label, title="Playground comparison", extra_tables=None):
    """A download callable that uses only values captured here (Streamlit runs it in a separate thread). The ZIP
    holds this comparison and every cross-page result derived from it (cost and precision, PEC wall,
    bias–variance), plus other pages' stored results."""
    params = zip_params()
    images = bool(st.session_state.get("w_zip_images", True))   # captured now (the ZIP runs in a thread)
    extras_in = page_extras_inputs()               # references only; the work happens when the ZIP is requested
    extra_tables = dict(extra_tables or {})

    def build() -> bytes:
        extras = page_extras(extras_in)
        figs = {"comparison/" + k: v for k, v in results_figures(res, False).items()}
        df = pd.DataFrame(res["rows"])
        tables = {}
        if len(df):
            df = measured_view(df)                       # measured experiments: signed / absolute error columns
            tables["comparison/cost_table"] = df
            tables["comparison/estimates"] = df[[c for c in ("method", "estimate", "bias", "sd", "RMSE",
                                                             "signed error to reference", "bootstrap sd",
                                                             "absolute error") if c in df.columns]]
        tables.update({"comparison/" + k: v for k, v in extra_tables.items()})
        f2, t2 = RP.cross_tab(res, params, extras)
        figs.update(f2)
        tables.update(t2)
        samples = {k: v for k, v in res["samples"].items() if v is not None}
        try:                                            # the run's own state and estimates, in contender order
            script = EX.script(state, script_values(res, state))
        except Exception:                                   # noqa: BLE001
            script = None
        return make_zip(title, tables=tables,
                        data={"repeated_experiments": pd.DataFrame({k: pd.Series(v) for k, v in samples.items()})},
                        figures=figs, state=state, script=script, images=images,
                        notes="Ideal ⟨O⟩ = %.6f; budget %d shots per method; %s. Folders: comparison/ (this result), "
                              "cost_precision/, pec_wall/, bias_variance/ and the other pages' stored results." % (
                                  res["ideal"], res["shots"], source_label))
    return build
