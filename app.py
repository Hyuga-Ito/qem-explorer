"""QEM Explorer — landing page.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="QEM Explorer", page_icon=":material/blur_on:", layout="wide",
                   initial_sidebar_state="expanded")

from qemx import compile as CP  # noqa: E402
from qemx import templates as T  # noqa: E402
from qemx.methods import REGISTRY, TRANSFORM_INFO  # noqa: E402
from qemx.noise import NoiseSpec  # noqa: E402
from qemx.recipes import compare, preset  # noqa: E402
from qemx.viz import charts as CH  # noqa: E402
from qemx.viz import theme as TH  # noqa: E402
from ui.nav import sidebar_nav  # noqa: E402
from ui.common import dark  # noqa: E402
from ui.widgets import why  # noqa: E402

sidebar_nav()

# ------------------------------------------------------------------ hero
st.title("QEM Explorer")
st.markdown("#### Quantum error mitigation, taken apart so you can see how it works.")
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.markdown(
        r"A noisy quantum computer returns $\tilde O=\mathrm{Tr}\,[\mathcal E(\rho)\,O]$ instead of "
        r"$O=\mathrm{Tr}\,[\rho\,O]$. Error **mitigation** does not correct the device. It runs more circuits, or "
        r"the same circuit in cleverly altered versions, and combines the results into an estimate whose "
        r"**bias** is smaller than the noisy one. The price is paid in **variance**:")
    st.latex(r"\mathrm{MSE}(N)=\underbrace{\bigl(\mathbb E[\hat O]-O\bigr)^2}_{\text{bias}^2}"
             r"+\underbrace{\frac{\gamma^2\,\sigma^2}{N}}_{\text{variance}}")
    st.markdown(
        "This tool lets you build a circuit, inject exactly the noise you want, and watch every method work "
        "on it: which circuits it runs, where the state goes wrong gate by gate, what the estimate costs, and "
        "where each method breaks.")
    row = st.container(horizontal=True, gap="small")
    with row:
        st.page_link("pages/1_Playground.py", label="Open the Playground", icon=":material/arrow_forward:")
        st.page_link("pages/9_Learn.py", label="New to QEM? Start with the guides", icon=":material/school:")
        st.page_link("pages/8_Batch_Run.py", label="Batch run", icon=":material/playlist_play:")
    row2 = st.container(horizontal=True, gap="small")
    with row2:
        st.page_link("pages/10_QEM_Performance.py", label="Research: QEM performance & selection",
                     icon=":material/insights:")
        st.page_link("pages/11_Noise_Discrepancy.py", label="Research: noise models & discrepancy",
                     icon=":material/difference:")
with c2:
    with st.container(border=True):
        st.markdown("**How the Playground works**")
        st.markdown(
            "1. **Build** a circuit: drag gates and noise channels onto the wires, start from a template "
            "(QFT, QPE, Ising, Shor, HHL, Simon, …), or type it.\n"
            "2. **Inject** noise in the sidebar — or pick a fake IBM device with its real calibration — and press "
            "**Run simulation**.\n"
            "3. **Inspect** the logical and the implemented circuit side by side, and scrub through the "
            "circuit gate by gate on Bloch spheres and the density matrix.\n"
            "4. **Compare** any set of methods: estimates, error bars, distributions, overheads.\n"
            "5. **Export** a standalone Qiskit script, download everything as a ZIP, or run the same recipes on "
            "IBM Quantum hardware.")


# ------------------------------------------------------------------ live demo
@st.fragment
def demo():
    st.subheader("Ten-second demo")
    st.markdown(r"A Bell pair $\tfrac{1}{\sqrt2}(|00\rangle+|11\rangle)$ with depolarizing noise on its CX. "
                r"The ideal value is $\langle Z_0Z_1\rangle=1$. Move the error rate.")
    c1, c2 = st.columns([1, 2.2], gap="large")
    with c1:
        p = st.slider(r"two-qubit depolarizing $p_2$", 0.0, 0.3, 0.08, 0.005, key="demo_p")
        shots = st.select_slider("shots", [256, 1024, 4096, 16384], 1024, key="demo_n")
        tpl = T.bell()
        comp = CP.build(tpl.spec)
        noise = NoiseSpec(p2=p)
        names = ["Unmitigated", "ZNE · linear", "ZNE · Richardson", "PEC · inverse"]
        ideal, outs, _ = compare(comp, noise, tpl.observable, [preset(k) for k in names], shots=shots)
        rows = [dict(title=o.title, family=REGISTRY[o.recipe.estimator].family, mean=o.result.mean,
                     sd=float(np.sqrt(o.result.var1 / shots))) for o in outs]
        pec, raw = outs[-1].result, outs[0].result
        ratio = pec.var1 / raw.var1 if raw.var1 > 0 else float("nan")
        st.metric("Shots PEC needs for the same error bar", "%.2f× unmitigated" % ratio,
                  "γ² = %.2f" % pec.gamma ** 2, delta_color="off", delta_arrow="off",
                  help="Ratio of per-shot variances, Var(PEC)/Var(unmitigated). It is close to γ² when "
                       "⟨O⟩² is small next to the variance; the bound γ² is what the sampling overhead is usually "
                       "quoted as.")
    with c2:
        fig = CH.estimates(rows, ideal, outs[0].result.mean, shots, dark=dark())
        st.plotly_chart(fig, width="stretch", config=TH.PLOT_CONFIG, key="demo_fig")
    why("estimates", r"Here the noise is exactly depolarizing, so PEC's model is right and its mean is exact; "
                     r"its error bar grows with $\gamma=\frac{15/f-7}{8}$, $f=1-p_2$. ZNE's linear fit under-corrects "
                     r"the curved decay $E(\lambda)=(1-p_2)^\lambda$; Richardson follows the curve at a larger variance.")


demo()

# ------------------------------------------------------------------ method map
st.subheader("The methods, side by side")
FAMILY_ROWS = [
    ("zne", "ZNE", "Noise can be amplified in a controlled way (folding, pulse stretching).",
     r"Several circuits; variance $\times(\sum_j|\eta_j|)^2$.", "Coherent errors that folding cancels or makes oscillate."),
    ("pec", "PEC", "A correct noise model for every gate.",
     r"$\Gamma^2=\prod_g\gamma_g^2$ more shots: exponential in circuit size.", "Noise its model does not contain."),
    ("learning", "Learning (CDR / FR / NIL)", "Training circuits with Clifford angles feel the same noise as the target.",
     r"Training circuits and their shots; variance $\times\lVert\mathbf c\rVert_1^2$.",
     "Noise that depends on the rotation angles."),
    ("vd", "Virtual distillation", "Several copies of the state and an entangling derangement.",
     r"$M\cdot n$ qubits; shots $\times\,\mathrm{Tr}(\rho^M)^{-2}$.", "Coherent errors: a pure but wrong state."),
    ("symmetry", "Symmetry verification", "The ideal state has a known symmetry (e.g. a parity).",
     r"Discarded shots: $\times\,1/P_{\rm accept}$.", "Errors that preserve the symmetry."),
    ("readout", "Readout mitigation", "Readout errors are classical bit flips after the circuit.",
     r"Calibration circuits; variance $\times\,\Gamma_{\rm ro}^2$.", "Anything that happens before measurement."),
]
CARD_TEX = {
    "zne": r"\hat E(0)=\sum_j\eta_j\,E(\lambda_j)",
    "pec": r"\mathcal U=\sum_i q_i\,\mathcal B_i\circ\mathcal O",
    "learning": r"\hat O=b+\mathbf c\cdot\mathbf x_{\rm noisy}",
    "vd": r"\langle O\rangle_M=\frac{\mathrm{Tr}(\rho^MO)}{\mathrm{Tr}\,\rho^M}",
    "symmetry": r"\hat O=\frac{\sum_{S(x)=s}\hat p(x)o(x)}{\sum_{S(x)=s}\hat p(x)}",
    "readout": r"\hat{\vec p}=A^{-1}\vec p_{\rm meas}",
}
cols = st.columns(3, gap="medium")
for i, (fam, name, assumes, pays, misses) in enumerate(FAMILY_ROWS):
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown("<span style='color:%s;font-weight:600'>●</span> **%s**" % (TH.FAMILY[fam], name),
                        unsafe_allow_html=True)
            st.latex(CARD_TEX[fam])
            st.markdown("**Assumes** %s  \n**Pays** %s  \n**Misses** %s" % (assumes, pays, misses))
st.markdown("**Noise tailoring** that combines with any of the above:")
c1, c2 = st.columns(2, gap="medium")
for col, key in ((c1, "twirl"), (c2, "dd")):
    with col:
        with st.container(border=True):
            info = TRANSFORM_INFO[key]
            st.markdown("**%s**" % info["label"])
            st.latex(info["tex"])
            st.markdown(info["why"])

# ------------------------------------------------------------------ the pages
st.subheader("What's inside")
PAGES = [
    ("pages/1_Playground.py", "Playground", ":material/science:",
     "Drag-and-drop composer, noise injector or fake device, logical vs implemented circuits, gate-by-gate time "
     "travel on Bloch spheres, head-to-head comparison of any methods, export."),
    ("pages/2_Cost_vs_Precision.py", "Cost vs precision", ":material/trending_up:",
     "RMSE against shots and QPU time, where mitigation starts to pay, and the PEC exponential wall."),
    ("pages/3_Bias_Variance.py", "Bias–variance bullseye", ":material/adjust:",
     "Repeated experiments as darts on a target: bias is where the cloud sits, variance how wide it is."),
    ("pages/4_Model_Mismatch.py", "Model mismatch", ":material/tune:",
     "Assumed vs true error rates: PEC under- and over-correction, readout drift, unmodeled coherent noise."),
    ("pages/5_Head_to_Head.py", "Head-to-head sweeps", ":material/compare_arrows:",
     "Any set of methods across a noise knob, and a map of who wins at every shot budget."),
    ("pages/6_Theory.py", "Theory & sources", ":material/menu_book:",
     "Every formula the app uses, derived and referenced, with calculators."),
    ("pages/7_IBM_Quantum.py", "Run on IBM Quantum", ":material/cloud_upload:",
     "The same recipes on real hardware (or rehearsed locally on a fake backend), with secure token handling."),
]
cols = st.columns(2, gap="medium")
for i, (path, label, icon, text) in enumerate(PAGES):
    with cols[i % 2]:
        with st.container(border=True):
            st.page_link(path, label=label, icon=icon)
            st.caption(text)

st.caption("Conventions: qubit 0 is the least significant bit (Qiskit order). $p$ is a depolarizing probability "
           r"($\rho\mapsto(1-p)\rho+p\,I/d$); the datasheet number is the average gate infidelity "
           r"$r=\tfrac{d-1}{d}\,p$ for purely depolarizing noise. All noise presets are illustrative.")
