"""Bias–variance bullseye: every dot is one experiment, placed by its error in two observables."""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bias–variance bullseye · QEM Explorer", page_icon=":material/adjust:", layout="wide")

from qemx import analysis as A  # noqa: E402
from qemx.bundle import make_zip  # noqa: E402
from qemx.recipes import PRESETS as RECIPE_PRESETS  # noqa: E402
from qemx.viz import analysis_charts as AC  # noqa: E402
from ui import cache as K  # noqa: E402
from ui import common as C  # noqa: E402
from ui import state as S  # noqa: E402
from ui import widgets as W  # noqa: E402
from ui.sidebar import sidebar  # noqa: E402
from ui.widgets import P  # noqa: E402

S.init()
ss = st.session_state
sidebar(run_label=None)
st.title("Bias–variance bullseye")
st.markdown("Throw darts: each dot is one complete experiment with **N** shots. Mitigation moves the cloud's "
            "center onto the bullseye (less bias) but makes the cloud bigger (more variance).")
compiled, err = C.compiled_now()
C.context_line(compiled)
if err:
    st.error("The circuit does not compile: %s" % err)

n = int(ss.n)
try:
    cat = S.observable_choices()                       # {key: label}; nothing of size 2^n is built on a rerun
except Exception as e:                                 # noqa: BLE001 - never crash the page on a huge register
    st.error("The observable list could not be built for %d qubits (%s)." % (n, W.md_escape(str(e)[:200])))
    cat = {ss.observable.key: ss.observable.label}
keys = [k for k in cat if k != ss.observable.key and cat[k] != ss.observable.label] or list(cat)
S.wdefault("w_bb_obsB", keys[0])
if ss.w_bb_obsB not in keys:
    ss.w_bb_obsB = keys[0]
S.wdefault("w_bb_rc", [r for r in ss.recipe_names if r in RECIPE_PRESETS][:5] or ["Unmitigated", "ZNE · Richardson"])
c = st.columns([1.3, 1.3, 2.4])
c[0].markdown("**x-axis:** %s (the Playground's observable)" % W.md_escape(ss.observable.label))
c[1].selectbox("y-axis observable", keys, key="w_bb_obsB", format_func=lambda k: cat[k], persist_state=P)
c[2].multiselect("Contenders", list(RECIPE_PRESETS), key="w_bb_rc", persist_state=P)


def req():
    ss.bb_run = True


st.button("Run (simulate both observables)", type="primary", icon=":material/play_arrow:", key="bb_btn", on_click=req,
          disabled=compiled is None or not ss.w_bb_rc)
obsA = ss.observable
sig = S.signature(extra=("bullseye", ss.w_bb_obsB, tuple(ss.w_bb_rc)))
if ss.pop("bb_run", False) and compiled is not None:
    okO, obsB = W.guarded("Observable B", S.observable_for, ss.w_bb_obsB)
    from qemx.recipes import preset
    recipes = tuple(preset(k, ss.symmetry) for k in ss.w_bb_rc)
    nj = K.noise_json(ss.noise)
    okA, ra = W.guarded("Observable A", K.comparison, sig + "A", compiled, nj, obsA, recipes, ss.backend, int(ss.shots),
                        ss.symmetry, n=n) if okO else (False, None)
    okB, rb = W.guarded("Observable B", K.comparison, sig + "B", compiled, nj, obsB, recipes, ss.backend, int(ss.shots),
                        ss.symmetry, n=n) if okA else (False, None)
    if okA and okB:
        ss.bb_result = dict(sig=sig, a=ra, b=rb, labels=(obsA.label, obsB.label))

res = ss.get("bb_result")
if res is None:
    st.info("Press **Run** to simulate the contenders for both observables.")
else:
    if res["sig"] != sig:
        st.warning("Settings changed since this run; press Run to refresh.")

    @st.fragment
    def darts_view():
        S.wdefault("w_bb_N_num", float(ss.shots))
        S.wdefault("w_bb_B", 150)

        def cb_n(src):
            if src == "sl":
                ss.w_bb_N_num = float(2 ** ss.w_bb_N_sl)
            else:
                ss.w_bb_N_sl = float(np.clip(np.log2(max(ss.w_bb_N_num, 1)), 6, 22))
        S.wdefault("w_bb_N_sl", float(np.clip(np.log2(max(ss.w_bb_N_num, 1)), 6, 22)))
        c = st.columns([3, 1.3, 1.3], vertical_alignment="bottom")
        c[0].slider("shots per experiment N (log₂)", 6.0, 22.0, step=0.5, key="w_bb_N_sl", on_change=cb_n,
                    args=("sl",), persist_state=P, format="2^%.1f",
                    help="Drag to the right: every cloud shrinks like 1/√N, the centers stay where they are.")
        c[1].number_input("N (exact)", min_value=1.0, step=1000.0, format="%.0f", key="w_bb_N_num", on_change=cb_n,
                          args=("num",), persist_state=P)
        c[2].number_input("experiments", min_value=10, step=50, key="w_bb_B", persist_state=P)
        N, B = int(ss.w_bb_N_num), int(ss.w_bb_B)
        ideal_a, outs_a, _ = res["a"]
        ideal_b, outs_b, _ = res["b"]

        def throw():                                   # B experiments per method: any B, inside the guard
            d_ = A.darts(outs_a, outs_b, ideal_a, ideal_b, N, B)
            return d_, {"bullseye": AC.bullseye(d_, N, labels=res["labels"], dark=C.dark()),
                        "mse_bars": AC.mse_bars(d_, N, dark=C.dark())}
        ok, out = W.guarded("The darts", throw)
        if not ok:
            return
        d, figs = out
        l, r = st.columns([1.25, 1], gap="medium")
        with l:
            W.show_fig(figs["bullseye"], "bullseye")
            W.why("bullseye")
        with r:
            W.show_fig(figs["mse_bars"], "mse_bars")
            W.why("mse_bars")
            rows = []
            for x in d:
                p = A.mse_parts(x, N)
                rows.append({"method": x["method"], "bias A": x["bias"][0], "bias B": x["bias"][1],
                             "σ A": float(np.sqrt(x["var"][0])), "σ B": float(np.sqrt(x["var"][1])),
                             "MSE": p["bias2"] + p["var"], "bias share": p["bias2"] / max(p["bias2"] + p["var"], 1e-300)})
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, width="stretch", column_config={
                k: st.column_config.NumberColumn(format="%+.4f" if k.startswith("bias ") else "%.4f")
                for k in ("bias A", "bias B", "σ A", "σ B", "MSE")} | {
                "bias share": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1)})
        dart_table = pd.concat([pd.DataFrame({"method": x["method"], "error_A": x["x"], "error_B": x["y"]}) for x in d],
                               ignore_index=True) if d else pd.DataFrame()
        state = S.to_json()
        _img = bool(ss.get("w_zip_images", True))

        def zip_bytes():
            return make_zip("Bias–variance bullseye", tables={"bias_variance": df}, data={"darts": dart_table},
                            figures={k: v for k, v in figs.items()}, state=state, images=_img,
                            notes="N = %d shots per experiment, %d experiments per method; x = %s, y = %s." % (
                                N, B, res["labels"][0], res["labels"][1]))
        st.download_button("Download all (ZIP): darts, table, figures (PNG/SVG)", zip_bytes,
                           file_name="qem_explorer_bullseye.zip", mime="application/zip", icon=":material/folder_zip:",
                           key="dl_all_bb")
    darts_view()
