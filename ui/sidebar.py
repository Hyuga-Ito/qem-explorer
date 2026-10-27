"""The shared sidebar: navigation, simulator (NumPy, Aer or a fake IBM device), shots, time limit, and the
noise injector. Every control is a persistent widget whose callback updates the model, so settings stay
exactly as the user left them on every page."""
from __future__ import annotations

import streamlit as st

from qemx import device
from qemx.backends.base import BACKENDS
from qemx.noise import PRESETS

from . import state as S
from .nav import sidebar_nav
from .widgets import P, num_slider

NOISE_UI = [
    # (field, label, slider max, step, group)
    ("p1", r"$p_1$ · depolarizing, one-qubit gates", 0.02, 1e-4, "Gate errors"),
    ("p2", r"$p_2$ · depolarizing, two-qubit gates", 0.10, 5e-4, "Gate errors"),
    ("amp_damp", r"$\gamma$ · amplitude damping per gate", 0.10, 5e-4, "Gate errors"),
    ("phase_damp", r"$\lambda$ · phase damping per gate", 0.10, 5e-4, "Gate errors"),
    ("kick", r"$\kappa$ · random-axis kick (rad)", 0.50, 0.005, "Gate errors"),
    ("over_rot", r"$\varepsilon$ · over-rotation (fraction of each pulse)", 0.20, 0.005, "Coherent errors"),
    ("zz_phase", r"$\theta_{ZZ}$ · ZZ phase after two-qubit gates (rad)", 0.50, 0.005, "Coherent errors"),
    ("z_phase", r"$\varphi$ · Z phase after every gate (rad)", 0.30, 0.005, "Coherent errors"),
    ("field_sigma", r"$\sigma$ · quasi-static random field (rad per idle layer)", 0.50, 0.005, "Idle qubits"),
    ("idle_dephase", r"$\lambda_{\rm idle}$ · Markovian dephasing per idle layer", 0.10, 5e-4, "Idle qubits"),
    ("idle_damp", r"$\gamma_{\rm idle}$ · amplitude damping per idle layer", 0.10, 5e-4, "Idle qubits"),
    ("ro01", r"$\epsilon_{01}$ · readout $P(1\,|\,0)$", 0.20, 0.001, "Readout"),
    ("ro10", r"$\epsilon_{10}$ · readout $P(0\,|\,1)$", 0.20, 0.001, "Readout"),
    ("xtalk_z", r"$\delta_{\rm x}$ · spectator Z phase on idle neighbors of a driven qubit (rad)", 0.30, 0.005,
     "Crosstalk (spectators)"),
    ("xtalk_zz", r"$\zeta$ · ZZ phase between a driven qubit and each idle neighbor (rad)", 0.30, 0.005,
     "Crosstalk (spectators)"),
]
PROBABILITY_FIELDS = {"p1", "p2", "amp_damp", "phase_damp", "idle_dephase", "idle_damp", "ro01", "ro10"}
SIM_OPTIONS = list(BACKENDS) + list(device.FAKE_DEVICES)


def sim_label(k: str) -> str:
    if k in BACKENDS:
        return BACKENDS[k]
    return "Fake device: " + device.device_label(k)


def sidebar(run_label: str | None = "Run simulation", run_key: str = "run_requested"):
    """Render the sidebar. Returns nothing; a Run button sets session_state[run_key] = True."""
    s = st.session_state
    sidebar_nav()
    with st.sidebar:
        if run_label:
            def req():
                s[run_key] = True
            st.button(run_label, type="primary", icon=":material/play_arrow:", key="sb_run_" + run_key,
                      on_click=req, width="stretch")
            st.caption("Heavy work starts only when you press Run. Settings apply immediately and mark old results "
                       "as out of date.")
        st.markdown("**Simulator**")
        if s.get("w_backend") not in SIM_OPTIONS:
            s.w_backend = s.backend if s.backend in SIM_OPTIONS else "numpy"
        st.selectbox("Backend", SIM_OPTIONS, format_func=sim_label, key="w_backend", on_change=S.cb_backend,
                     persist_state=P,
                     help="NumPy and Aer run the noise injector's model and agree to machine precision. A fake "
                          "device compiles for a real IBM processor (its coupling map and native gates) and uses "
                          "its calibration snapshot: gate errors, T1/T2 relaxation in every idle window, and "
                          "per-qubit readout errors.")
        st.markdown("<span style='font-size:0.85rem'>Shot budget per method</span>", unsafe_allow_html=True)
        a, b = st.columns([3, 2], vertical_alignment="bottom")
        a.select_slider("Shot budget per method", S.SHOT_STEPS, key="w_shots_sl", on_change=S.cb_shots, args=("sl",),
                        format_func=lambda v: f"{v:,}", label_visibility="collapsed", persist_state=P)
        b.number_input("shots", min_value=1, step=1, key="w_shots_num", on_change=S.cb_shots, args=("num",),
                       label_visibility="collapsed", persist_state=P,
                       help="The total number of shots every method may spend on one estimate — its target, "
                            "training and calibration circuits together — so all methods are compared at the same "
                            "cost. Type any number; the slider jumps in powers of two.")
        c, d = st.columns(2)
        c.number_input("Time per shot (µs)", min_value=0.0, step=10.0, format="%.1f", key="w_tshot",
                       on_change=S.cb_tshot, persist_state=P,
                       help="Repetition period incl. reset; converts shots to QPU time (illustrative).")
        d.number_input("Time limit (s)", min_value=0.0, step=60.0, format="%.0f", key="w_tlimit",
                       on_change=S.cb_tlimit, persist_state=P,
                       help="A run stops cleanly after this many seconds; 0 = no limit. There is no limit on "
                            "qubits or depth.")
        S.wdefault("w_zip_images", True)
        st.toggle("PNG/SVG images in ZIP downloads", key="w_zip_images", persist_state=P,
                  help="On: every figure is also rendered as PNG and SVG (needs Kaleido and a Chrome/Chromium; a "
                       "separate, time-limited process — tables and data never wait for it). Off: tables, data, "
                       "state and each figure's Plotly JSON only; instant, no browser needed.")
        st.divider()
        if device.is_device(s.backend):
            device_panel()
        else:
            noise_injector()


def noise_injector():
    s = st.session_state
    st.markdown("**Noise injector**")
    st.selectbox("Preset", list(PRESETS) + ["Custom"], key="w_nz_preset", on_change=S.cb_noise_preset,
                 persist_state=P, help="Illustrative values, not a specific device. Every value below can be typed "
                                       "exactly in the box next to its slider.")
    groups = {}
    for f, lab, hi, step, grp in NOISE_UI:
        groups.setdefault(grp, []).append((f, lab, hi, step))
    for grp, items in groups.items():
        with st.expander(grp, expanded=grp == "Gate errors", key="sb_exp_" + grp):
            for f, lab, hi, step in items:
                num_slider(lab, "w_nz_" + f, 0.0, hi, step, float(getattr(s.noise, f)),
                           on_change=S.cb_noise_field, args=(f, ), fmt="%.5f" if step < 1e-3 else "%.4f",
                           hard_max=1.0 if f in PROBABILITY_FIELDS else None)
    r1, r2 = s.noise.gate_infidelity(1), s.noise.gate_infidelity(2)
    st.caption("Datasheet view: average gate infidelity r = %.2e (1q), %.2e (2q)." % (r1, r2))


def device_panel():
    s = st.session_state
    st.markdown("**Device model**")
    st.caption(device.device_label(s.backend))
    st.caption("The fake backend's calibration replaces the noise injector: every native gate has its own error "
               "and duration, idle windows relax with each qubit's T1 and T2, and each qubit has its own readout "
               "errors. Noise channels you place in the circuit still apply. PEC and TREX use the device's "
               "median rates as their model.")
    info = s.get("device_info")
    if info:
        st.caption("Physical qubits in use: %s" % ", ".join(map(str, info.get("phys", []))))
