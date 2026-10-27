"""Helpers shared by the analysis pages."""
from __future__ import annotations

import numpy as np
import streamlit as st

from qemx import device
from qemx.circuit import ParseError

from . import cache as K
from . import state as S


def dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def compiled_now():
    """(compiled circuit, error message) for the current state; never raises."""
    ss = st.session_state
    try:
        spec = S.spec()
        c = K.get_compiled(spec, ss.inputs, ss.basis, S.compile_options(), ss.relabel)
        return c, None
    except (ValueError, ParseError) as e:
        return None, str(e)
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)


def context_line(compiled):
    ss = st.session_state
    from .widgets import md_escape, plural
    parts = ["**%s**" % md_escape(ss.get("template_name", "circuit")), plural(int(ss.n), "qubit")]
    if compiled is not None:
        parts.append("%s after compilation" % plural(compiled.implemented.count("2q"), "two-qubit gate"))
    parts.append("observable %s" % md_escape(ss.observable.label))
    if device.is_device(ss.backend):
        parts.append("device model %s" % device.FAKE_DEVICES[ss.backend][0])
    else:
        parts.append("noise: %s" % ss.noise_preset)
    st.caption(" · ".join(parts) + ". Change them in the Playground or the sidebar; results here update when you "
                                   "press the page's Run button.")


def run_comparison(compiled):
    """The Playground comparison (shared: the same results appear on both pages)."""
    from . import widgets as W
    ss = st.session_state
    sig = S.signature()
    recipes = tuple(S.recipes())
    ok, cmp_ = W.guarded("The comparison", K.comparison, sig, compiled, K.noise_json(ss.noise), ss.observable, recipes,
                         ss.backend, int(ss.shots), ss.symmetry, n=compiled.implemented.n)
    if not ok:
        return None
    ideal, outs, runner = cmp_
    from qemx.recipes import table_rows
    ok, tr = W.guarded("The cost table", table_rows, outs, ideal, int(ss.shots), ss.t_shot_us,
                       n=compiled.implemented.n)
    if not ok:
        return None
    samples, rows = tr
    ss.results = dict(sig=sig, ideal=ideal, outs=outs, samples=samples, rows=rows, calls=runner.calls,
                      shots=int(ss.shots), backend=ss.backend, state=S.to_json())
    return ss.results


def results_current():
    ss = st.session_state
    return ss.get("results") is not None and ss.results["sig"] == S.signature()


def sim_backend_for_sweeps() -> str:
    """Noise sweeps vary the injector's model, so they run on NumPy or Aer (a device model is fixed)."""
    b = st.session_state.backend
    return b if not device.is_device(b) else "numpy"


def stale_note(button: str):
    """The shown results were computed for other inputs."""
    st.warning("These results were computed for different settings (circuit, noise, simulator, contenders or the "
               "controls above). Press **%s** to refresh them." % button, icon=":material/history:")
