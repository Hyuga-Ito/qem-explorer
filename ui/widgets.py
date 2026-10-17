"""Reusable Streamlit pieces: slider + typed number, the "Why this happens?" expander, figures with export,
schema-driven method parameters, the recipe builder, and guarded execution with readable errors.

Every value widget here has a fixed key, no default argument, and ``persist_state="session"``; its first value
comes from ``state.wdefault`` and model updates go through ``on_change`` callbacks (see ``ui/state.py``)."""
from __future__ import annotations

import logging
import math
import re
import traceback

import pandas as pd
import streamlit as st
try:
    from streamlit.runtime.scriptrunner_utils.exceptions import ScriptControlException
except Exception:                                             # pragma: no cover - older Streamlit
    class ScriptControlException(Exception):
        pass

from qemx import bundle as _bundle  # noqa: F401  (sets BROWSER_PATH for Kaleido when a Chromium is available)
from qemx.explain import WHY
from qemx.methods import DD, READOUT, REGISTRY, TWIRL, Recipe
from qemx.noise import NoiseSpec
from qemx.resources import RunTimeout, explain
from qemx.viz import theme as T

from . import state as S

LOG = logging.getLogger("qem_explorer")
P = "session"                    # persist_state for every widget


# ------------------------------------------------------------------ typed numbers next to sliders
def num_slider(label: str, key: str, lo: float, hi: float, step: float, value: float, on_change, args=(),
               fmt: str = "%.4f", help: str | None = None, hard_max: float | None = None, container=None):
    """A slider for quick exploration and a number field for exact values, kept in sync.

    The number field accepts any value in [lo, hard_max] (hard_max=None: no upper bound); the slider shows it
    clamped to its range. `on_change(*args, src)` receives src = "sl" or "num"."""
    c = container or st
    S.wdefault(key + "_num", float(value))
    S.wdefault(key + "_sl", float(min(max(value, lo), hi)))
    if st.session_state[key + "_sl"] > hi or st.session_state[key + "_sl"] < lo:
        st.session_state[key + "_sl"] = float(min(max(st.session_state[key + "_sl"], lo), hi))
    c.markdown("<div style='font-size:0.875rem;margin-bottom:-0.6rem'>%s</div>" % label if "$" not in label
               else label, unsafe_allow_html="$" not in label, help=help)
    a, b = c.columns([3, 2], vertical_alignment="center")
    a.slider(label, float(lo), float(hi), step=float(step), key=key + "_sl", format=fmt, on_change=on_change,
             args=tuple(args) + ("sl",), label_visibility="collapsed", persist_state=P)
    b.number_input(label, min_value=float(lo), max_value=None if hard_max is None else float(hard_max),
                   step=float(step), format=fmt, key=key + "_num", on_change=on_change, args=tuple(args) + ("num",),
                   label_visibility="collapsed", persist_state=P)


# ------------------------------------------------------------------ text from users or exceptions
_MD_SPECIAL = "\\`*_{}[]<>()#+-.!|~$:/"


def md_escape(text) -> str:
    """Escape Markdown/HTML-significant characters so that text from a file, an exception or a user is shown
    literally (no links, no formatting, no LaTeX)."""
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in str(text)).replace("\n", " ")


def plain(text, limit: int = 80) -> str:
    """For places that render Markdown but not backslash escapes (badges): drop the significant characters."""
    return re.sub(r"[\\`*_{}\[\]<>#|~$!:]", "", str(text))[:limit]


def count_str(k) -> str:
    """Counts that fit a narrow metric card: 987, 12.3k, 4.56M."""
    k = int(k)
    if abs(k) < 1000:
        return str(k)
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(k) >= div:
            return "%.3g%s" % (k / div, suf)
    return str(k)


def plural(k: int, word: str) -> str:
    return "%d %s%s" % (k, word, "" if k == 1 else "s")


# ------------------------------------------------------------------ explanations and figures
def why(key: str, extra: str = "", container=None):
    with (container or st).expander("Why this happens?", icon=":material/lightbulb:"):
        st.markdown(WHY[key] + ("\n\n" + extra if extra else ""))


def show_fig(fig, key: str, data: pd.DataFrame | None = None, container=None):
    """A Plotly figure that keeps zoom/rotation across reruns, with a per-figure export menu."""
    c = container or st
    try:
        fig.update_layout(uirevision=key)
    except Exception:
        pass
    c.plotly_chart(fig, width="stretch", config=T.PLOT_CONFIG, key="fig_" + key)
    with c.popover("Export", icon=":material/download:"):
        st.download_button("Interactive HTML", lambda: fig.to_html(include_plotlyjs="cdn"), file_name=key + ".html",
                           mime="text/html", key="dl_html_" + key)
        if data is not None:
            st.download_button("Data (CSV)", data.to_csv(index=False), file_name=key + ".csv", mime="text/csv",
                               key="dl_csv_" + key)

        def png():
            return fig.to_image(format="png", width=int(fig.layout.width or 1400), height=int(fig.layout.height or 600),
                                scale=3)

        def svg():
            return fig.to_image(format="svg", width=int(fig.layout.width or 1400), height=int(fig.layout.height or 600))
        st.download_button("PNG (3× resolution)", png, file_name=key + ".png", mime="image/png", key="dl_png_" + key,
                           help="Rendered with Kaleido; needs Chrome or Chromium on the server.")
        st.download_button("SVG (vector)", svg, file_name=key + ".svg", mime="image/svg+xml", key="dl_svg_" + key)


# ------------------------------------------------------------------ guarded execution
def guarded(what: str, fn, *args, n: int | None = None, **kw):
    """Run fn under the session's time limit; show a readable error instead of a crash. Returns (ok, value)."""
    from qemx.resources import deadline
    try:
        with deadline(st.session_state.get("time_limit", 0)):
            return True, fn(*args, **kw)
    except (ScriptControlException, KeyboardInterrupt, SystemExit, GeneratorExit):
        raise                                                 # Streamlit's own control flow (stop / rerun)
    except BaseException as e:                                # noqa: BLE001 - every failure becomes a message
        title, msg = explain(e, what, n)
        LOG.warning("%s failed: %s", what, "".join(traceback.format_exception_only(type(e), e)).strip())
        st.error("**%s.** %s" % (title, msg), icon=":material/error:" if not isinstance(e, RunTimeout)
                 else ":material/timer_off:")
        with st.expander("Technical details"):
            st.code("%s: %s" % (type(e).__name__, str(e)[:2000]), language="text")
        return False, None


def memory_notice(est, what: str):
    """A warning (never a block) when a run looks heavy for this machine."""
    if est.level == "risky":
        st.warning("**Heavy run.** %s %s. It may run out of memory; the app will catch that and tell you, but the "
                   "machine may become slow first." % (what, est.text()), icon=":material/memory:")
    elif est.level == "heavy":
        st.info("%s %s." % (what, est.text()), icon=":material/memory:")


# ------------------------------------------------------------------ method parameters
def _pkey(prefix: str, method_key: str, p) -> str:
    return "w_%s_%s_%s" % (prefix, method_key, p.name)


def _initial(p, noise: NoiseSpec, n: int, symmetry=None):
    """The first value of a parameter's widget."""
    if p.kind == "float" and p.default is None:          # PEC's assumed rates default to the injector's values
        return float(noise.p1 if p.name == "p1" else noise.p2)
    if p.kind == "floats":
        return ", ".join("%g" % v for v in p.default)
    if p.kind == "ints":
        dflt = list(symmetry[0]) if (symmetry and p.name == "qubits") else list(p.default)
        return [q for q in dflt if q < n]
    if p.kind == "text":
        return str(symmetry[2]) if (symmetry and len(symmetry) > 2 and p.name == "more") else str(p.default)
    if p.kind == "int":
        return int(p.default)
    if p.kind == "float":
        return float(p.default)
    if p.kind == "bool":
        return bool(p.default)
    return p.default


def widget_bounds(p) -> tuple:
    """(min_value, max_value) for a parameter's number widget: the schema's lo/hi where Streamlit can show them
    (integers within ±(2^53 − 1), finite floats), else None."""
    def ok(v):
        if v is None:
            return None
        if p.kind == "int":
            return int(v) if abs(v) <= S.JS_SAFE_INT else None
        return float(v) if math.isfinite(v) else None
    return ok(p.lo), ok(p.hi)


def parse_floats(txt) -> list:
    """Finite numbers separated by commas or semicolons (ValueError otherwise)."""
    vals = []
    for x in str(txt).replace(";", ",").split(","):
        x = x.strip()
        if not x:
            continue
        if not NUMBER.fullmatch(x):
            raise ValueError("%r is not a number" % x[:20])
        v = float(x)
        if not math.isfinite(v):
            raise ValueError("%r is not finite" % x[:20])
        vals.append(v)
    return vals


NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
INTEGER = re.compile(r"[+-]?\d+", re.ASCII)


def parse_ints(txt) -> list:
    """Whole numbers (ASCII digits) separated by commas or semicolons (ValueError otherwise)."""
    out = []
    for x in str(txt).replace(";", ",").split(","):
        x = x.strip()
        if not x:
            continue
        if not INTEGER.fullmatch(x) or len(x) > 12:
            raise ValueError("%r is not a whole number" % x[:20])
        out.append(int(x))
    return out


def _value(p, raw, n: int):
    """A widget value turned into the parameter value (ValueError for an unreadable list)."""
    if p.kind == "choice":
        return raw if raw in p.options else p.default
    if p.kind == "int":
        return int(raw)
    if p.kind == "float":
        return float(raw)
    if p.kind == "floats":
        vals = parse_floats(raw)
        if not vals:
            raise ValueError("enter at least one number")
        return vals
    if p.kind == "ints":
        return [q for q in (raw or []) if isinstance(q, int) and 0 <= q < n]
    if p.kind == "bool":
        return bool(raw)
    return str(raw)


def read_params(method_key: str, prefix: str, noise: NoiseSpec, n: int, symmetry=None) -> tuple:
    """(parameters, errors) from the widget values in session state, without rendering — for callbacks, which
    must not use values captured in an earlier run. Hidden parameters (``when``) take their defaults."""
    s = st.session_state
    out, errs = {}, []
    for p in REGISTRY[method_key].params:
        if not S.param_visible(p, out):
            out[p.name] = p.default
            continue
        raw = s.get(_pkey(prefix, method_key, p), _initial(p, noise, n, symmetry))
        try:
            out[p.name] = _value(p, raw, n)
        except (TypeError, ValueError) as e:
            errs.append("%s: %s" % (p.label, e))
            out[p.name] = list(p.default) if p.kind == "floats" else p.default
    return out, errs


def param_inputs(method_key: str, prefix: str, noise: NoiseSpec, n: int, symmetry=None) -> dict:
    """Controls for a method's parameters, generated from its schema (typed numbers everywhere). A parameter whose
    ``when`` conditions do not hold for the values chosen above it is not shown and keeps its default."""
    m = REGISTRY[method_key]
    out = {}
    for p in m.params:
        if not S.param_visible(p, out):
            out[p.name] = p.default
            continue
        k = _pkey(prefix, method_key, p)
        S.wdefault(k, _initial(p, noise, n, symmetry))
        if p.kind == "choice":
            if st.session_state[k] not in p.options:
                st.session_state[k] = p.default
            out[p.name] = st.selectbox(p.label, p.options, key=k, help=p.help or None, persist_state=P)
        elif p.kind in ("int", "float"):
            lo, hi = widget_bounds(p)
            cast = int if p.kind == "int" else float
            try:
                v = cast(st.session_state[k])
            except (TypeError, ValueError):
                v = cast(_initial(p, noise, n, symmetry))
            v = min(max(v, lo if lo is not None else v), hi if hi is not None else v)
            st.session_state[k] = v                     # a stored value outside the bounds would be rejected
            if p.kind == "int":
                out[p.name] = int(st.number_input(p.label, min_value=lo, max_value=hi, step=int(p.step or 1), key=k,
                                                  help=p.help or None, persist_state=P))
            else:
                out[p.name] = float(st.number_input(p.label, min_value=lo, max_value=hi, step=float(p.step or 1e-3),
                                                    format="%.6f", key=k, help=p.help or None, persist_state=P))
        elif p.kind == "floats":
            txt = st.text_input(p.label, key=k, help=p.help or None, persist_state=P, max_chars=400)
            try:
                out[p.name] = _value(p, txt, n)
            except ValueError:
                st.error("Enter finite numbers separated by commas.")
                out[p.name] = list(p.default)
        elif p.kind == "ints":
            st.session_state[k] = [q for q in st.session_state[k] if isinstance(q, int) and q < n]
            out[p.name] = st.multiselect(p.label, list(range(n)), key=k, persist_state=P)
        elif p.kind == "bool":
            out[p.name] = st.toggle(p.label, key=k, persist_state=P)
        elif p.kind == "text":
            out[p.name] = st.text_input(p.label, key=k, help=p.help or None, persist_state=P, max_chars=200)
    return out


def custom_options(custom_recipes) -> dict:
    """{"custom:<Recipe.key()>": Recipe} for the custom contenders: option values that identify a recipe exactly
    (two contenders may share a title, e.g. PEC with different assumed rates)."""
    out = {}
    for d in custom_recipes or []:
        try:
            r = Recipe.from_json(d)
        except Exception:                                   # noqa: BLE001 - a malformed entry is not offered
            continue
        out.setdefault("custom:" + r.key(), r)
    return out


def custom_labels(custom: dict) -> dict:
    """Display labels for custom_options(): the title, "(#k)" for a repeated title."""
    from qemx.export import recipe_labels
    return {k: "Custom: " + lab for k, lab in zip(custom, recipe_labels(list(custom.values())))}


def recipe_builder(noise: NoiseSpec, n: int, symmetry=None):
    """Build any recipe: transforms + estimator + readout mitigation."""
    s = st.session_state
    S.wdefault("w_rb_est", "zne")
    est = st.selectbox("Estimator", list(REGISTRY), format_func=lambda k: REGISTRY[k].label, key="w_rb_est",
                       persist_state=P)
    st.latex(REGISTRY[est].tex)
    param_inputs(est, "rb", noise, n, symmetry)
    c1, c2, c3 = st.columns(3)
    for k, d in (("w_rb_tw", "none"), ("w_rb_dd", "none"), ("w_rb_ro", "none")):
        S.wdefault(k, d)
    c1.selectbox("Twirling", list(TWIRL), format_func=TWIRL.get, key="w_rb_tw", persist_state=P)
    c2.selectbox("Dynamical decoupling", list(DD), format_func=DD.get, key="w_rb_dd", persist_state=P)
    c3.selectbox("Readout mitigation", list(READOUT), format_func=READOUT.get, key="w_rb_ro", persist_state=P)

    def add():
        """Callback: reads the widgets NOW (values typed just before the click are included)."""
        est_ = s.get("w_rb_est", "zne")
        if est_ not in REGISTRY:
            s.rb_msg = "Choose an estimator."
            return
        params, errs = read_params(est_, "rb", s.noise, int(s.n), s.get("symmetry"))
        if errs:
            s.rb_msg = "Not added: " + "; ".join(errs)
            return
        try:
            S.validate_params(est_, params)
            r = Recipe.make(est_, s.get("w_rb_tw", "none"), s.get("w_rb_dd", "none"), s.get("w_rb_ro", "none"),
                            **params)
        except (ValueError, TypeError) as e:
            s.rb_msg = "Not added: %s" % str(e)[:200]
            return
        known = {x.key() for x in S.recipes()}
        if r.key() in known:
            s.rb_msg = "%s is already a contender." % r.title()
        else:
            saved = [Recipe.from_json(d).key() for d in s.custom_recipes]
            if r.key() not in saved:                     # into the saved library …
                s.custom_recipes = s.custom_recipes + [r.to_json()]
            s.custom_active = S.custom_active_keys() + [r.key()]     # … and active
            s.rb_msg = ""
    st.button("Add this contender", icon=":material/add:", key="rb_add", on_click=add)
    if s.get("rb_msg"):
        st.warning(md_escape(s.rb_msg))



# ------------------------------------------------------------------ responsive layout
RESPONSIVE_CSS = """<style>
/* Two-column page layouts (keyed containers named rsp_* or pg_build) wrap into one column when the right-hand
   column would be narrower than 300 px, e.g. a 900 px window with the sidebar open. */
.st-key-pg_build > div[data-testid="stHorizontalBlock"],
.st-key-pg_build > div > div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 1rem; }
.st-key-pg_build > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
.st-key-pg_build > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
  min-width: min(300px, 100%); }
</style>"""


def responsive_css():
    """Let the Playground's Build columns stack on narrow windows (CSS only, no layout logic in Python)."""
    st.html(RESPONSIVE_CSS)
