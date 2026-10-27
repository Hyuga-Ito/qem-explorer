r"""Math in tooltips. Streamlit renders a widget's ``help`` text as Markdown in a small popup where KaTeX math is
unreliable (raw ``$\sqrt{b^2+v/N}$`` or broken layout). Every ``help`` string therefore goes through ``plain()``,
which rewrites the TeX between dollar signs as Unicode math (√(b² + v/N)); ``install()`` applies it to every widget,
metric and column-config tooltip in the app, once, at start-up.
"""
from __future__ import annotations

import functools
import re

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "Gamma": "Γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε",
         "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "Theta": "Θ", "vartheta": "ϑ", "kappa": "κ",
         "lambda": "λ", "Lambda": "Λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "Pi": "Π", "rho": "ρ",
         "sigma": "σ", "Sigma": "Σ", "tau": "τ", "phi": "φ", "varphi": "φ", "Phi": "Φ", "chi": "χ", "psi": "ψ",
         "Psi": "Ψ", "omega": "ω", "Omega": "Ω", "ell": "ℓ", "hbar": "ℏ"}
SYMBOLS = {"le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈", "sim": "∼",
           "propto": "∝", "infty": "∞", "sum": "∑", "prod": "∏", "cdot": "·", "times": "×", "pm": "±", "mp": "∓",
           "mid": "|", "lvert": "|", "rvert": "|", "vert": "|", "lVert": "‖", "rVert": "‖", "to": "→",
           "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "mapsto": "↦", "in": "∈", "notin": "∉",
           "subset": "⊂", "otimes": "⊗", "oplus": "⊕", "dagger": "†", "ldots": "…", "cdots": "⋯", "dots": "…",
           "langle": "⟨", "rangle": "⟩", "partial": "∂", "nabla": "∇", "forall": "∀", "exists": "∃",
           "equiv": "≡", "perp": "⊥", "circ": "∘", "star": "⋆", "ast": "∗", "lfloor": "⌊", "rfloor": "⌋",
           "lceil": "⌈", "rceil": "⌉", "sqrt": "√", "quad": " ", "qquad": "  ", "lbrace": "{", "rbrace": "}",
           "{": "{", "}": "}", "%": "%", "#": "#", "&": "&", "_": "_", "|": "‖"}
SUP = dict(zip("0123456789+-=()nijkmrstuvwxyzabcdefghlop*", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱʲᵏᵐʳˢᵗᵘᵛʷˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰˡᵒᵖ*"))
SUP.update({"T": "ᵀ", "†": "†", "∗": "*", "−": "⁻", "∞": "^∞"})
SUB = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
SUB.update({"−": "₋"})
FONTS = {"mathbb": {"E": "𝔼", "R": "ℝ", "Z": "ℤ", "C": "ℂ", "N": "ℕ", "P": "ℙ", "Q": "ℚ"},
         "mathcal": {"E": "ℰ", "N": "𝒩", "O": "𝒪", "U": "𝒰", "D": "𝒟", "C": "𝒞", "B": "ℬ", "T": "𝒯", "L": "ℒ",
                     "S": "𝒮", "H": "ℋ", "P": "𝒫"}}
VULGAR = {("1", "2"): "½", ("1", "3"): "⅓", ("2", "3"): "⅔", ("1", "4"): "¼", ("3", "4"): "¾", ("1", "8"): "⅛"}
ACCENTS = {"hat": "̂", "bar": "̄", "tilde": "̃", "vec": "⃗", "dot": "̇"}


def _group(s: str, i: int) -> tuple:
    """(content, next index) of a {…} group or a single token starting at i."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], len(s)
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _script(text: str, table: dict, mark: str) -> str:
    t = text.replace(" ", "")
    if t and all(c in table for c in t):
        return "".join(table[c] for c in t)
    return mark + (t if len(t) == 1 else "(%s)" % t)


def _wrap(x: str) -> str:
    return x if re.fullmatch(r"[\w.ⁱ⁰-⁹₀-₉²³¹′]+|[^\s+\-−·/]+", x) else "(%s)" % x


def tex(s: str) -> str:
    """TeX math (without the dollar signs) → Unicode text."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            name, i = m.group(1), i + len(m.group(0))
            if name.isalpha() and name not in ("frac", "dfrac", "tfrac", "sqrt") and name not in FONTS \
                    and name not in ACCENTS and name not in ("mathrm", "text", "textrm", "mathit", "mathbf",
                                                           "boldsymbol", "operatorname", "mbox", "textbf", "textit"):
                while i < len(s) and s[i] == " ":           # TeX ignores spaces after a control word
                    i += 1
            if name in ("frac", "dfrac", "tfrac"):
                a, i = _group(s, i)
                b, i = _group(s, i)
                ta, tb = tex(a), tex(b)
                out.append(VULGAR.get((ta, tb)) or "%s/%s" % (_wrap(ta), _wrap(tb)))
            elif name == "sqrt":
                a, i = _group(s, i)
                ta = tex(a)
                out.append("√" + (ta if len(ta) == 1 else "(%s)" % ta))
            elif name in ("mathrm", "text", "textrm", "mathit", "mathbf", "boldsymbol", "operatorname", "mbox",
                          "textbf", "textit", "rm", "it", "bf"):
                if name in ("rm", "it", "bf"):
                    continue
                a, i = _group(s, i)
                out.append(tex(a) if name not in ("text", "textrm", "mbox") else a)
            elif name in FONTS:
                a, i = _group(s, i)
                out.append("".join(FONTS[name].get(ch, ch) for ch in a))
            elif name in ACCENTS:
                a, i = _group(s, i)
                ta = tex(a)
                out.append(ta + ACCENTS[name] if len(ta) == 1 else ta)
            elif name in ("left", "right", "big", "Big", "bigl", "bigr", "Bigl", "Bigr", "displaystyle", "limits",
                          "nolimits", "!", ",", ";", ":", " ", "textstyle"):
                if name in (",", ";", ":", " "):
                    out.append(" " if name != "," else " ")
            elif name in GREEK:
                out.append(GREEK[name])
            elif name in SYMBOLS:
                out.append(SYMBOLS[name])
            elif name in ("min", "max", "log", "ln", "exp", "sin", "cos", "tan", "det", "Tr", "arg", "gcd", "mod",
                          "bmod", "lim", "inf", "sup"):
                out.append(name if name != "bmod" else "mod")
            else:
                out.append(name)
        elif c in "^_":
            a, i = _group(s, i + 1)
            out.append(_script(tex(a), SUP if c == "^" else SUB, c))
        elif c in "{}":
            i += 1
        elif c == "~":
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    txt = "".join(out).replace("-", "−")
    txt = re.sub(r"\s*([=<>≤≥≈])\s*", r" \1 ", txt)
    return re.sub(r"  +", " ", txt).strip()


_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)


def plain(text):
    """A help text with every $…$ / $$…$$ math span rewritten as Unicode; other Markdown is kept."""
    if not isinstance(text, str) or "$" not in text:
        return text
    return _MATH.sub(lambda m: tex(m.group(1) or m.group(2)), text)


WIDGETS = ("button", "download_button", "checkbox", "toggle", "radio", "selectbox", "multiselect", "slider",
           "select_slider", "text_input", "number_input", "text_area", "date_input", "time_input", "file_uploader",
           "color_picker", "metric", "segmented_control", "pills", "popover", "link_button", "page_link",
           "form_submit_button", "data_editor", "chat_input", "camera_input", "audio_input", "markdown", "caption",
           "title", "header", "subheader", "latex", "badge", "expander")
COLUMNS = ("Column", "TextColumn", "NumberColumn", "CheckboxColumn", "SelectboxColumn", "ProgressColumn",
           "LinkColumn", "ListColumn", "DatetimeColumn", "DateColumn", "TimeColumn", "LineChartColumn",
           "BarChartColumn", "AreaChartColumn", "ImageColumn", "JsonColumn", "MultiselectColumn")


def _patched(fn):
    if getattr(fn, "_qx_plain_help", False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if isinstance(kwargs.get("help"), str):
            kwargs["help"] = plain(kwargs["help"])
        return fn(*args, **kwargs)
    wrapper._qx_plain_help = True
    return wrapper


def install() -> None:
    """Patch every Streamlit element that takes ``help`` (idempotent)."""
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator
    for name in WIDGETS:
        orig = getattr(DeltaGenerator, name, None)
        if orig is not None:
            setattr(DeltaGenerator, name, _patched(orig))
    main = getattr(st, "_main", None)
    for name in WIDGETS:
        if main is not None and hasattr(main, name) and hasattr(st, name):
            setattr(st, name, getattr(main, name))
    cc = st.column_config
    for name in COLUMNS:
        f = getattr(cc, name, None)
        if f is not None:
            setattr(cc, name, _patched(f))
