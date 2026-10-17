"""Build direction_A_design.pdf from the Markdown source (pandoc + LuaLaTeX, Times with Unicode fallbacks).
Pipe-table column widths are set proportional to their content (pandoc reads them from the separator dashes)."""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "direction_A_design.md")
OUT = os.path.splitext(SRC)[0] + ".pdf"


SYM = {"λ": r"\lambda", "χ": r"\chi", "ε": r"\varepsilon", "γ": r"\gamma", "β": r"\beta", "κ": r"\kappa",
       "σ": r"\sigma", "θ": r"\theta", "φ": r"\phi", "η": r"\eta", "α": r"\alpha", "μ": r"\mu", "π": r"\pi",
       "ρ": r"\rho", "Σ": r"\Sigma", "Γ": r"\Gamma", "⟨": r"\langle ", "⟩": r"\rangle ", "⊗": r"\otimes ",
       "√": r"\surd ", "≤": r"\le ", "≥": r"\ge ", "≈": r"\approx ", "∝": r"\propto ", "→": r"\to ",
       "∞": r"\infty ", "≠": r"\ne ", "∈": r"\in ", "⌊": r"\lfloor ", "⌋": r"\rfloor ", "⌈": r"\lceil ",
       "⌉": r"\rceil ", "±": r"\pm ", "∂": r"\partial ", "ψ": r"\psi", "Ô": r"\hat O", "ψ": r"\psi"}
SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ᵏ", "0123456789-+k"))
SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"))


GREEK = "λχεγβκσθφηαμπρψΣΓ"
SUBRE = re.compile(r"(?<![\w\\])([A-Za-z" + GREEK + r"])_(\{[^}\n]{1,20}\}|[A-Za-z0-9" + GREEK + r"]{1,6})")


def _sym(text: str) -> str:
    """Unicode symbols -> LaTeX math source (no delimiters)."""
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in SUP or ch in SUB:
            table = SUP if ch in SUP else SUB
            j = i
            while j < len(text) and text[j] in table:
                j += 1
            out.append("%s{%s}" % ("^" if table is SUP else "_", "".join(table[c] for c in text[i:j])))
            i = j
            continue
        out.append(SYM[ch] + " " if ch in SYM else ch)
        i += 1
    return "".join(out)


def mathify(md: str) -> str:
    """Unicode symbols and x_sub patterns outside code and math -> raw-TeX \ensuremath{…}, so the text font (Times)
    never lacks a glyph and subscripts render as subscripts."""
    out = []
    parts = re.split(r"(```.*?```|`[^`]*`|\$\$.*?\$\$|\$[^$\n]+\$)", md, flags=re.S)
    for k, part in enumerate(parts):
        if k % 2 == 1:
            out.append(part)
            continue
        if part.startswith("---\n") and k == 0:              # YAML metadata block: leave untouched
            end = part.find("\n---", 4)
            if end > 0:
                out.append(part[:end + 4])
                part = part[end + 4:]

        def sub_repl(m):
            base, sb = m.group(1), m.group(2)
            sb = sb[1:-1] if sb.startswith("{") else sb
            sbt = _sym(sb)
            if len(sb) > 1 and re.fullmatch(r"[A-Za-z,]+", sb):
                sbt = r"\mathrm{%s}" % sb
            return r"\ensuremath{%s_{%s}}" % (_sym(base).strip(), sbt)
        part = SUBRE.sub(sub_repl, part)
        buf, i = [], 0
        while i < len(part):
            ch = part[i]
            if part.startswith(r"\ensuremath{", i):           # already converted: copy to the matching brace
                depth, j = 0, i + len(r"\ensuremath")
                while j < len(part):
                    if part[j] == "{":
                        depth += 1
                    elif part[j] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                buf.append(part[i:j + 1])
                i = j + 1
                continue
            if ch in SUP or ch in SUB:
                table = SUP if ch in SUP else SUB
                j = i
                while j < len(part) and part[j] in table:
                    j += 1
                buf.append(r"\ensuremath{%s{%s}}" % ("^" if table is SUP else "_", "".join(table[c] for c in part[i:j])))
                i = j
                continue
            buf.append(r"\ensuremath{%s}" % SYM[ch].strip() if ch in SYM else ch)
            i += 1
        out.append("".join(buf))
    return "".join(out)


def widen(md: str) -> str:
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|(\s*:?-+:?\s*\|)+\s*$", lines[i + 1]):
            j = i
            block = []
            while j < len(lines) and lines[j].startswith("|"):
                block.append(lines[j])
                j += 1
            rows = [re.split(r"(?<!\\)\|", r.strip())[1:-1] for r in block]
            ncol = len(rows[0])
            L = [max(len(r[c].strip()) if c < len(r) else 0 for k, r in enumerate(rows) if k != 1) for c in range(ncol)]
            L = [min(max(x, 12), 90) for x in L]
            tot = sum(L)
            dashes = [max(3, int(round(120 * x / tot))) for x in L]
            block[1] = "|" + "|".join("-" * d for d in dashes) + "|"
            out += block
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


hdr = r"""
\setmonofont{DejaVu Sans Mono}[Scale=0.82]
\usepackage{xcolor}\definecolor{qblue}{HTML}{156082}\definecolor{qred}{HTML}{B91C1C}
\usepackage{titlesec}\titleformat{\section}{\Large\bfseries\color{qblue}}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries}{}{0pt}{}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\thepage}\fancyhead[R]{\small QEM Explorer v1.3 · Direction A}\renewcommand{\headrulewidth}{0pt}
\usepackage{etoolbox}\AtBeginEnvironment{longtable}{\footnotesize}\AtBeginEnvironment{Shaded}{\footnotesize}
\usepackage{caption}\captionsetup{font=small}
\renewcommand{\arraystretch}{1.15}
"""
with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, "doc.md")
    open(src, "w").write(widen(mathify(open(SRC).read())))
    h = os.path.join(tmp, "h.tex")
    open(h, "w").write(hdr)
    subprocess.run(["pandoc", src, "-o", OUT, "--pdf-engine=xelatex", "-V", "papersize=a4", "-V",
                    "geometry:margin=1.9cm", "-V", "fontsize=10pt", "-V", "colorlinks=true", "-H", h,
                    "--resource-path", HERE, "-V", "fontfamily=fontspec", "-V", "mainfont=TeX Gyre Termes", "-V", "monofont=DejaVu Sans Mono",
                    "-V", "mathfont=TeX Gyre Termes Math", "--toc", "--toc-depth=2", "-f", "markdown+pipe_tables"], check=True)
print("wrote", OUT)
