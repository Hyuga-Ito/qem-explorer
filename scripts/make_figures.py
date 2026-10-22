r"""Build the algorithm diagrams (quantikz / TikZ → PDF → SVG) shown in the Learn and Theory pages.

Run ``python scripts/make_figures.py`` with a TeX installation (pdflatex, quantikz, pdftocairo). The SVGs in
docs/figures/ are committed, so the app itself never needs LaTeX.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")

PREAMBLE = r"""\documentclass[border=8pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{quantikz2,positioning,arrows.meta,calc,shapes.geometric,decorations.pathreplacing}
\usepackage{mathpazo}
\usepackage{amsmath,amssymb,braket}
\definecolor{qxblue}{HTML}{156082}
\definecolor{qxink}{HTML}{0B1A22}
\definecolor{qxtint}{HTML}{E4EEF3}
\definecolor{qxsand}{HTML}{F5ECD7}
\definecolor{qxrose}{HTML}{F6DEDE}
\definecolor{qxsage}{HTML}{E2EFE3}
\tikzset{g/.style={fill=qxtint,draw=qxink}, o/.style={fill=qxsand,draw=qxink}, e/.style={fill=qxrose,draw=qxink},
  q/.style={fill=qxsage,draw=qxink},
  grp/.style={dashed,rounded corners,draw=qxblue!80,inner xsep=4pt},
  box/.style={draw=qxink,rounded corners=3pt,fill=qxtint,align=center,inner sep=6pt,font=\small},
  arr/.style={-{Stealth[length=2.2mm]},thick,draw=qxink}}
\begin{document}
"""

FIGS = {}

FIGS["bell"] = r"""
\begin{quantikz}[row sep=0.35cm]
\lstick{$\ket{0}$} & \gate[style=g]{H} & \ctrl{1} & \meter{} \\
\lstick{$\ket{0}$} & & \targ{} & \meter{}
\end{quantikz}
"""

FIGS["ghz"] = r"""
\begin{quantikz}[row sep=0.3cm]
\lstick{$\ket{0}$} & \gate[style=g]{H} & \ctrl{1} & & & \meter{} \\
\lstick{$\ket{0}$} & & \targ{} & \ctrl{1} & & \meter{} \\
\lstick{$\ket{0}$} & & & \targ{} & \ctrl{1} & \meter{} \\
\lstick{$\ket{0}$} & & & & \targ{} & \meter{}
\end{quantikz}
"""

FIGS["qft"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.35cm, column sep=0.3cm]
\lstick{$q_2$} & \gate[style=g]{H} & \gate[style=g]{R_2} & \gate[style=g]{R_3} & & & & \swap{2} & \rstick{$y_0$} \\
\lstick{$q_1$} & & \ctrl{-1} & & \gate[style=g]{H} & \gate[style=g]{R_2} & & & \rstick{$y_1$} \\
\lstick{$q_0$} & & & \ctrl{-2} & & \ctrl{-1} & \gate[style=g]{H} & \targX{} & \rstick{$y_2$}
\end{quantikz}\\[4pt]
\small $R_k=\mathrm{diag}(1,e^{2\pi i/2^k})$. The approximate QFT of degree $m$ keeps only the $R_k$ with $k\le m+1$.
\end{tabular}
"""

FIGS["qpe"] = r"""
\begin{quantikz}[row sep=0.3cm, column sep=0.32cm]
\lstick{$\ket{0}$} & \gate[style=g]{H} & & & \ctrl{3} & \gate[3,style=g]{\mathrm{QFT}^{\dagger}} & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & & \ctrl{2} & & & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & \ctrl{1} & & & & \meter{} \\
\lstick{$\ket{u}$} & \qwbundle{} & \gate[style=o]{U} & \gate[style=o]{U^{2}} & \gate[style=o]{U^{4}} & & \rstick{$\ket{u}$}
\end{quantikz}
"""

FIGS["ising"] = r"""
\begin{quantikz}[row sep=0.3cm, column sep=0.3cm]
\lstick{$\ket{0}$} & \gate[2,style=g]{R_{ZZ}(-2J\Delta t)}\gategroup[3,steps=3,style=grp]{\small one Trotter step, repeated $r$ times} & & \gate[style=g]{R_X(-2h\Delta t)} & \meter{} \\
\lstick{$\ket{0}$} & & \gate[2,style=g]{R_{ZZ}(-2J\Delta t)} & \gate[style=g]{R_X(-2h\Delta t)} & \meter{} \\
\lstick{$\ket{0}$} & & & \gate[style=g]{R_X(-2h\Delta t)} & \meter{}
\end{quantikz}
"""

FIGS["bv"] = r"""
\begin{quantikz}[row sep=0.3cm, column sep=0.35cm]
\lstick{$\ket{0}$} & \gate[style=g]{H} & \gate[4,style=o]{U_f:\ \ket{x}\ket{b}\mapsto\ket{x}\ket{b\oplus s\cdot x}} & \gate[style=g]{H} & \meter{} \rstick{$s_0$} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & & \gate[style=g]{H} & \meter{} \rstick{$s_1$} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & & \gate[style=g]{H} & \meter{} \rstick{$s_2$} \\
\lstick{$\ket{-}$} & & & &
\end{quantikz}
"""

FIGS["hsp"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.45cm, column sep=0.4cm]
\lstick{$\ket{0}^{\otimes n}$} & \qwbundle{n} & \gate[style=g]{F_G} & \gate[2,style=o]{U_f} & \gate[style=g]{F_G^{\vphantom{\dagger}}} & \meter{} & \setwiretype{c} \rstick{$y\in H^{\perp}$} \\
\lstick{$\ket{0}^{\otimes m}$} & \qwbundle{m} & & & & \meter{} & \setwiretype{c} \rstick{coset label $f(x)$}
\end{quantikz}\\[6pt]
\small $G=\mathbb{Z}_2^n$: $F_G=H^{\otimes n}$, $f(x)=Ax$ with $\ker A=H$.\quad
$G=\mathbb{Z}_N$: $F_G=\mathrm{QFT}_N$, $f(x)=x \bmod d$, $y\in\{0,N/d,2N/d,\dots\}$.
\end{tabular}
"""

FIGS["grover"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.4cm, column sep=0.4cm]
\lstick{$\ket{0}^{\otimes n}$} & \qwbundle{n} & \gate[style=g]{H^{\otimes n}} & \gate[style=o]{O_m}\gategroup[1,steps=2,style=grp]{\small Grover iteration $G=D\,O_m$, $\ k\approx\frac{\pi}{4}\sqrt{N}$ times} & \gate[style=g]{D} & \meter{} & \setwiretype{c} \rstick{$m$}
\end{quantikz}\\[8pt]
\begin{tikzpicture}[baseline=(D.base)]\node (D) {$D\;=$};\end{tikzpicture}
\begin{quantikz}[row sep=0.25cm, column sep=0.28cm]
& \gate[style=g]{H} & \gate[style=g]{X} & \ctrl{2} & \gate[style=g]{X} & \gate[style=g]{H} & \\
& \gate[style=g]{H} & \gate[style=g]{X} & \control{} & \gate[style=g]{X} & \gate[style=g]{H} & \\
& \gate[style=g]{H} & \gate[style=g]{X} & \control{} & \gate[style=g]{X} & \gate[style=g]{H} &
\end{quantikz}
\hspace{1.2em}
\begin{tikzpicture}[baseline=(O.base)]\node (O) {$O_m\;=$};\end{tikzpicture}
\begin{quantikz}[row sep=0.25cm, column sep=0.28cm]
& \gate[style=g]{X^{\bar m_2}} & \ctrl{2} & \gate[style=g]{X^{\bar m_2}} & \\
& \gate[style=g]{X^{\bar m_1}} & \control{} & \gate[style=g]{X^{\bar m_1}} & \\
& \gate[style=g]{X^{\bar m_0}} & \control{} & \gate[style=g]{X^{\bar m_0}} &
\end{quantikz}\\[4pt]
\small $P(m)=\sin^2\!\big((2k+1)\theta\big)$, $\ \sin\theta=1/\sqrt{N}$; the controlled-$Z$ fires only on $\ket{1\cdots1}$.
\end{tabular}
"""

FIGS["vqe"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.3cm, column sep=0.3cm]
\lstick{$\ket{0}$} & \gate[style=g]{R_Y(\theta_1)}\gategroup[4,steps=3,style=grp]{\small layer, repeated $L$ times} & \ctrl{1} & & \gate[style=g]{R_Y} & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{R_Y(\theta_2)} & \targ{} & \ctrl{1} & \gate[style=g]{R_Y} & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{R_Y(\theta_3)} & \ctrl{1} & \targ{} & \gate[style=g]{R_Y} & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{R_Y(\theta_4)} & \targ{} & & \gate[style=g]{R_Y} & \meter{}
\end{quantikz}\\[6pt]
\begin{tikzpicture}
\node[box] (q) {quantum computer\\ prepare $U(\vec\theta)\ket{0}$, estimate\\ $E(\vec\theta)=\langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle$};
\node[box,fill=qxsand,right=1.8cm of q] (c) {classical optimizer\\ $\vec\theta\leftarrow\mathrm{update}(\vec\theta,E)$\\ until $E\to E_0$};
\draw[arr] (q.north east) to[bend left=18] node[above,font=\small]{$E(\vec\theta)$} (c.north west);
\draw[arr] (c.south west) to[bend left=18] node[below,font=\small]{$\vec\theta$} (q.south east);
\end{tikzpicture}
\end{tabular}
"""

FIGS["qaoa"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.3cm, column sep=0.3cm]
\lstick{$\ket{0}$} & \gate[style=g]{H} & \gate[4,style=o]{e^{-i\gamma_1 C}}\gategroup[4,steps=2,style=grp]{\small layer 1} & \gate[style=g]{R_X(2\beta_1)} & \ \cdots\ & \gate[4,style=o]{e^{-i\gamma_p C}}\gategroup[4,steps=2,style=grp]{\small layer $p$} & \gate[style=g]{R_X(2\beta_p)} & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & & \gate[style=g]{R_X(2\beta_1)} & \ \cdots\ & & \gate[style=g]{R_X(2\beta_p)} & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & & \gate[style=g]{R_X(2\beta_1)} & \ \cdots\ & & \gate[style=g]{R_X(2\beta_p)} & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & & \gate[style=g]{R_X(2\beta_1)} & \ \cdots\ & & \gate[style=g]{R_X(2\beta_p)} & \meter{}
\end{quantikz}\\[4pt]
\small $C=\sum_{(i,j)\in E}\tfrac12(1-Z_iZ_j)$: $e^{-i\gamma C}$ is one $R_{ZZ}(-\gamma)$ per edge (up to a global phase).
\end{tabular}
"""

FIGS["qml"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.3cm, column sep=0.5cm]
\lstick{$\ket{0}$} & \gate[3,style=o]{U_\Phi(\vec x)} & \gate[3,style=g]{W(\vec\theta)} & \meter{} \\
\lstick{$\ket{0}$} & & & \meter{} \\
\lstick{$\ket{0}$} & & & \meter{}
\end{quantikz}\\[4pt]
\small feature map $U_\Phi(\vec x)$ ($\times r$), trainable ansatz $W(\vec\theta)$; the model is $f(\vec x)=\langle Z\cdots Z\rangle$.
\end{tabular}
"""

FIGS["qec3"] = r"""
\begin{quantikz}[row sep=0.3cm, column sep=0.42cm]
\lstick{$\ket{\psi}$} & \ctrl{1}\gategroup[3,steps=2,style=grp]{\small encode} & \ctrl{2} & \gate[3,style=e]{E}\gategroup[3,steps=1,style=grp]{\small error} & \ctrl{3}\gategroup[5,steps=4,style=grp]{\small syndrome} & & & & \targ{}\gategroup[5,steps=3,style=grp]{\small correct} & & & \ctrl{1}\gategroup[3,steps=2,style=grp]{\small decode} & \ctrl{2} & \rstick{$\ket{\psi}$} \\
\lstick{$\ket{0}$} & \targ{} & & & & \ctrl{2} & \ctrl{3} & & & \targ{} & & \targ{} & & \\
\lstick{$\ket{0}$} & & \targ{} & & & & & \ctrl{2} & & & \targ{} & & \targ{} & \\
\lstick{$\ket{0}_{a}$} & & & & \targ{} & \targ{} & & & \ctrl{-3} & \ctrl{-2} & \octrl{-1} & & & \\
\lstick{$\ket{0}_{b}$} & & & & & & \targ{} & \targ{} & \octrl{-1} & \ctrl{-1} & \ctrl{-1} & & &
\end{quantikz}
"""

FIGS["repetition"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.25cm, column sep=0.32cm]
\lstick{$d_0$} & \gate[style=g]{R_Y(\theta)} & \ctrl{2} & \ctrl{4} & \gate[style=e]{\mathcal{E}_p}\gategroup[5,steps=5,style=grp]{\small one round, repeated $r$ times} & \ctrl{1} & & & & \meter{} \\
\lstick{$a_0\,\ket{0}$} & & & & & \targ{} & \targ{} & & \meter{} & \\
\lstick{$d_1$} & & \targ{} & & \gate[style=e]{\mathcal{E}_p} & & \ctrl{-1} & \ctrl{1} & & \meter{} \\
\lstick{$a_1\,\ket{0}$} & & & & & & & \targ{} & \meter{} & \\
\lstick{$d_2$} & & & \targ{} & \gate[style=e]{\mathcal{E}_p} & & & \ctrl{-1} & & \meter{}
\end{quantikz}\\[4pt]
\small ancillas measure $Z_{d_0}Z_{d_1}$ and $Z_{d_1}Z_{d_2}$ each round; the logical bit is the majority of $d_0d_1d_2$.
\end{tabular}
"""

FIGS["steane"] = r"""
\begin{quantikz}[row sep=0.18cm, column sep=0.4cm]
\lstick{$\ket{\psi}$} & \gate[7,style=g]{U_{\rm enc}}\gategroup[7,steps=1,style=grp]{\small encode $[[7,1,3]]$} & \gate[7,style=e]{E} & \meter{} & \setwiretype{c} \rstick[7]{Hamming syndrome\\ $\to$ correct one bit\\ $\to$ parity $=Z_L$} \\
\lstick{$\ket{0}$} & & & \meter{} & \setwiretype{c} \\
\lstick{$\ket{0}$} & & & \meter{} & \setwiretype{c} \\
\lstick{$\ket{0}$} & & & \meter{} & \setwiretype{c} \\
\lstick{$\ket{0}$} & & & \meter{} & \setwiretype{c} \\
\lstick{$\ket{0}$} & & & \meter{} & \setwiretype{c} \\
\lstick{$\ket{0}$} & & & \meter{} & \setwiretype{c}
\end{quantikz}
"""

FIGS["five"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.25cm, column sep=0.45cm]
\lstick{$\ket{\psi}$} & \gate[5,style=g]{U_{\rm enc}} & \gate[5,style=e]{E} & \gate[5,style=g]{U_{\rm enc}^{\dagger}} & \gate[style=q]{P_{\rm corr}} & \meter{} \\
\lstick{$\ket{0}$} & & & & \ctrl{-1}\wire[d][3]{c} & \meter{} \\
\lstick{$\ket{0}$} & & & & & \meter{} \\
\lstick{$\ket{0}$} & & & & & \meter{} \\
\lstick{$\ket{0}$} & & & & & \meter{}
\end{quantikz}\\[4pt]
\small $U_{\rm enc}^{\dagger}EU_{\rm enc}$ = (logical Pauli on $q_0$) $\otimes$ (syndrome $s$ on $q_1$–$q_4$); a 16-entry lookup gives $P_{\rm corr}$; $q_0$ reads $Z_L$.
\end{tabular}
"""

FIGS["surface"] = r"""
\begin{tikzpicture}[scale=1.25, every node/.style={font=\small}]
\fill[qxsand,draw=qxink] (0,0) rectangle (2,2);
\node at (1,1) {$X_0X_1X_2X_3$};
\fill[qxtint,draw=qxink] (0,2) arc (180:0:1 and 0.55) -- cycle;
\node at (1,2.27) {$Z_0Z_1$};
\fill[qxtint,draw=qxink] (2,0) arc (0:-180:1 and 0.55) -- cycle;
\node at (1,-0.27) {$Z_2Z_3$};
\foreach \p/\l in {(0,2)/0,(2,2)/1,(0,0)/2,(2,0)/3}{\fill[qxink] \p circle (0.09); }
\node[above left] at (0,2) {$q_0$}; \node[above right] at (2,2) {$q_1$};
\node[below left] at (0,0) {$q_2$}; \node[below right] at (2,0) {$q_3$};
\draw[qxblue,very thick,dashed] (-0.35,2) -- (-0.35,0) node[midway,left]{$Z_L=Z_0Z_2$};
\draw[qxblue!60!red,very thick,dashed] (0,2.85) -- (2,2.85) node[midway,above]{$X_L=X_0X_1$};
\end{tikzpicture}
"""

FIGS["shor"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.3cm, column sep=0.32cm]
\lstick{$\ket{0}$} & \gate[style=g]{H} & & & \ctrl{3} & \gate[3,style=g]{\mathrm{QFT}^{\dagger}} & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & & \ctrl{2} & & & \meter{} \\
\lstick{$\ket{0}$} & \gate[style=g]{H} & \ctrl{1} & & & & \meter{} \\
\lstick{$\ket{1}$} & \qwbundle{4} & \gate[style=o]{\times a} & \gate[style=o]{\times a^{2}} & \gate[style=o]{\times a^{4}} & &
\end{quantikz}\\[4pt]
\small multiplications $\bmod 15$; the counting register reads $y\approx 2^t s/r$, and continued fractions give the order $r$.
\end{tabular}
"""

FIGS["hhl"] = r"""
\begin{quantikz}[row sep=0.3cm, column sep=0.3cm]
\lstick{ancilla $\ket{0}$} & & & \gate[style=q]{R_Y(\phi_\lambda)} & & \meter{} \rstick{post-select 1} \\
\lstick{clock $\ket{0}^{\otimes t}$} & \qwbundle{t} & \gate[2,style=g]{\mathrm{QPE}(e^{iAt_0})} & \ctrl{-1} & \gate[2,style=g]{\mathrm{QPE}^{\dagger}} & \\
\lstick{$\ket{b}$} & & & & & \rstick{$\propto A^{-1}\ket{b}$}
\end{quantikz}
"""

FIGS["zne_fold"] = r"""
\begin{tabular}{c}
\begin{quantikz}[row sep=0.35cm, column sep=0.45cm]
& \gate[2,style=g]{U} & \gate[2,style=o]{U^{\dagger}} & \gate[2,style=g]{U} & \meter{} \\
& & & & \meter{}
\end{quantikz}\\[4pt]
\small global folding $U(U^{\dagger}U)^{k}$ multiplies the noise by $\lambda=1+2k$ and leaves the ideal result unchanged.
\end{tabular}
"""


# Captions are typeset by the app (and by the manual) at text size, under the diagram, instead of inside the
# picture: a caption line inside the picture widens it, so the circuit itself has to be drawn smaller.
EXTRA_CAPTIONS = {"surface": r"$[[4,1,2]]$: detects any single error, corrects none; ancillas $q_4,q_5,q_6$ measure "
                             r"the three checks."}
_CAP = re.compile(r"\\\\\[\d+pt\]\s*\n\\small\s+(.*?)\n\\end\{tabular\}\s*$", re.S)


def split_caption(name: str, body: str) -> tuple:
    """(TeX body without its caption line, caption as Markdown with $…$ math or '')."""
    m = _CAP.search(body.strip())
    cap = EXTRA_CAPTIONS.get(name, "")
    if m:
        cap = m.group(1).strip()
        body = body.strip()[:m.start()] + "\n\\end{tabular}\n"
    return body, to_markdown(cap)


def to_markdown(tex: str) -> str:
    """A TeX caption line → Markdown that KaTeX renders (no \\ket, no spacing macros outside math)."""
    md = re.sub(r"\\ket\{([^{}]*)\}", r"\\lvert \1\\rangle", tex)
    md = md.replace("\\quad", " · ").replace("\\,", " ")
    return re.sub(r"\s+", " ", md).strip()


def build(name: str, body: str, workdir: str) -> str:
    tex = os.path.join(workdir, name + ".tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + body + "\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=workdir,
                       capture_output=True, text=True, timeout=180, encoding="utf-8")
    if r.returncode != 0:
        err = [l for l in r.stdout.splitlines() if l.startswith("!")]
        raise RuntimeError("%s: %s" % (name, "; ".join(err[:3]) or r.stdout[-400:]))
    svg = os.path.join(OUT, name + ".svg")
    subprocess.run(["pdftocairo", "-svg", os.path.join(workdir, name + ".pdf"), svg], check=True)
    return svg


def main(names=None):
    for _s in (sys.stdout, sys.stderr):       # printed text is UTF-8 on every platform
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    work = tempfile.mkdtemp(prefix="qxfig_")
    cap_path = os.path.join(OUT, "captions.json")
    captions = json.load(open(cap_path, encoding="utf-8")) if os.path.exists(cap_path) else {}
    try:
        for name, body in FIGS.items():
            if names and name not in names:
                continue
            body, cap = split_caption(name, body)
            captions[name] = cap
            with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:      # the source, next to the SVG
                f.write(PREAMBLE + body + "\n\\end{document}\n")
            print("built", build(name, body, work))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump({k: captions[k] for k in sorted(captions)}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
