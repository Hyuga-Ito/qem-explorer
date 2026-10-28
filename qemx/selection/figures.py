r"""Shared figure machinery of the protocol-selection study (1.6.0): palettes and their color-vision check,
the saving convention, and the certificates figure. The six main figures, the other supplementary figures and the
tables are in ``qemx.selection.plots``. Everything is drawn only from saved source-data tables
(``qemx.selection.analysis``) — never from a live computation — so that every figure regenerates from the results
directory and a test can check figure against data.

Each figure is written as vector PDF and SVG plus a 300-dpi PNG, with its source data (CSV, exactly the numbers
plotted), its caption and a short interpretation note (``<name>.md``). Labels: every figure states whether it shows
simulated observations (emulated fake-device trials), model predictions or analytic illustrations; none shows
hardware data.

Palettes (validated for color-vision deficiency with ``palette_report``):

* categorical — one fixed color per protocol (Okabe–Ito based, as in the 1.5 study figures; ZNE variants share the
  blue family and are told apart by lightness *and* marker; RAW is a light neutral gray), and one per policy kind.
  Smallest pairwise OKLab ΔE×100: 15.3 (normal vision), 9.6 (protan), 7.6 (deutan; CDR–NIL — always paired with
  distinct markers), 8.5 (tritan); ``palette_report`` recomputes them and a test keeps them;
* sequential — a single-hue blue ramp (magnitudes such as regret);
* diverging — red ↔ blue around a neutral gray midpoint (signed differences such as Δ log risk).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

PROTOCOL_COLORS = {"RAW": "#C8C8C8", "RO": "#E69F00", "ZNE-L": "#56B4E9", "ZNE-R": "#0072B2", "ZNE-E": "#003F63",
                   "PEC": "#D55E00", "CDR": "#009E73", "NIL": "#CC79A7"}
PROTOCOL_MARKERS = {"RAW": "o", "RO": "s", "ZNE-L": "D", "ZNE-R": "^", "ZNE-E": "v", "PEC": "P", "CDR": "X",
                    "NIL": "*"}
# 1.8.0 candidates. Seventeen categories cannot all keep the ≥ 15 OKLab separation of the eight 1.6/1.7 colours
# (unchanged above), so the candidates use a composite encoding: colour AND a unique marker, with candidate names on
# the axes wherever the figure allows. The nine colours were chosen by a max–min search over OKLCh (lightness
# 0.35–0.79 for contrast on white) given the fixed eight: smallest pairwise ΔE×100 over all 17 colours 10.4 (normal),
# 7.1 protan, 6.4 deutan, 7.0 tritan (tests/test_selection_eval.py). The first 1.8.0 colours (family hues with a
# lightness step) failed that check (4.1 normal: ZNE-R against ZNE-E-B) and were replaced before any 1.8.0 figure.
PROTOCOL_COLORS.update({"RO-TREX": "#7B2004", "RO-TW": "#96BC79", "RO-DD": "#4C6339", "ZNE-L-TW": "#A974F7",
                        "ZNE-E-B": "#6325A5", "PEC-TW": "#BA1D2A", "FR": "#4C323A", "SV": "#8F4D70",
                        "ZNE-L∘SV": "#B1389B"})
PROTOCOL_MARKERS.update({"RO-TREX": "h", "RO-TW": "<", "RO-DD": ">", "ZNE-L-TW": "d", "ZNE-E-B": "1",
                         "PEC-TW": "p", "FR": "x", "SV": "H", "ZNE-L∘SV": "8"})
POLICY_STYLE = {"RAW": ("#C8C8C8", "o", ":"), "SBS": ("#E69F00", "s", "--"), "SBS-B": ("#D55E00", "D", "--"),
                "SIZE-B": ("#CC79A7", "^", "-."), "MODEL": ("#0072B2", "o", "-"),
                "MODEL+ABSTAIN": ("#009E73", "v", "-"), "ALL-CANDIDATE": ("#000000", "", ":")}
SEQ_HEX = ("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
DIV_HEX = ("#B2182B", "#EF8A62", "#E6E6E6", "#67A9CF", "#2166AC")
LABEL_SIM = "Simulated observations (emulated fake-device trials); no hardware data."


# ------------------------------------------------------------------ palette checks
def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _oklab(rgb_lin):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = np.cbrt(M1 @ rgb_lin)
    return M2 @ lms


# Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear RGB
_CVD = {"protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
        "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                            [-0.011820, 0.042940, 0.968881]]),
        "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]])}


def palette_report(hexes) -> dict:
    """Smallest pairwise OKLab ΔE (×100) of a categorical palette under normal vision and simulated protan, deutan
    and tritan vision (all pairs, not only neighbors)."""
    lin = [_srgb_to_lin(_hex(h)) for h in hexes]
    out = {}
    for kind, M in [("normal", np.eye(3))] + list(_CVD.items()):
        lab = [_oklab(np.clip(M @ c, 0, 1)) for c in lin]
        d = [100 * float(np.linalg.norm(lab[i] - lab[j])) for i in range(len(lab)) for j in range(i + 1, len(lab))]
        out[kind] = min(d)
    return out


# ------------------------------------------------------------------ helpers
def _plt():
    import logging
    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "qemx-selection", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "font.family": "DejaVu Sans",
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E5E5",
                         "grid.linewidth": 0.5, "svg.fonttype": "none", "pdf.fonttype": 42})
    return plt


def _save(fig, out_dir, name, data: dict, caption: str, interpretation: str, config: dict | None = None) -> dict:
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for ext in ("pdf", "svg", "png"):
        p = os.path.join(out_dir, "%s.%s" % (name, ext))
        meta = {"pdf": {"Creator": "qemx.selection.figures", "Producer": None, "CreationDate": None},
                "svg": {"Creator": "qemx.selection.figures", "Date": None},
                "png": {"Software": None}}[ext]                 # no dates: regeneration is byte-identical
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight", metadata=meta)
        paths[ext] = p
    for key, df in data.items():
        p = os.path.join(out_dir, "%s__%s.csv" % (name, key))
        pd.DataFrame(df).to_csv(p, index=False, encoding="utf-8")
        paths["data:" + key] = p
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write("# %s\n\n**Caption.** %s\n\n**Interpretation.** %s\n\n**Data.** %s\n" % (
            name, caption, interpretation, ", ".join("`%s__%s.csv`" % (name, k) for k in data)))
        if config:
            f.write("\n**Configuration.** `%s`\n" % json.dumps(config, sort_keys=True))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def _policy_kind(name: str) -> str:
    for k in ("MODEL+ABSTAIN", "MODEL", "ALL-CANDIDATE", "SIZE-B", "SBS-B", "SBS", "RAW"):
        if name.startswith(k):
            return k
    return name


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("qemx_seq", SEQ_HEX)


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("qemx_div", DIV_HEX)


INTERVAL_KINDS = [  # (kind, short label, statement kind, colour)
    ("split-conformal", "split-conformal", "split-conformal", "#009E73"),
    ("split-conformal-simultaneous", "split-conf.\nsimult.", "split-conformal", "#009E73"),
    ("cv-residual", "CV\nresidual", "cv-residual", "#0072B2"),
    ("cv-residual-simultaneous", "CV resid.\nsimult.", "cv-residual", "#0072B2"),
    ("joint (simultaneous)", "CV resid.\njoint", "cv-residual", "#0072B2"),
    ("residual-gaussian", "residual\nGaussian", "empirical", "#D55E00"),
]


def supp_coverage_certificates(adir, out_dir) -> dict:
    """(a) Empirical coverage of the log-risk intervals per held-out fold against the nominal level, grouped by the
    kind of statement each interval supports; (b) radii of the Hoeffding and empirical-Bernstein bounds (clipped
    estimator, H = 4; units: trials, or design means of learning protocols under fresh designs) against the MSE of
    the chosen protocol, per budget; (c) realized regret against the post-hoc 2ε check."""
    import pandas as pd
    plt = _plt()
    ic = pd.read_csv(os.path.join(adir, "interval_coverage.csv"), encoding="utf-8")
    ce = pd.read_csv(os.path.join(adir, "certificates.csv"), encoding="utf-8")
    known = [k for k, *_ in INTERVAL_KINDS]
    kinds = [k for k in known if k in set(ic.kind)] + sorted(set(ic.kind) - set(known))
    meta = {k: (s, st, c) for k, s, st, c in INTERVAL_KINDS}
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0), layout="constrained", gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    wid = {}
    for i, k in enumerate(kinds):
        d = ic[ic.kind == k]
        c = meta.get(k, (k, "", "#555555"))[2]
        jit = (np.arange(len(d)) - (len(d) - 1) / 2.0) * 0.06
        ax[0].scatter(np.full(len(d), i) + jit, d.coverage, s=10, color=c, lw=0, zorder=3)
        ax[0].hlines(d.coverage.mean(), i - 0.25, i + 0.25, color=c, lw=1.6)
        w = d.width_log10.replace([np.inf, -np.inf], np.nan) if "width_log10" in d.columns else pd.Series(dtype=float)
        wid[k] = float(w.median()) if w.notna().any() else float("inf")
    nominal = float(ic.nominal.iloc[0]) if len(ic) else 0.9
    ax[0].axhline(nominal, color="#333333", lw=0.8, ls="--")
    ax[0].set_xticks(np.arange(len(kinds)), [meta.get(k, (k,))[0].replace("\n", " ") for k in kinds], fontsize=6,
                     rotation=30, ha="right", rotation_mode="anchor")
    ax[0].set_ylabel("empirical coverage (per fold)")
    ax[0].set_ylim(min(0.5, float(ic.coverage.min()) - 0.05) if len(ic) else 0.5, 1.02)
    ax[0].set_title("(a) interval coverage, measured", loc="left", fontsize=7)
    for st, c, lab in (("split-conformal", "#009E73", "split-conformal: ≥ 1−α if circuits exchangeable"),
                       ("cv-residual", "#0072B2", "CV residual: calibrated in CV, no guarantee"),
                       ("empirical", "#D55E00", "Gaussian: no guarantee")):
        ax[0].plot([], [], "o", color=c, ms=3.5, label=lab)
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    hb = ce[ce.method.isin(["hoeffding", "bernstein"])]
    col = "radius" if "radius" in hb.columns else "halfwidth"
    agg = dict(radius=(col, "median"), informative=("informative", "mean"), mse=("mse_clipped", "median"))
    if "units_chosen" in hb.columns:
        agg["units"] = ("units_chosen", "median")
    g = hb.groupby(["method", "B"]).agg(**agg).reset_index()
    for m, c, mk in (("hoeffding", "#0072B2", "o"), ("bernstein", "#009E73", "s")):
        d = g[g.method == m].sort_values("B")
        ax[1].plot(d.B, d.radius, mk + "-", color=c, ms=3.5, lw=1.4, label="%s radius t" % m)
    d = g[g.method == "hoeffding"].sort_values("B")
    ax[1].plot(d.B, d.mse, "^:", color="#555555", ms=3.5, lw=1.2, label="MSE of the chosen protocol")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("budget B (shots)")
    ax[1].set_ylabel("median over tasks")
    ax[1].set_title("(b) finite-sample bounds (clipped)", loc="left", fontsize=7)
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    te = ce[(ce.method == "two-eps") & ce.bound.notna() & ce.regret.notna()].copy()
    te = te[(te.bound > 0)]
    floor = 1e-9
    ax[2].scatter(np.maximum(te.bound, floor), np.maximum(te.regret, floor), s=5, color="#0072B2", alpha=0.5, lw=0)
    if len(te):
        lo = max(floor, float(min(te.bound.min(), max(te.regret.min(), floor))))
        hi = float(max(te.bound.max(), te.regret.max()))
        ax[2].plot([lo, hi], [lo, hi], color="#333333", lw=0.8)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("2ε (selection-half error,\npost hoc)")
    ax[2].set_ylabel("realized regret (floor 10$^{-9}$)")
    ax[2].set_title("(c) 2ε: a post-hoc check", loc="left", fontsize=7)
    fig.suptitle(LABEL_SIM, fontsize=6.5, color="#555555")
    holds = float(te.holds.mean()) if len(te) else float("nan")
    inf_share = float(hb.informative.mean()) if len(hb) else float("nan")
    units = (", ".join("B=%s: %g" % (int(r_.B), r_.units) for _, r_ in g[g.method == "hoeffding"].iterrows())
             if "units" in g.columns else "trials")
    groups = int(ic.groups.median()) if "groups" in ic.columns and len(ic) else None
    cap = ("Interval coverage and finite-sample bounds; each panel says what kind of statement it supports. (a) "
           "Measured coverage of log10-risk intervals per held-out fold (points; bar: mean over %d folds%s), against "
           "the nominal level (dashed): group split-conformal intervals (the model refitted on part of the training "
           "circuits, calibrated on the rest; per protocol and Bonferroni-simultaneous; a ≥ 1 − α guarantee per "
           "circuit under exchangeability of base circuits — measured here, not assumed), cross-validated residual "
           "quantiles (marginal, simultaneous and joint over the feasible protocols of a row; calibrated in cross-"
           "validation, no finite-sample guarantee) and the residual-Gaussian interval (no guarantee). Every "
           "interval targets the evaluated label log10(R̂ + 10⁻⁶), which contains label noise. Median full widths in "
           "log10: %s. (b) Median radius t of the Hoeffding and empirical-Bernstein bounds (α = 0.05 over the K "
           "feasible protocols) on the clipped squared error (estimates projected onto [−1, 1], H = 4) — a "
           "finite-sample bound for the clipped estimator, not for the unclipped MSE that the study reports; units: "
           "independent trials, or for the learning candidates (CDR, NIL; FR in 1.8.0) under fresh designs their design means (median units of the "
           "chosen protocol: %s); share of informative bounds (upper bound of the chosen protocol below RAW's lower "
           "bound): %.3f. (c) Realized regret against 2ε with ε the largest selection-half error: a post-hoc check "
           "computed with the evaluation labels, not a certificate; it holds on %.3f of rows (regrets below 10⁻⁹ "
           "drawn at 10⁻⁹). %s" % (
               int(ic.fold.nunique()) if "fold" in ic.columns else 0,
               (", about %d base circuits each" % groups) if groups else "",
               ", ".join("%s %.2f" % kv for kv in wid.items() if np.isfinite(kv[1])) or "n/a", units, inf_share,
               holds, LABEL_SIM))
    interp = ("Bounded-loss certificates are orders of magnitude wider than the risks they would certify at these "
              "trial counts; conformal intervals are calibrated marginally over base circuits, not per task; the 2ε "
              "check is descriptive.")
    return _save(fig, out_dir, "supp_coverage_certificates", dict(coverage=ic, certificates=g, two_eps=te), cap,
                 interp)


def markdown_table(df) -> str:
    """A Markdown table of a DataFrame (numbers to 4 significant digits), without optional dependencies."""
    def fmt(x):
        if isinstance(x, float):
            return "" if math.isnan(x) else "%.4g" % x
        return str(x).replace("|", "\\|")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"
