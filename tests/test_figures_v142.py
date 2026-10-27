"""The thesis figure system of scripts/dirA_figures.py (1.4.2): type presets, the diverging scale, the fidelity gap,
non-overlapping direct labels, the χ² ladder and the source line."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture()
def F():
    import matplotlib
    matplotlib.use("Agg")
    import dirA_figures
    dirA_figures.setup_matplotlib()
    yield dirA_figures
    import matplotlib.pyplot as plt
    plt.close("all")


def test_font_presets(F):
    import matplotlib as mpl
    for name in F.FONT_PRESETS:
        assert F.setup_matplotlib(name) == name
    assert mpl.rcParams["font.family"] == ["sans-serif"] and mpl.rcParams["pdf.fonttype"] == 42
    F.setup_matplotlib("times")
    assert mpl.rcParams["font.serif"][0] == "TeX Gyre Termes" and mpl.rcParams["font.size"] == 8
    with pytest.raises(ValueError):
        F.setup_matplotlib("comic")


def test_palettes_are_mirrored_and_monotone(F):
    """The diverging scale: grey midpoint, lightness falling monotonically towards both ends (OKLab L)."""
    def L(hexcol):
        rgb = np.array([int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        m = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                      [0.0883024619, 0.2817188376, 0.6299787005]]) @ lin
        lms = np.cbrt(m)
        return 0.2104542553 * lms[0] + 0.7936177850 * lms[1] - 0.0040720468 * lms[2]
    Ls = [L(c) for c in F.DIV_COLORS]
    mid = len(Ls) // 2
    assert np.argmax(Ls) == mid
    assert all(np.diff(Ls[:mid + 1]) > 0) and all(np.diff(Ls[mid:]) < 0)
    assert all(abs(Ls[mid - k] - Ls[mid + k]) < 0.03 for k in range(1, mid + 1))      # mirrored arms
    assert len(set(F.FAMILY_MARKERS.values())) == len(F.FAMILY_MARKERS)              # one shape per family


def test_fidelity_gap_recovers_the_rates(F):
    rng = np.random.default_rng(0)
    n2 = rng.integers(5, 300, 60).astype(float)
    D = pd.DataFrame(dict(method="RAW", predictor="depolarizing", status="ok", family=rng.choice(["qft", "pe"], 60),
                          two_qubit=n2, fc_obs=0.98 * 0.99 ** n2, fc_pred=0.98 * 0.996 ** n2))
    fig = F.fidelity_gap(D)
    assert fig.gap_rates["device"] == pytest.approx(0.01, rel=1e-6)
    assert fig.gap_rates["model"] == pytest.approx(0.004, rel=1e-6)
    fig = F.fidelity_gap(D.assign(fc_obs=0.99 ** n2), fit="origin")
    assert fig.gap_rates["device"] == pytest.approx(0.01, rel=1e-6)


def test_place_labels_do_not_overlap(F):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs, ys = np.array([0.5, 0.51, 0.52, 0.5]), np.array([0.5, 0.5, 0.51, 0.52])
    anns = F.place_labels(ax, xs, ys, ["QFT", "brickwork", "context probe", "BV"])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    from matplotlib.text import Text
    boxes = [Text.get_window_extent(a, r) for a in anns]               # the text, without its leader line
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (i, j)


def test_ladder_and_source_line(F):
    L = pd.DataFrame(dict(model=["none", "calibration (priced)", "context", "calibration + context"],
                          chi2=[400.0, 70, 60, 50], dof=[20, 19, 16, 15], chi2_per_dof=[20.0, 3.7, 3.75, 3.3]))
    fig = F.attribute(F.ladder_chart(L, title="t", verdict="context"), "emulated: fake_fez + planted error")
    assert any(t.get_text().startswith("emulated") for t in fig.texts)
