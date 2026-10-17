"""'Download all': one ZIP with every table, the raw samples, the figures and the state.

Layout of the archive::

    README.txt                 what is inside, when and how it was made
    tables/<name>.csv|.tex     every table (LaTeX with booktabs-ready tabular)
    data/<name>.csv            samples of repeated experiments and other raw arrays
    figures/<name>.png         high-resolution images (3× scale) — needs Kaleido with a Chrome/Chromium
    figures/<name>.svg         vector images for papers
    figures/<name>.json        the Plotly figure itself, only when images cannot be rendered
                               (plotly.io.read_json(path).show() reopens it interactively)
    state/qem_explorer_state.json   the app state (no credentials are ever part of it)
    script/qem_explorer_export.py   the standalone Qiskit script, when available

Images are rendered in a separate, time-limited process (one Kaleido browser for all figures once the first image
proved a browser works). If Kaleido or a browser is missing, crashes or hangs, the archive still contains every table,
the data, the state and each figure's Plotly JSON, plus every image finished before the failure, and the README says
how to enable PNG/SVG. No HTML files are written.
"""
from __future__ import annotations

import datetime as _dt
import glob
import io
import json
import os
import re
import threading
import zipfile

import pandas as pd

# Kaleido needs a Chrome/Chromium. Use the one Playwright ships with when present (sandbox, CI).
if not os.environ.get("BROWSER_PATH"):
    for _c in sorted(glob.glob("/opt/pw-browsers/chromium*/chrome-linux/chrome")):
        os.environ["BROWSER_PATH"] = _c
        break


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_").lstrip(".")
    return (s or "item")[:80]


def safe_path(s: str) -> str:
    """A name with optional sub-folders ("cost_precision/rmse_vs_shots"), each part sanitized (no '..')."""
    parts = [p for p in str(s).replace("\\", "/").split("/") if p.strip() and p.strip(". ")]
    return "/".join(safe_name(p) for p in parts) or "item"


def _unique(name: str, used: set) -> str:
    base, k = name, 2
    while name in used:
        name = "%s_%d" % (base, k)
        k += 1
    used.add(name)
    return name


_RENDER_LOCK = threading.Lock()          # one renderer process at a time (downloads run in their own threads)
RENDER_TIMEOUT_S = 90.0                  # wall-clock limit of one export's image rendering (QEMX_RENDER_TIMEOUT)
RENDER_START_TIMEOUT_S = 30.0            # … of which the renderer may spend this long before its first image


def render_images(figs: dict, scale: float = 3.0, formats=("png", "svg"), timeout: float | None = None) -> tuple:
    """({(name, fmt): bytes}, error message or ''). Rendering runs in a child process (``qemx.render_worker``) with a
    wall-clock limit: a missing or crashing Chrome, a dead renderer or a hang costs at most ``timeout`` seconds
    (default RENDER_TIMEOUT_S, env QEMX_RENDER_TIMEOUT; at most RENDER_START_TIMEOUT_S before the first image), and
    every image finished before that is kept. The numerical tables, configuration and figure JSON never depend on this
    step: make_zip writes them regardless."""
    out, err = {}, ""
    if not figs:
        return out, err
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    limit = float(os.environ.get("QEMX_RENDER_TIMEOUT", timeout if timeout is not None else RENDER_TIMEOUT_S))
    start_limit = min(limit, RENDER_START_TIMEOUT_S)
    names = list(figs)
    stems = ["f%03d" % i for i in range(len(names))]
    total = len(names) * len(formats)
    jobdir = tempfile.mkdtemp(prefix="qemx-render-")
    proc = None
    with _RENDER_LOCK:
        try:
            with open(os.path.join(jobdir, "job.json"), "w", encoding="utf-8") as f:
                json.dump(dict(items=[[st, figs[nm].to_json()] for st, nm in zip(stems, names)],
                               formats=list(formats), scale=float(scale), server=True), f)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env = dict(os.environ)
            env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            with open(os.path.join(jobdir, "worker.log"), "wb") as log:
                proc = subprocess.Popen([sys.executable, "-m", "qemx.render_worker", jobdir], cwd=jobdir, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                t0 = time.monotonic()

                def finished():
                    return sum(os.path.exists(os.path.join(jobdir, "%s.%s" % (st, fm))) for st in stems
                               for fm in formats)
                while True:
                    if proc.poll() is not None:
                        break
                    if finished() == total:                  # every image is written: shutdown is not our concern
                        break
                    el = time.monotonic() - t0
                    if el > limit or (el > start_limit and finished() == 0):
                        err = ("the image renderer did not finish within %.0f s (%d of %d images rendered)"
                               % (el, finished(), total))
                        break
                    time.sleep(0.05)
            for st, nm in zip(stems, names):
                for fm in formats:
                    path = os.path.join(jobdir, "%s.%s" % (st, fm))
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            out[(nm, fm)] = f.read()
            if not err and len(out) < total and not os.path.exists(os.path.join(jobdir, "done")):
                why = ""
                ep = os.path.join(jobdir, "error.txt")
                if os.path.exists(ep):
                    with open(ep, encoding="utf-8", errors="replace") as f:
                        why = f.read().strip()
                err = (why or "the image renderer stopped unexpectedly (exit code %s)" % proc.returncode) + (
                    " (%d of %d images rendered)" % (len(out), total) if out else "")
        except Exception as e:                           # noqa: BLE001 - the export continues without images
            err = "%s: %s" % (type(e).__name__, str(e)[:300])
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(5)
                except Exception:                        # noqa: BLE001
                    pass
            shutil.rmtree(jobdir, ignore_errors=True)
    return out, err


def make_zip(title: str, tables: dict | None = None, data: dict | None = None, figures: dict | None = None,
             state: dict | None = None, script: str | None = None, notes: str = "", images: bool = True,
             scale: float = 3.0) -> bytes:
    tables, data, figures = tables or {}, data or {}, figures or {}
    buf = io.BytesIO()
    used_t, used_d, used = set(), set(), set()
    manifest = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            nm = _unique(safe_path(name), used_t)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            z.writestr("tables/%s.csv" % nm, df.to_csv(index=False))
            try:
                z.writestr("tables/%s.tex" % nm, df.to_latex(index=False, float_format="%.5g"))
            except Exception:
                pass
            manifest.append("tables/%s.csv (+ .tex): %s" % (nm, name))
        for name, df in data.items():
            nm = _unique(safe_path(name), used_d)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            z.writestr("data/%s.csv" % nm, df.to_csv(index=False))
            manifest.append("data/%s.csv: %s" % (nm, name))
        fig_names = {}
        for name, fig in figures.items():
            if fig is None:
                continue
            nm = _unique(safe_path(name), used)
            fig_names[nm] = (fig, name)
        img_err = ""
        imgs = {}
        if images and fig_names:
            imgs, img_err = render_images({k: v[0] for k, v in fig_names.items()}, scale=scale)
            for (nm, fmt), b in imgs.items():
                z.writestr("figures/%s.%s" % (nm, fmt), b)
        for nm, (fig, name) in fig_names.items():
            got = [fm for fm in ("png", "svg") if (nm, fm) in imgs]
            if got == ["png", "svg"]:
                manifest.append("figures/%s.png (%.0f×) + .svg: %s" % (nm, scale, name))
            else:                                        # no (complete) rendering: the figure's Plotly JSON as well
                z.writestr("figures/%s.json" % nm, fig.to_json())
                manifest.append("figures/%s.json (Plotly figure)%s: %s" % (
                    nm, "".join(" + .%s" % fm for fm in got), name))
        if state is not None:
            z.writestr("state/qem_explorer_state.json", json.dumps(state, indent=1, default=str))
            manifest.append("state/qem_explorer_state.json: the app state (circuit, noise, recipes; no credentials)")
        if script:
            z.writestr("script/qem_explorer_export.py", script)
            manifest.append("script/qem_explorer_export.py: standalone Qiskit script reproducing the numbers")
        readme = ["QEM Explorer — %s" % title,
                  "Created %s" % _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""]
        if notes:
            readme += [notes, ""]
        readme += ["Contents:"] + ["  " + m for m in manifest]
        if img_err:
            readme += ["", "PNG/SVG images were not rendered (%s)." % img_err,
                       "Every table, the data, the state and each figure's Plotly JSON are complete regardless.",
                       "Install Kaleido (pip install kaleido) and Google Chrome, or run `plotly_get_chrome`,",
                       "then download again. Meanwhile each figure is saved as Plotly JSON:",
                       "  import plotly.io as pio; pio.read_json('figures/<name>.json').show()"]
        z.writestr("README.txt", "\n".join(readme) + "\n")
    return buf.getvalue()
