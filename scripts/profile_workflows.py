#!/usr/bin/env python3
"""Reproducible timing and memory profile of representative workflows (1.5).

Each workflow runs in its own fresh Python process (cold caches), ``--repeat`` times; reported: median and range of
wall-clock seconds, and the process's peak resident memory. Works on any checkout (``--root``), so the same script
profiles the audited 1.4.2 tree and this one:

    python scripts/profile_workflows.py --root . --out profile_after.json
    python scripts/profile_workflows.py --root ../qem-explorer-1.4.2 --out profile_before.json

Workflows (all offline, NumPy engine unless stated):

* ``import``       import the app's modules (qemx + ui) — part of cold start
* ``app_first``    AppTest: Home page first render
* ``playground``   AppTest: Playground first render, then a warm rerun
* ``run_default``  AppTest: Playground "Run simulation" with the default contenders (QFT-3 template, defaults)
* ``compare``      engine: qemx.recipes.compare on QFT(4) with six presets at 100 000 shots
* ``learning``     engine: CDR (8 × 1024 training shots) on QFT(3) at 100 000 shots (protocol moments included)
* ``export``       qemx.bundle.make_zip with 6 Plotly figures, PNG + SVG rendering
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

CODE = {
    "import": r"""
import time; t=time.perf_counter()
import qemx.recipes, qemx.templates, qemx.compile, ui.state, ui.results
T=time.perf_counter()-t
""",
    "app_first": r"""
import time
import qiskit_ibm_runtime
from streamlit.testing.v1 import AppTest
t=time.perf_counter(); at=AppTest.from_file("app.py", default_timeout=900).run(); T=time.perf_counter()-t
assert not at.exception
""",
    "playground": r"""
import time
import qiskit_ibm_runtime
from streamlit.testing.v1 import AppTest
at=AppTest.from_file("app.py", default_timeout=900).run()
t=time.perf_counter(); at.switch_page("pages/1_Playground.py").run(); T1=time.perf_counter()-t
t=time.perf_counter(); at.run(); T2=time.perf_counter()-t
assert not at.exception
T={"first": T1, "warm_rerun": T2}
""",
    "run_default": r"""
import time
import qiskit_ibm_runtime
from streamlit.testing.v1 import AppTest
at=AppTest.from_file("app.py", default_timeout=900).run()
at.switch_page("pages/1_Playground.py").run()
b=[x for x in at.button if x.key and x.key.startswith("sb_run_")][0]
t=time.perf_counter(); b.click().run(); T=time.perf_counter()-t
assert not at.exception
""",
    "compare": r"""
import time
from qemx import templates as Tm, compile as CP
from qemx.noise import NoiseSpec
from qemx.recipes import compare, preset
tpl=Tm.build("qft", {"n": 4}); comp=CP.build(tpl.spec, relabel=tpl.relabel)
rs=[preset(n) for n in ("Unmitigated","ZNE · Richardson","ZNE · linear","PEC · inverse","Readout · TREX","Virtual distillation (M=2)")]
t=time.perf_counter(); compare(comp, NoiseSpec(p1=0.001,p2=0.01,ro01=0.02,ro10=0.03), tpl.observable, rs, shots=100000); T=time.perf_counter()-t
""",
    "learning": r"""
import time
from qemx import templates as Tm, compile as CP
from qemx.noise import NoiseSpec
from qemx.methods import Recipe
from qemx.recipes import compare
tpl=Tm.build("qft", {"n": 3}); comp=CP.build(tpl.spec, relabel=tpl.relabel)
rc=Recipe.make("learning", features="cdr", model="ols", n_train=8, train_shots=1024)
t=time.perf_counter(); compare(comp, NoiseSpec(p1=0.003,p2=0.03), tpl.observable, [rc], shots=100000, seed=13); T=time.perf_counter()-t
""",
    "export": r"""
import time
import plotly.graph_objects as go
from qemx.bundle import make_zip
figs={"f%d"%i: go.Figure(go.Scatter(x=list(range(50)), y=[j*j for j in range(50)])) for i in range(6)}
t=time.perf_counter(); make_zip("profile", figures=figs); T=time.perf_counter()-t
""",
}
TAIL = r"""
import json, resource, sys
r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("@@RESULT@@"+json.dumps({"T": T, "peak_mb": r/1024.0}))
"""


def run_one(root: str, name: str, timeout: float) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", CODE[name] + TAIL], cwd=root, env=dict(env, PYTHONIOENCODING="utf-8"), capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8")
    wall = time.perf_counter() - t0
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("@@RESULT@@")]
    if p.returncode != 0 or not line:
        return dict(error=(p.stderr or p.stdout)[-600:], process_s=wall)
    d = json.loads(line[0][len("@@RESULT@@"):])
    d["process_s"] = wall
    return d


def main(argv=None) -> int:
    for _s in (sys.stdout, sys.stderr):       # printed text is UTF-8 on every platform
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    res = dict(root=root, python=sys.version.split()[0], platform=platform.platform(), cpu_count=os.cpu_count(),
               repeat=a.repeat, created=time.strftime("%Y-%m-%dT%H:%M:%S"), workflows={})
    for name in (a.only or list(CODE)):
        runs = [run_one(root, name, a.timeout) for _ in range(a.repeat)]
        ok = [r for r in runs if "error" not in r]
        entry = dict(runs=runs)
        if ok:
            if isinstance(ok[0]["T"], dict):
                for k in ok[0]["T"]:
                    xs = [r["T"][k] for r in ok]
                    entry[k] = dict(median=statistics.median(xs), min=min(xs), max=max(xs))
            else:
                xs = [r["T"] for r in ok]
                entry["seconds"] = dict(median=statistics.median(xs), min=min(xs), max=max(xs))
            entry["peak_mb"] = max(r["peak_mb"] for r in ok)
        res["workflows"][name] = entry
        print(name, {k: v for k, v in entry.items() if k != "runs"}, flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
