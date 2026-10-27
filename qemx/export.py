r"""Export the current UI state as one standalone Python script.

The script contains, in this order:

1. the state (circuit, inputs, basis, compilation, noise, observable, contenders) as JSON;
2. your circuit written out in plain Qiskit (``user_circuit()``), for reading and editing;
3. ``main()``: rebuilds the implemented circuit, runs every contender on Qiskit Aer (or NumPy),
   and prints the estimates next to the values the app computed;
4. ``run_on_ibm()``: the same circuits on IBM Quantum hardware (raw, ZNE, symmetry verification,
   tensored/M3 readout mitigation), credentials from the environment or a saved account;
5. an appendix: a verbatim copy of the QEM Explorer engine, so the numbers are the app's numbers.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re

from .circuit import CircuitSpec, fmt_angle

ENGINE_MODULES = ["resources.py", "budget.py", "stats.py", "circuit.py", "observables.py", "noise.py", "backends/base.py", "backends/numpy_dm.py",
                  "backends/aer.py", "engine.py", "compile.py", "transforms.py", "methods/base.py", "methods/raw.py",
                  "methods/zne.py", "methods/pec.py", "methods/learning.py", "methods/nil.py", "methods/symmetry.py",
                  "methods/vd.py", "methods/compose.py",
                  "device.py", "device_exact.py", "hamiltonian.py"]
_ROOT = os.path.dirname(os.path.abspath(__file__))
_REL = re.compile(r"^\s*from\s+\.+[\w.]*\s+import\s")


def engine_source() -> str:
    """The engine modules flattened into one namespace (relative imports removed)."""
    out = []
    for m in ENGINE_MODULES:
        lines = open(os.path.join(_ROOT, m), encoding="utf-8").read().splitlines()
        body, skip = [], False
        for ln in lines:
            if skip:
                if ")" in ln:
                    skip = False
                continue
            if ln.startswith("from __future__"):
                continue
            if _REL.match(ln):
                if "(" in ln and ")" not in ln:
                    skip = True
                continue
            body.append(ln)
        out.append("# " + "-" * 30 + " qemx/%s " % m + "-" * 30)
        out.append("\n".join(body))
    out.append("# modules share one namespace here; `device.x`, `device_exact.x` and `resources.x` refer to these functions\n"
               "import types as _types\n"
               "device = resources = device_exact = _types.SimpleNamespace(**{k: v for k, v in dict(globals()).items() "
               "if not k.startswith('__')})")
    return "\n\n".join(out)


def _qiskit_lines(spec: CircuitSpec) -> list:
    out = []
    for o in spec.ops:
        if o.kind == "barrier":
            out.append("    qc.barrier(%s)" % ", ".join(map(str, o.qubits)))
        elif o.kind == "gate":
            args = [fmt_angle(p).replace("pi", "np.pi") for p in o.params] + [str(q) for q in o.qubits]
            out.append("    qc.%s(%s)" % (o.name, ", ".join(args)))
        elif o.kind == "noise":
            out.append("    # noise channel placed in the app: %s(%g) on qubit(s) %s (Aer only, see noisy_* below)"
                       % (o.name, o.params[0], ", ".join(map(str, o.qubits))))
    return out or ["    pass"]


HEADER = '''#!/usr/bin/env python3
"""QEM Explorer export — {name}
Generated {date} by QEM Explorer {version}.

What it does
    Rebuilds the circuit, the compilation, the noise injector and every QEM contender you had in the app,
    and prints each estimate next to the value the app showed.

Run in a terminal
    python {fname}                  # the executor you used in the app: the app's numbers, digit for digit
    python {fname} --aer            # Qiskit Aer (density_matrix); PEC then samples instances (Monte Carlo)
    python {fname} --numpy          # the exact NumPy executor
    python {fname} --ibm ibm_xxx    # IBM Quantum hardware (raw, ZNE, symmetry, tensored/M3 readout)

Run in Jupyter / IPython
    Paste the whole file into one cell and run it: the last lines detect IPython and call main([]) without
    sys.exit, so the kernel keeps running. Afterwards call it again with options in a list, e.g.
        main(["--aer"])          main(["--numpy"])          main(["--ibm", "ibm_xxx"])
    or import pieces (build_state(), user_circuit()) for your own analysis.

Requirements
    pip install "qiskit>=1.0" "qiskit-aer>=0.13" numpy scipy scikit-learn psutil
    (+ qiskit-ibm-runtime for --ibm and for fake-device backends)

Campaign ledger (only for --ibm)
    Real hardware is used only through QEM Explorer's campaign QPU ledger: the qemx package must be importable
    and QEMX_CAMPAIGN_LEDGER must name a ledger (python -m qemx.campaign init). The job is admitted on its high
    usage estimate, tagged, recorded and reconciled like the app's jobs; without the ledger nothing is submitted.

Credentials (only for --ibm)
    A token is a password. Before this notebook goes into a repository or an appendix, clear it and use one of
    the other two. This script never contains a token: it reads IBM_QUANTUM_TOKEN (or QISKIT_IBM_TOKEN) from
    the environment, or falls back to an account saved with QiskitRuntimeService.save_account(), and it only
    reports which of the two it used.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
'''

MAIN = r'''

# =====================================================================================================
# 1. The state you built in QEM Explorer
# =====================================================================================================
STATE = json.loads(r"""{state}""")
APP_VALUES = json.loads(r"""{expected}""")       # the app's estimate of each contender, in STATE order (null: none)


# =====================================================================================================
# 2. Your circuit in plain Qiskit (logical circuit, before input preparation and basis change)
# =====================================================================================================
def user_circuit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit({n})
{qlines}
    return qc


# =====================================================================================================
# 3. Rebuild everything and run it
# =====================================================================================================
def build_state():
    spec = CircuitSpec.from_json(STATE["circuit"])
    cm = dict(STATE["compile"])
    init = str(cm.pop("initial_layout", "") or "")
    dev = STATE.get("backend") if is_device(STATE.get("backend")) else None
    lattice = cm.get("mode") == "isa" and cm.get("coupling", "all") != "all"
    layout = tuple(int(x) for x in init.replace(",", " ").split()) if (init and (dev or lattice)) else None
    if layout is not None and layout_problem(layout, spec.n, cm.get("coupling", "all"), dev):
        layout = None                  # the app ignores a layout that does not fit the circuit; so does the script
    compiled = build(spec, STATE["inputs"], STATE["basis"], CompileOptions(**cm, device=dev, initial_layout=layout),
                     relabel=STATE["relabel"])
    noise = NoiseSpec(**STATE["noise"])
    ob = STATE["observable"]
    if "pauli" in ob:                  # a Hamiltonian: measured setting by setting (qubit-wise commuting groups)
        obs = parse_pauli_sum(ob["pauli"], spec.n)
    else:
        obs = Observable(ob["key"], ob["label"], ob["tex"], tuple(ob["values"]))
    sym = STATE.get("symmetry")
    symmetry = (tuple(sym[:-1]), sym[-1]) if sym else None
    recipes = [Recipe.from_json(d) for d in STATE["recipes"]]
    return compiled, noise, obs, symmetry, recipes


def main(argv=None):
    """Terminal: python this_file.py [--aer | --numpy | --ibm NAME]. Notebook: main(["--aer"])."""
    for _s in (sys.stdout, sys.stderr):              # printed text is UTF-8 on every platform (labels use ⟨⟩, λ, γ)
        if hasattr(_s, "reconfigure") and (getattr(_s, "encoding", "") or "").lower().replace("_", "-") != "utf-8":
            try:
                _s.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
    if argv is None:
        argv = [] if _in_ipython() else sys.argv[1:]      # a kernel's own argv (-f kernel.json) is not ours
    argv = [str(a) for a in argv]
    if "--ibm" in argv:
        i = argv.index("--ibm")
        return run_on_ibm(argv[i + 1] if i + 1 < len(argv) else None)
    backend = "numpy" if "--numpy" in argv else "aer" if "--aer" in argv else STATE.get("backend", "numpy")
    if is_device(STATE.get("backend")) and ("--numpy" in argv or "--aer" in argv):
        STATE["backend"] = backend               # simulate the injector's model instead of the device model
    compiled, noise, obs, symmetry, recipes = build_state()
    print("circuit: %d qubits, %d two-qubit gates after compilation (%s)" % (
        compiled.implemented.n, compiled.implemented.count("2q"), STATE["compile"]["mode"]))
    runner = Runner(backend, layout=compiled.info.get("phys") if is_device(backend) else None)
    if runner.device_model is not None:
        noise = runner.device_model.equivalent_noise()
        print("device model: %s on physical qubits %s" % (device_label(backend), compiled.info.get("phys")))
    shots = int(STATE["shots"])
    ideal = None
    print("%-44s %10s %9s %9s %8s %10s" % ("method", "estimate", "bias", "sd", "gamma", "app"))
    seen = {{}}
    for i, rc in enumerate(recipes):
        if ideal is None:
            ideal = ideal_of(compiled, obs, runner)
            print("%-44s %10.6f" % ("ideal", ideal))
        r = run_recipe(compiled, noise, obs, rc, runner, shots, symmetry)
        t = rc.title()
        seen[t] = seen.get(t, 0) + 1
        label = t if seen[t] == 1 else "%s (#%d)" % (t, seen[t])     # contenders are matched by position, not name
        app = APP_VALUES[i] if isinstance(APP_VALUES, list) and i < len(APP_VALUES) else None
        mark = "" if app is None else ("  ✓" if abs(app - r.mean) < 1e-6 else "  (app %.6f)" % app)
        acc = account(r, shots, randomized_target=rc.twirl != "none")      # equal total budget, as in the app
        print("%-44s %10.6f %+9.5f %9.5f %8.3f %10s%s" % (label[:44], r.mean, r.mean - ideal,
                                                          acc["sd"], r.gamma,
                                                          "" if app is None else "%.6f" % app, mark))
        for note in r.notes:
            print("    note:", note)
    return 0


def _in_ipython() -> bool:
    """True inside Jupyter / IPython (where sys.exit would interrupt the cell with SystemExit)."""
    try:
        from IPython import get_ipython
    except Exception:
        return False
    try:
        return get_ipython() is not None
    except Exception:
        return False


# =====================================================================================================
# 4. The same circuits on IBM Quantum hardware
# =====================================================================================================
def _to_measured_qiskit(spec, measure_map):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(spec.n, spec.n)
    for o in spec.ops:
        if o.kind == "gate":
            getattr(qc, o.name)(*o.params, *o.qubits)
        elif o.kind == "barrier":
            qc.barrier(*(o.qubits or range(spec.n)))
    for c, q in measure_map.items():
        qc.measure(q, c)
    return qc


def _campaign_ledger():
    """QEM Explorer's campaign QPU ledger (QEMX_CAMPAIGN_LEDGER): real hardware is never used without it. Returns
    (qemx.campaign, ledger), or (None, None) after saying why nothing can be submitted."""
    try:
        from qemx import campaign as CG
    except ImportError:
        print("not submitted: --ibm admits the job against QEM Explorer's campaign QPU ledger; run this script where "
              "the qemx package is importable and set QEMX_CAMPAIGN_LEDGER (python -m qemx.campaign init ...)")
        return None, None
    try:
        ledger = CG.ledger_from_env()
        CG.check_no_unresolved(ledger)
    except (CG.LedgerError, CG.LedgerBusy, CG.CampaignLimit) as e:
        print("not submitted:", e)
        return None, None
    return CG, ledger


def run_on_ibm(backend_name=None):
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    CG, ledger = _campaign_ledger()                       # before any login: no ledger, no submission
    if ledger is None:
        return 2
    token = os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        print("credentials: environment variable")
    else:
        service = QiskitRuntimeService()
        print("credentials: saved account")
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    compiled, noise, obs, symmetry, recipes = build_state()
    if isinstance(obs, PauliSum):
        print("This script's --ibm path measures diagonal observables. For a Hamiltonian (several measurement "
              "settings) use the app's 'Run on IBM Quantum' page, which runs every setting.")
        return 1
    shots = int(STATE["shots"])
    n = compiled.implemented.n
    plan, circuits = [], []

    def add(spec, mm):
        circuits.append(_to_measured_qiskit(spec, mm))
        return len(circuits) - 1

    ident = {{i: i for i in range(n)}}
    cal = (add(CircuitSpec(n, (), "cal0"), ident), add(CircuitSpec(n, tuple(Op("x", (q,), ()) for q in range(n))), ident))
    for rc in recipes:
        p = rc.p
        if rc.estimator in ("raw", "symmetry") and rc.readout in ("none", "tensored", "m3") and rc.twirl == "none":
            plan.append((rc, [add(compiled.implemented, compiled.measure_map)]))
        elif rc.estimator == "zne" and rc.readout in ("none", "tensored", "m3") and rc.twirl == "none":
            idx = []                             # per scale: (circuit indices, achieved λ)
            for lam in sorted(float(x) for x in p["scales"]):
                if p["folding"] == "global":
                    folds = [fold_global(compiled.implemented, lam)]
                else:                            # the method's fold subset (balanced: stratified without a model)
                    K = int(p.get("fold_draws", 1) or 1) if is_fractional(lam) else 1
                    folds = [fold_local(compiled.implemented, lam, draw_seed(0, k), p.get("subset", "random"))
                             for k in range(K)]
                idx.append(([add(f, compiled.measure_map) for f in folds],
                            achieved_lambda(compiled.implemented, folds[0])))
            plan.append((rc, idx))
        else:
            print("skipped on hardware in this version:", rc.title())
    # optimization_level=0 keeps folded U†U pairs (a higher level would cancel them)
    isa = transpile(circuits, backend=backend, optimization_level=0)
    est = CG.estimate_usage(shots * len(isa), len(isa), 1)
    met = CG.default_max_execution_time(est)
    sampler = SamplerV2(mode=backend)                     # job mode, the campaign default

    def prepare(tag):
        sampler.options.max_execution_time = met
        return CG.tag_sampler(sampler, tag)
    try:
        job, job_id, _, _ = CG.submit_one(ledger, "exported script: %d circuits x %d shots on %s" % (
            len(isa), shots, backend.name), est, lambda tag: sampler.run(isa, shots=shots), "exported-script",
            prepare=prepare, max_execution_time=met)
    except (CG.CampaignLimit, CG.SubmissionRejected, CG.SubmissionUnknown) as e:
        print("not submitted" if not isinstance(e, CG.SubmissionUnknown) else "outcome unknown", "-", e)
        return 2
    print("job id:", job_id)
    result = job.result()
    print("campaign ledger:", CG.reconcile_job(ledger, job_id, job))

    def probs(i):
        counts = result[i].data.c.get_counts()
        p = np.zeros(2 ** n)
        for k, v in counts.items():
            p[int(k, 2)] += v
        return p / p.sum()

    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
    p0, p1 = probs(cal[0]), probs(cal[1])
    A = [np.array([[1 - p0 @ bits[:, i], p1 @ (1 - bits[:, i])], [p0 @ bits[:, i], 1 - p1 @ (1 - bits[:, i])]])
         for i in range(n)]

    def mitigate(p, mode):
        if mode == "tensored":
            t = p.reshape([2] * n)
            for i in range(n):
                ax = n - 1 - i
                t = np.moveaxis(np.tensordot(np.linalg.inv(A[i]), t, axes=([1], [ax])), 0, ax)
            return t.reshape(-1)
        if mode == "m3":
            return m3_solve(p, A, n)
        return p

    o = obs.o
    for rc, idx in plan:
        if rc.estimator == "zne":                # E(λ) of every scale (the mean over its fold draws), then the fit
            lams = [lam for _, lam in idx]
            E = [float(np.mean([o @ mitigate(probs(i), rc.readout) for i in ii])) for ii, _ in idx]
            est = extrapolate_policy(rc.p["rule"], lams, E, rc.p.get("on_fail", "linear"))[0]   # with its failure policy
        else:
            ps = [mitigate(probs(i), rc.readout) for i in idx]
            if rc.estimator == "raw":
                est = o @ ps[0]
            else:
                mask = parity_mask(n, rc.p["qubits"], int(rc.p["eigenvalue"]))
                est = (ps[0] * mask) @ o / (ps[0] * mask).sum()
        print("%-44s %10.6f" % (rc.title()[:44], est))
    return 0
'''


def recipe_labels(recipes: list) -> list:
    """The labels the app gives contenders (title, then "title (#k)" for a repeated title), in order."""
    seen, out = {}, []
    for rc in recipes:
        t = rc.title()
        seen[t] = seen.get(t, 0) + 1
        out.append(t if seen[t] == 1 else "%s (#%d)" % (t, seen[t]))
    return out


def app_values(state: dict, expected) -> list:
    """The app's estimates as a list aligned with state["recipes"] (None where there is none). ``expected`` is that
    list already, or {label: value} keyed by the app's labels (duplicate titles carry "(#k)"), so two contenders
    with the same title are never confused."""
    if not expected:
        return []
    n = len(state.get("recipes") or [])
    if isinstance(expected, (list, tuple)):
        vals = list(expected)[:n]
    else:
        from .methods.base import Recipe
        labels = recipe_labels([Recipe.from_json(d) for d in state.get("recipes") or []])
        vals = [expected.get(lab) for lab in labels]
    return [float(v) if v is not None and v == v else None for v in vals]


_DOC_UNSAFE = re.compile(r'[\x00-\x1f\x7f"\\]')


def _doc_text(text, limit: int = 120) -> str:
    """Text placed inside the generated script's docstring: no quotes, backslashes or control characters."""
    return _DOC_UNSAFE.sub(" ", str(text))[:limit]


def script(state: dict, expected=None) -> str:
    """The standalone script for ``state``. ``expected``: the app's estimates, a list aligned with
    state["recipes"] or {label: value} (see app_values)."""
    spec = CircuitSpec.from_json(state["circuit"])
    fname = "qem_explorer_export.py"
    from . import __version__
    head = HEADER.format(name=_doc_text(spec.name), date=_dt.date.today().isoformat(), version=__version__,
                         fname=fname)
    main = MAIN.format(state=json.dumps(state, indent=1), expected=json.dumps(app_values(state, expected), indent=1),
                       n=spec.n, qlines="\n".join(_qiskit_lines(spec)))
    tail = '''

# =====================================================================================================
# Appendix: the QEM Explorer engine, copied verbatim (so this script reproduces the app exactly)
# =====================================================================================================
''' + engine_source() + '''


# =====================================================================================================
# Entry point: works as a terminal script and as a pasted Jupyter cell
# =====================================================================================================
if __name__ == "__main__":
    if _in_ipython():
        main([])                 # notebook: no sys.exit; call main(["--aer"]) etc. in the next cell for options
    else:
        sys.exit(main())
# To run it by hand (for example after importing this file as a module):
# main()            # or main(["--aer"]), main(["--numpy"]), main(["--ibm", "ibm_xxx"])
'''
    return head + main + tail
