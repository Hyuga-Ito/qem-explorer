"""1.5: two users at once. Concurrent comparisons on shared code paths give exactly the serial results, and two app
sessions with different settings never see each other's results through the shared caches."""
import os
import threading

import numpy as np
import pytest

from qemx import compile as CP
from qemx import templates as T
from qemx.noise import NoiseSpec
from qemx.recipes import compare, preset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPES = ("Unmitigated", "ZNE · Richardson", "Readout · TREX", "CDR · OLS")


def _run(noise, out, key):
    tpl = T.build("qft", {"n": 3})
    comp = CP.build(tpl.spec, relabel=tpl.relabel)
    ideal, outs, _ = compare(comp, noise, tpl.observable, [preset(r) for r in RECIPES], shots=50_000, seed=3)
    out[key] = [(o.title, None if o.result is None else (o.result.mean, o.result.var1)) for o in outs]


def test_concurrent_comparisons_equal_serial_ones():
    noises = {"a": NoiseSpec(p1=0.001, p2=0.01, ro01=0.02, ro10=0.02),
              "b": NoiseSpec(p1=0.004, p2=0.03, ro01=0.05, ro10=0.01)}
    serial = {}
    for k, nz in noises.items():
        _run(nz, serial, k)
    par = {}
    th = [threading.Thread(target=_run, args=(nz, par, k)) for k, nz in noises.items() for _ in range(1)]
    for t in th:
        t.start()
    for t in th:
        t.join(600)
    assert set(par) == set(serial)
    for k in serial:
        for (t1, r1), (t2, r2) in zip(serial[k], par[k]):
            assert t1 == t2
            assert (r1 is None) == (r2 is None)
            if r1 is not None:
                assert np.allclose(r1, r2, rtol=0, atol=0), (k, t1)        # bitwise: no shared mutable state
    assert serial["a"] != serial["b"]


@pytest.mark.slow
def test_two_sessions_keep_their_own_results():
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except Exception:                                    # pragma: no cover
        pass
    from streamlit.testing.v1 import AppTest

    def session(p2):
        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900).run()
        at.switch_page("pages/1_Playground.py").run()
        at.session_state["noise"] = at.session_state["noise"].replace(p2=p2)
        at.run()
        b = [x for x in at.button if x.key and x.key.startswith("sb_run_")][0]
        b.click().run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    A = session(0.01)
    ra = [(o.title, o.result.mean) for o in A.session_state["results"]["outs"] if o.result is not None]
    B = session(0.05)
    rb = [(o.title, o.result.mean) for o in B.session_state["results"]["outs"] if o.result is not None]
    A.run()                                              # A reruns after B filled the shared caches
    ra2 = [(o.title, o.result.mean) for o in A.session_state["results"]["outs"] if o.result is not None]
    assert ra == ra2 and ra != rb
    assert A.session_state["results"]["sig"] != B.session_state["results"]["sig"]
