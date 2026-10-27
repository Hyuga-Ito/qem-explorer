"""1.5: cited parameter presets. Every preset builds, states its provenance, and the quantities the
'exact reproduction' presets claim are reproduced by the app's own simulator."""
import math
import re

import pytest

from qemx import compile as CP
from qemx import io_semantics as IO
from qemx import presets_cited as PC
from qemx import templates as T


def _ideal(template, params):
    tpl = T.build(template, T.clamp_params(template, params))
    comp = CP.build(tpl.spec, relabel=tpl.relabel)
    return IO.derived_output(comp.logical, tpl.relabel, tpl.observable)


@pytest.mark.parametrize("p", PC.PRESETS, ids=lambda p: p.key)
def test_every_preset_is_complete_and_builds(p):
    assert p.kind in PC.KINDS
    assert p.template in T.REGISTRY
    names = {tp.name for tp in T.REGISTRY[p.template][2]}
    assert set(p.params) <= names, set(p.params) - names
    r = PC.resolve(p.key)
    assert T.clamp_params(p.template, p.params) == r["params"]           # nothing is silently clamped
    T.build(p.template, r["params"])
    for f in ("purpose", "instance", "convention", "units", "sizes", "limitations"):
        assert getattr(p, f).strip(), f
    if p.kind != "educational":
        assert p.sources and all(s.arxiv and loc for s, loc in p.sources)
    if p.kind == "adapted":
        assert p.differences
    text = PC.describe(p.key)
    assert not re.search(r"recommended", text, re.I)
    assert "Resolved values" in text and p.checked in text


@pytest.mark.parametrize("n", [4, 6, 8])
def test_qaoa_ring_p1_reproduces_three_quarters(n):
    params = dict(PC.BY_KEY["qaoa-ring-p1"].params, n=n)
    assert _ideal("qaoa", params)["observable_value"] == pytest.approx(0.75, abs=1e-9)


def test_qaoa_3_regular_p1_is_above_the_worst_case_guarantee():
    v = _ideal("qaoa", PC.BY_KEY["qaoa-3reg-p1"].params)["observable_value"]
    assert v >= 0.6924


def test_qft_known_answer_and_aqft_truncation():
    p = PC.BY_KEY["qft-known-answer"].params
    d = _ideal("qft", p)
    assert d["deterministic"] and int(d["top"][0][0], 2) == p["s"]
    a = PC.BY_KEY["aqft-coppersmith"]
    da = _ideal("aqft", a.params)
    assert 0.5 < da["observable_value"] < 1.0                           # truncation costs success probability
    assert PC.resolve(a.key)["derived"]["Coppersmith m_C"] == a.params["m"] + 1


def test_trotter_bound_is_an_upper_bound_on_the_actual_error():
    import numpy as np
    from qiskit.quantum_info import Operator, SparsePauliOp
    p = PC.BY_KEY["ising-trotter-1st"]
    r = PC.resolve(p.key)
    n, h, dt, steps = (r["params"][k] for k in ("n", "h", "dt", "steps"))
    tpl = T.build("ising", r["params"])
    from qemx.circuit import to_qiskit
    U = Operator(to_qiskit(tpl.spec)).data
    terms = [("I" * (n - i - 2) + "ZZ" + "I" * i, -1.0) for i in range(n - 1)] + \
            [("I" * (n - i - 1) + "X" + "I" * i, -h) for i in range(n)]
    H = SparsePauliOp.from_list(terms).to_matrix()
    w, V = np.linalg.eigh(H)
    exact = V @ np.diag(np.exp(-1j * w * dt * steps)) @ V.conj().T
    # the circuit may differ from the exact evolution by a global phase: minimize over it
    ph = np.trace(exact.conj().T @ U)
    ph = ph / abs(ph) if abs(ph) > 0 else 1.0
    err = np.linalg.norm(U - ph * exact, 2)
    assert 0 < err <= r["derived"]["total bound (r steps) ≤"]


def test_search_finds_presets_by_words():
    assert [p.key for p in PC.search("trotter")] == ["ising-trotter-1st"]
    assert {p.key for p in PC.search("qaoa ring")} == {"qaoa-ring-p1"}
    assert {p.key for p in PC.search("", template="qft")} == {"qft-known-answer", "qft-basis-input"}
    assert PC.search("coppersmith")[0].key == "aqft-coppersmith"
