"""Streamlit caches around the engine. Keys are content hashes, JSON strings or signatures, so a cache hit needs the
exact same circuit, compilation, noise, backend and recipes. The resource caches below are shared by every session of
the server, so a key must always describe what the cached value was computed from, and only values that are not
mutated after they are built are shared; comparisons (mutable runners and outcomes) are cached per session."""
from __future__ import annotations

import dataclasses
import json

import streamlit as st

from qemx import compile as CP
from qemx import timeline as TL
from qemx.circuit import CircuitSpec
from qemx.noise import NoiseSpec
from qemx.observables import Observable
from qemx.recipes import compare


def _digest(spec: CircuitSpec) -> str:
    from ui.state import spec_digest
    return spec_digest(spec)


@st.cache_resource(max_entries=64, show_spinner=False)
def compiled(spec_digest: str, _spec: CircuitSpec, inputs: tuple, basis: str, opts_json: str, relabel: bool):
    d = json.loads(opts_json)
    if d.get("initial_layout") is not None:
        d["initial_layout"] = tuple(d["initial_layout"])
    return CP.build(_spec, inputs, basis, CP.CompileOptions(**d), relabel=relabel)


def get_compiled(spec: CircuitSpec, inputs, basis, opts: CP.CompileOptions, relabel: bool):
    """The compiled circuit, cached by the circuit's content hash (computed once per CircuitSpec object)."""
    return compiled(_digest(spec), spec, tuple(inputs), basis, json.dumps(dataclasses.asdict(opts), sort_keys=True),
                    bool(relabel))


def circuit_key(comp) -> tuple:
    """A cheap fingerprint of a compiled circuit for cache keys that also carry a full state signature."""
    impl = comp.implemented
    return (impl.n, len(impl.ops), impl.count("2q"), tuple(sorted(comp.measure_map.items())))


@st.cache_resource(max_entries=24, show_spinner=False)
def _timeline(sig: str, circuit: tuple, _compiled, noise_json: str, kind: str, lam: float, folding: str, dd: str,
              backend: str):
    return TL.build(_compiled, NoiseSpec(**json.loads(noise_json)), kind, lam, folding, dd, backend)


def timeline(sig: str, compiled_, noise_json: str, kind: str, lam: float, folding: str, dd: str, backend: str):
    """Time-travel trace. ``sig`` must be the signature of the state ``compiled_`` was built from (the cache is
    shared by every session); the circuit's fingerprint is part of the key as a second guard."""
    return _timeline(sig, circuit_key(compiled_), compiled_, noise_json, kind, float(lam), folding, dd, backend)


SESSION_COMPARISONS = 4          # comparisons kept per session (Playground, Bias–variance A and B, one spare)


def comparison(sig: str, _compiled, noise_json: str, _observable: Observable, _recipes: tuple, backend: str,
               shots: int, _symmetry):
    """The comparison (ideal, outcomes, runner) for this state, cached **per session** (1.5.1, review 6.3).

    A comparison holds mutable objects — the runner's memo of simulations, outcomes and their resamplers — that
    later views keep using. 1.5.0 kept them in a server-wide ``st.cache_resource``, shared by every session; no
    corruption was reproduced, but nothing guaranteed that two sessions never mutated the same objects at once. Each
    session now owns its comparisons (a small LRU in its own state); compiled circuits and time-travel traces, which
    are not mutated after they are built, stay shared."""
    key = (sig, noise_json, tuple(r.key() if hasattr(r, "key") else repr(r) for r in _recipes), backend, int(shots))
    store = st.session_state.get("_qemx_comparisons")
    if store is None:
        from collections import OrderedDict
        store = OrderedDict()
        st.session_state["_qemx_comparisons"] = store
    if key in store:
        store.move_to_end(key)
        return store[key]
    val = compare(_compiled, NoiseSpec(**json.loads(noise_json)), _observable, list(_recipes), backend, shots,
                  _symmetry, check_budget=True)        # budget preflight: unaffordable protocols are not run
    store[key] = val
    while len(store) > SESSION_COMPARISONS:
        store.popitem(last=False)
    return val


def noise_json(noise: NoiseSpec) -> str:
    return json.dumps(dataclasses.asdict(noise), sort_keys=True)


def _opts(opts_json: str) -> CP.CompileOptions:
    d = json.loads(opts_json)
    if d.get("initial_layout") is not None:
        d["initial_layout"] = tuple(d["initial_layout"])
    return CP.CompileOptions(**d)


@st.cache_resource(max_entries=16, show_spinner=False)
def stages(spec_digest: str, _spec: CircuitSpec, inputs: tuple, basis: str, opts_json: str, relabel: bool):
    from qemx import stages as STG
    return STG.compile_stages(_spec, inputs, basis, _opts(opts_json), relabel)


def get_stages(spec: CircuitSpec, inputs, basis, opts: CP.CompileOptions, relabel: bool):
    """Logical → routed → native stages for the current compile options (cached)."""
    return stages(_digest(spec), spec, tuple(inputs), basis, json.dumps(dataclasses.asdict(opts), sort_keys=True),
                  bool(relabel))


@st.cache_resource(max_entries=16, show_spinner=False)
def layout_rank(spec_digest: str, inputs: tuple, basis: str, opts_json: str, relabel: bool, device_name: str,
                _compiled, top: int = 12):
    from qemx import layout_search as LS
    return LS.rank_layouts(_compiled, device_name, top=top)
