"""Streamlit caches around the engine. Keys are content hashes, JSON strings or signatures, so a cache hit needs the
exact same circuit, compilation, noise, backend and recipes. These caches are shared by every session of the
server, so a key must always describe what the cached value was computed from."""
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


@st.cache_resource(max_entries=12, show_spinner=False)
def comparison(sig: str, _compiled, noise_json: str, _observable: Observable, _recipes: tuple, backend: str,
               shots: int, _symmetry):
    return compare(_compiled, NoiseSpec(**json.loads(noise_json)), _observable, list(_recipes), backend, shots,
                   _symmetry, check_budget=True)       # budget preflight: unaffordable protocols are not run


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
