"""1.5: feature dictionary, space-filling design and confirmatory split, and the baseline recommender's evaluation
(grouped folds without leakage, abstention, baselines) on synthetic data with a known answer."""
import math

import numpy as np
import pytest

from qemx import features as FT
from qemx import recommend as RC


def test_feature_dictionary_is_complete_and_oracles_are_refused():
    names = [f.name for f in FT.DICTIONARY]
    assert len(names) == len(set(names))
    assert {f.level for f in FT.DICTIONARY} <= set(FT.LEVELS)
    for fs, cols in FT.SETS.items():
        for c in cols:
            assert FT.BY_NAME[c].recommendable, (fs, c)
    with pytest.raises(ValueError, match="oracle"):
        FT.vector({"ideal": 1.0}, ["ideal"])
    assert not FT.BY_NAME["entropy_half"].scalable and FT.BY_NAME["two_qubit_logical"].scalable
    assert "NOT an entanglement measure" in FT.BY_NAME["output_entropy_bits"].definition
    md = FT.dictionary_markdown()
    main, aliases = md.split("## Names of earlier schemas")            # 1.5.1: the /1 → /2 alias table follows
    assert main.count("\n| `") == len(FT.DICTIONARY) and aliases.count("\n| `") == len(FT.ALIASES)


def test_features_from_a_record_keep_missing_values_as_nan_with_reasons():
    rec = dict(cell=dict(n=4, scale=1.0, coherent_fraction=0.0, family="ghz"),
               circuit=dict(two_qubit=6, depth=10, logical_two_qubit=3, routing_overhead=3),
               descriptors=dict(non_clifford=1, obs_weight=4, obs_terms=1, idle_fraction=0.2),
               **{"lambda": dict(lam_lc=0.05, lam_lc_gate=0.04, lam_lc_idle=0.01, lam_ro=0.02, lc_2q_gates=5)})
    v, miss = FT.from_record(rec, 1e4)
    # 1.5.1 (schema /2): two_qubit_density is two_qubit_per_qubit_layer (a rate, same formula)
    assert v["two_qubit_per_qubit_layer"] == pytest.approx(6 / 40) and v["lam_per_2q"] == pytest.approx(0.01)
    assert v["two_qubit_compile_overhead"] == 3 and "two_qubit_density" not in v
    assert v["log10_budget"] == pytest.approx(4)
    assert math.isnan(v["entropy_half"]) and "entropy_half" in miss            # never an invented zero
    assert math.isnan(v["entropy_profile_max"])


def test_entanglement_profile_sees_what_the_final_state_hides():
    from qemx.circuit import parse
    spec = parse("h 0; cx 0 1; cx 0 1; h 0", 2, "undo")     # Bell pair in the middle, |00⟩ at the end
    prof = FT.entanglement_profile(spec)
    assert max(prof) == pytest.approx(1.0) and prof[-1] == pytest.approx(0.0, abs=1e-12)


def test_space_filling_design_and_confirmatory_split():
    from qemx.phase.manifest import confirmatory_split, expand
    m = dict(manifest_version=1, study="t", methods=dict(RAW=dict(estimator="raw")), method_sets=dict(c=["RAW"]),
             circuits=dict(c=[dict(family="ising", n=[3, 4], steps=[1, 2, 3, 4, 5, 6], T=1.0, J=1.0, h=1.0)]),
             stages=[dict(name="S", engine="E1", circuits="c", methods="c", scales=[0.5, 1.0, 2.0], readout="on",
                          design=dict(kind="maximin", per_stratum=5, seed=3),
                          confirmatory=dict(fraction=0.3, seed=1))])
    cells = expand(m)
    assert len(cells) == 10                                          # 5 per (family, n) stratum of 18
    assert [c.cell_id for c in expand(m)] == [c.cell_id for c in cells]          # deterministic
    for n in (3, 4):
        sc = {c.scale for c in cells if c.n == n}
        st = {dict(c.fparams)["steps"] for c in cells if c.n == n}
        assert sc == {0.5, 1.0, 2.0} and {1, 6} <= st and len(st) >= 3   # spans the stratum (maximin: corners first)
    sp = confirmatory_split(m)
    by_circuit = {}
    for c in cells:
        by_circuit.setdefault((c.n, c.fparams), set()).add(sp[c.cell_id])
    assert all(len(v) == 1 for v in by_circuit.values())            # a base circuit never straddles the split
    bad = dict(m, stages=[dict(m["stages"][0], design=dict(kind="lhs"))])
    with pytest.raises(ValueError):
        from qemx.phase.manifest import check
        check(bad)


def _synthetic(seed=0, groups=40, per=6):
    """Method A is best when x0 > 0, B when x0 < 0; RAW is never best by more than noise when |x0| is tiny."""
    rng = np.random.default_rng(seed)
    X, Y, G, fam = [], [], [], []
    for g in range(groups):
        base = rng.normal(size=2)
        for _ in range(per):
            x = base + 0.05 * rng.normal(size=2)
            yA = -1.0 - 0.8 * x[0] + 0.05 * rng.normal()
            yB = -1.0 + 0.8 * x[0] + 0.05 * rng.normal()
            yR = -0.6 + 0.05 * rng.normal()
            X.append(x)
            Y.append([yR, yA, yB])
            G.append("g%d" % g)
            fam.append("f%d" % (g % 4))
    return np.array(X), np.array(Y), np.ones((len(X), 3), bool), np.array(G), np.array(fam)


def test_recommender_learns_a_known_rule_on_grouped_folds():
    X, Y, F, G, fam = _synthetic()
    methods = ["RAW", "A", "B"]
    folds = RC.group_folds(G, k=5, seed=0)
    RC.check_no_leakage(folds, G)
    ev = RC.evaluate(X, Y, F, methods, folds, alpha=0.1)
    s = ev["summary"]
    assert s["coverage"] > 0.6 and s["accuracy"] > 0.9
    assert s["regret_median"] < 0.05
    assert s["baseline_raw_regret_median"] > 0.5                  # always-raw is much worse here
    lofo = RC.evaluate(X, Y, F, methods, RC.leave_one_out(fam, "family"), alpha=0.1)
    assert lofo["summary"]["accuracy"] > 0.8


def test_row_splits_leak_and_are_refused():
    X, Y, F, G, fam = _synthetic(groups=10)
    idx = np.arange(len(G))
    bad = [RC.Fold(idx[idx % 2 == 0], idx[idx % 2 == 1], "row split")]
    with pytest.raises(AssertionError, match="both sides"):
        RC.check_no_leakage(bad, G)


def test_abstention_raw_and_infeasible():
    X, Y, F, G, fam = _synthetic()
    rec = RC.RidgeRecommender(["RAW", "A", "B"], alpha=0.1).fit(X, Y, F)
    far = rec.recommend(np.array([50.0, 0.0]), [True, True, True])
    assert far.kind == "abstain-coverage" and far.method is None
    assert rec.recommend(np.array([0.0, 0.0]), [True, False, False]).method == "RAW"
    assert rec.recommend(np.array([1.0, 0.0]), [False, False, False]).kind == "no-feasible"
    assert rec.recommend(np.array([np.nan, 0.0]), [True, True, True]).kind == "abstain-coverage"
    # a world where no method beats raw: the honest answer is raw
    Y2 = Y.copy()
    Y2[:, 1:] = Y2[:, :1] + 0.01
    rec2 = RC.RidgeRecommender(["RAW", "A", "B"], alpha=0.1).fit(X, Y2, F)
    r = rec2.recommend(X[0], [True, True, True])
    assert r.kind == "raw" and r.method == "RAW"


def test_rule_baseline():
    assert RC.rule_based(0.001, 1e5, {"PEC": True}) == "RAW"
    assert RC.rule_based(0.5, 1e7, {"PEC": True, "ZNE-R": True}) == "PEC"
    assert RC.rule_based(3.0, 1e5, {"PEC": True, "ZNE-R": True}) == "ZNE-R"
