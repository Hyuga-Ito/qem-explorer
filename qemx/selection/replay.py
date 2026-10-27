r"""Full-policy evaluation of the deployment API on labeled tasks (1.8.0; review finding F1).

A *model-component* benchmark scores the candidate a fitted model ranks first among the candidates the labels say are
feasible. The *application policy* is more: the bounded-cost guard, feature acquisition, out-of-distribution checks,
the preflight contract (structural, then the learning designs), the margin rule, the fallback, and the candidate it
finally returns. ``replay`` calls ``deploy.recommend`` itself on every (task, budget) of a labels directory and scores
the returned candidate with the labels' *evaluation* half; nothing is approximated by benchmark-only feasibility.

Every row is kept. A returned candidate without feasible labels at that budget is ``unscored`` (counted, reported, its
risk not imputed); the summary's primary difference is computed only when no row is unscored, otherwise it is reported
as undefined with the count. Costs: the policy uses no shots before execution (all preflight levels are classical);
its classical seconds are reported per row.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .. import textio as TX
from .models import EPS0


class StaleTable(RuntimeError):
    """A cached labels table that does not match the validated records it claims to summarize."""


LABEL_COLUMNS = ("feasible", "mse_eval", "mse_sel")


def derived_labels(labels_dir: str, task_ids):
    """The labels table of ``task_ids`` derived from their (validated) records and arrays (``labels.summarize``).
    A cached ``labels.csv`` in ``labels_dir`` is never trusted by itself (1.9.0; second review N4): when it exists, its
    rows for these tasks must equal the derived ones (same keys; feasibility; MSE to 1e-12 relative), else
    ``StaleTable``. Duplicate (task, protocol, budget) rows are refused."""
    import pandas as pd
    from . import labels as LB
    tab = LB.summarize(labels_dir, task_ids=task_ids)
    key = ["task_id", "protocol", "B"]
    if len(tab) and tab.duplicated(key).any():
        raise StaleTable("%s: duplicate (task, protocol, budget) rows in the derived labels" % labels_dir)
    p = os.path.join(labels_dir, "labels.csv")
    if os.path.exists(p):
        cached = pd.read_csv(p, encoding="utf-8")
        if cached.duplicated(key).any():
            raise StaleTable("%s: duplicate (task, protocol, budget) rows" % p)
        c = cached[cached.task_id.isin(set(task_ids))]
        m = tab[key + list(LABEL_COLUMNS)].merge(c[[k for k in key + list(LABEL_COLUMNS) if k in c.columns]],
                                                 on=key, how="outer", suffixes=("", "_cached"), indicator=True)
        bad = int((m["_merge"] != "both").sum())
        for col in LABEL_COLUMNS:
            if col + "_cached" not in m:
                continue
            a_, b_ = m[col], m[col + "_cached"]
            if col == "feasible":
                diff = a_.map(_feasible) != b_.map(_feasible)
            else:
                x, y = a_.astype(float).to_numpy(), b_.astype(float).to_numpy()
                both_nan = np.isnan(x) & np.isnan(y)
                diff = ~(both_nan | (np.abs(x - y) <= 1e-12 * np.maximum(np.abs(x), np.abs(y)) + 1e-300))
            bad += int(np.sum(diff))
        if bad:
            raise StaleTable("%s does not match the validated records of %s (%d differing or unmatched rows); it is "
                             "stale or was not derived from them — delete it to regenerate" % (p, labels_dir, bad))
    return tab


def _labels(labels_dir: str):
    """1.8.0 helper kept for scripts: the labels table of a labels directory, now always derived from its validated
    manifest records (a cached CSV is checked against them, never trusted alone)."""
    b = bound_population(labels_dir, require_complete=False)
    if b["table"] is None:
        raise ValueError("%s: no validated task records (%s); a labels table alone is never accepted"
                         % (labels_dir, b["completion"]))
    return b["table"]


def bound_population(labels_dir: str, *, require_complete: bool = True, subset=None, subset_reason: str = "") -> dict:
    """The population a replay evaluates (1.9.0; second review N4): the bound run's manifest (``run.json``), every
    record validated (``integrity.completion``: task identity, label configuration, run, array digests). By default
    the run must be complete (``IncompleteRun`` otherwise). ``subset`` (task ids) declares an explicit exploratory
    subset: it needs ``subset_reason``; the intended population, the excluded tasks and the reason are returned and
    must be reported (the estimand changes). Corrupted records always refuse."""
    from . import integrity as IN
    comp = IN.completion(labels_dir)
    if comp["corrupted"]:
        raise IN.CorruptedRun("%s: %d corrupted record(s): %s" % (labels_dir, len(comp["corrupted"]), "; ".join(
            "%s (%s)" % (c["task_id"], c["why"]) for c in comp["corrupted"][:5])))
    from . import binding as BD
    run = BD.load(labels_dir)
    manifest = list(run["manifest"]) if run else sorted(set(comp["completed_ids"]) | {t for v in comp["failed"].values()
                                                                                   for t in v} | set(comp["missing"]))
    failed = {t: st for st, v in comp["failed"].items() for t in v}
    if subset is not None:
        if not subset_reason:
            raise ValueError("an exploratory subset needs a stated reason (subset_reason)")
        unknown = sorted(set(subset) - set(manifest))
        if unknown:
            raise ValueError("subset tasks not in the manifest of %s: %s" % (labels_dir, ", ".join(unknown[:5])))
        intended = sorted(set(subset))
    else:
        intended = sorted(manifest)
    excluded = []
    for t in sorted(set(manifest) - set(intended)):
        excluded.append(dict(task_id=t, why="outside the declared subset: %s" % subset_reason))
    usable = []
    for t in intended:
        if t in failed:
            excluded.append(dict(task_id=t, why="no labels (status %s)" % failed[t]))
        elif t in comp["missing"]:
            excluded.append(dict(task_id=t, why="no record (never labeled)"))
        else:
            usable.append(t)
    if require_complete and subset is None and (len(usable) != len(manifest)):
        raise IN.IncompleteRun("%s: replay of the bound population needs every manifest task labeled; %s. Finish "
                               "labeling, or declare an exploratory subset with its reason." % (
                                   labels_dir, IN.describe(comp)))
    tab = derived_labels(labels_dir, usable) if usable else None
    return dict(labels_dir=labels_dir, run_identity=comp["run_identity"], manifest=len(manifest),
                intended=len(intended), used=usable, excluded=excluded, subset=subset is not None,
                subset_reason=subset_reason or None, data_digest=IN.data_digest(labels_dir, usable) if usable else None,
                completion=IN.describe(comp), table=tab)


def bundle_identity(bundle_path: str) -> dict:
    """Digests identifying a bundle: its bundle.json and arrays, its configuration and training freeze."""
    from . import labels as LB
    meta = TX.read_json(os.path.join(bundle_path, "bundle.json"))
    arr = [f for f in sorted(os.listdir(bundle_path)) if f.endswith(".npz")]
    return dict(name=os.path.basename(os.path.normpath(bundle_path)), version=meta.get("version"),
                configuration=meta.get("configuration"),
                bundle_json_sha256=LB.file_digest(os.path.join(bundle_path, "bundle.json")),
                arrays_sha256={f: LB.file_digest(os.path.join(bundle_path, f)) for f in arr},
                tau=(meta.get("selective") or {}).get("tau"),
                training_freeze=((meta.get("training") or {}).get("freeze") or {}),
                training_run_identities=(meta.get("training") or {}).get("run_identities"))


class UncheckedBaseline(ValueError):
    """The SBS-B baseline of an evaluation cannot be checked against its rule (no training labels and no bundle trained
    on the evaluation's training parts)."""


def _first_feasible_choice(ranks: dict, tab) -> dict:
    """SBS-B's choice per (task, budget): the first label-feasible candidate in the budget's training rank (the
    nearest trained budget on the log scale for an unseen one), RAW when none is feasible — ``SBSBudget.choose``."""
    feas = {}
    for r in tab.itertuples():
        feas[(r.task_id, r.protocol, int(r.B))] = _feasible(r.feasible) and math.isfinite(float(r.mse_eval))
    bs = sorted(ranks)
    out = {}
    for (t, _k, B) in feas:
        if (t, B) in out:
            continue
        rk = ranks.get(B) or (ranks[min(bs, key=lambda b: abs(math.log10(b) - math.log10(B)))] if bs else [])
        out[(t, B)] = next((p for p in rk if feas.get((t, p, B))), "RAW")
    return out


def sbsb_rule_choices(labels_dir: str, tab, bundle_metas=()) -> tuple:
    """SBS-B's choices on an evaluated population by the SBS-B rule, independently of any saved choices file (1.9.0;
    independent verification): ``(choices {(task_id, B): protocol}, source)``.

    The rule is refitted on the evaluation's own training labels (``EVALUATED.train_parts`` next to ``labels_dir``,
    complete and validated) when they are present; otherwise taken from the fallback ranks of a bundle trained on the
    same parts. Neither available → ``UncheckedBaseline``: a baseline column that cannot be checked is not used."""
    ev_p = os.path.join(labels_dir, "EVALUATED")
    ev = TX.read_json(ev_p) if os.path.exists(ev_p) else {}
    parts = [x for x in str(ev.get("train_parts") or "").split(",") if x]
    run_dir = os.path.dirname(os.path.abspath(labels_dir))
    if parts and all(os.path.isdir(os.path.join(run_dir, p, "tasks")) for p in parts):
        from . import binding as BD, dataset as DT, integrity as IN, policies as PO
        try:
            ds = None
            for p in parts:
                d, _ = BD.collect(os.path.join(run_dir, p), require_complete=True)
                ds = d if ds is None else DT.concat(ds, d)
        except (IN.IncompleteRun, IN.CorruptedRun, FileNotFoundError):
            ds = None                                 # arrays not (all) extracted: use a matching bundle instead
        if ds is not None:
            pol = PO.SBSBudget().fit(ds, np.arange(len(ds.rows)))
            ranks = {int(b): [ds.protocols[j] for j in r] for b, r in pol.rank.items()}
            return _first_feasible_choice(ranks, tab), "refitted on the training labels %s" % ",".join(
                os.path.join(os.path.basename(run_dir), p) for p in parts)
    for name, meta in bundle_metas:
        tp = str((meta.get("training") or {}).get("parts"))
        if parts and tp != ",".join(parts) and tp != str(ev.get("train_parts")):
            continue
        ranks = {int(b): r for b, r in (meta.get("fallback") or {}).get("ranks", {}).items()}
        if ranks:
            return _first_feasible_choice(ranks, tab), "fallback ranks of bundle %s (trained on %s)" % (name, tp)
    raise UncheckedBaseline(
        "%s: the SBS-B baseline cannot be checked against its rule — the training labels %s are not available (extract "
        "their trial data) and no bundle given was trained on them" % (labels_dir, ",".join(parts) or "(not recorded)"))


def check_sbsb_column(rows: dict, rule: dict, column: str = "SBS-B|choice", what: str = "choices file") -> int:
    """Compare a saved SBS-B column {(task, B): row} with the rule's choices; ValueError on any difference. Returns the
    number of rows checked (every saved row must be covered by the rule)."""
    bad = [k for k, r in rows.items() if k in rule and r[column] != rule[k]]
    uncovered = [k for k in rows if k not in rule]
    if bad:
        raise ValueError("the SBS-B choices of %s differ from the SBS-B rule on %d rows (e.g. %s: %s vs %s); the "
                         "baseline column was changed or belongs to another training" % (
                             what, len(bad), bad[0], rows[bad[0]][column], rule[bad[0]]))
    return len(rows) - len(uncovered)


def _feasible(v) -> bool:
    """A labels table's ``feasible`` value (a bool, or its CSV spelling) as a bool; anything else is False."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "1.0")


def replay(labels_dir: str, bundle_path: str, *, allow_simulation: bool = False, preflight_level=None,
           design_seeds: str = "deployment", budgets=None, progress=None, excluded: list | None = None,
           require_complete: bool = True, subset=None, subset_reason: str = "",
           provenance: dict | None = None) -> list:
    """Rows (task, B, status, choice, accepted, model choice, skipped candidates, preflight of the choice, risk of the
    choice on the evaluation half, seconds, feature cache). ``design_seeds``: 'deployment' (each experiment's own
    design seed, as the application) or 'study' (the labels' allocation seed — ``emulate.allocation_seed``, scheme 1
    or 2 — the design the benchmark's feasibility was decided on). Label tasks without labels (error, timeout, memory)
    cannot be scored by any policy; their ids are appended to ``excluded`` (when given) so the caller reports them.

    Seconds are classical wall time of the call; features are cached per experiment, so the first budget of a task is
    a full recomputation (``cache`` = 'miss') and later budgets reuse the budget-independent features ('hit').

    1.9.0 (second review N4): the evaluated population is the bound run's manifest, every record validated
    (``bound_population``); an incomplete run is refused unless an exploratory ``subset`` with its reason is declared.
    ``excluded`` receives every manifest task not evaluated and why; ``provenance`` (a dict, filled in) records the
    labels' run identity and data digest, the bundle's digests, the population and the exclusions."""
    from . import deploy as DP
    from . import emulate as EM
    from . import labels as LB
    from . import task as TK
    pop = bound_population(labels_dir, require_complete=require_complete, subset=subset, subset_reason=subset_reason)
    tab = pop["table"]
    if tab is None:
        raise ValueError("%s: no labeled task to replay (%s)" % (labels_dir, pop["completion"]))
    risk = {(r.task_id, r.protocol, int(r.B)): (_feasible(r.feasible), float(r.mse_eval)) for r in tab.itertuples()}
    b = DP.load_bundle(bundle_path)
    if excluded is not None:
        excluded.extend(pop["excluded"])
    if provenance is not None:
        provenance.update(labels=dict({k: v for k, v in pop.items() if k not in ("table", "used")},
                                      evaluated_tasks=len(pop["used"])),
                          bundle=bundle_identity(bundle_path), allow_simulation=bool(allow_simulation),
                          design_seeds=design_seeds, preflight_level=preflight_level, policy=DP.POLICY,
                          estimand="fresh-design average risk R_a^fresh(T, B) of the returned candidate (the labels' "
                                   "evaluation half averages independently drawn training designs); not the risk "
                                   "conditional on the design the API returns for execution")
    rows = []
    for tid in pop["used"]:
        rec = LB.load_record(labels_dir, tid)
        spec = TK.TaskSpec.from_json(rec["spec"])
        seeds = None
        if design_seeds == "study":
            seeds = {k: EM.allocation_seed(rec, k) for k in rec["protocols"]}
        for B in (budgets or rec["budgets"]):
            t0 = time.perf_counter()
            task = DP.from_spec(spec, int(B))
            try:
                r = DP.recommend(task, b, allow_simulation=allow_simulation, preflight_level=preflight_level,
                                 design_seeds=seeds)
            except DP.Unsupported as e:
                rows.append(dict(task_id=tid, base_id=rec["base_id"], family=spec.family, n=spec.n, B=int(B),
                                 status="unsupported", reason=str(e), choice="", scored=False, risk=math.nan,
                                 seconds=time.perf_counter() - t0))
                continue
            k = r["choice"]
            feas, mse = risk.get((tid, k, int(B)), (False, math.nan))
            rows.append(dict(task_id=tid, base_id=rec["base_id"], family=spec.family, n=spec.n, B=int(B),
                             status=r["status"], accepted=bool(r["accepted"]), choice=k,
                             model_choice=r.get("model_choice"), fallback=r.get("fallback"),
                             ood=len(r.get("ood", [])), margin=r.get("margin"),
                             skipped=";".join("%s:%s" % (x["protocol"], x["status"]) for x in r["skipped_candidates"]),
                             choice_preflight=r["preflight"].get(k, {}).get("status"),
                             choice_level=r["preflight"].get(k, {}).get("level"),
                             scored=bool(feas and np.isfinite(mse)), risk=mse if feas else math.nan,
                             seconds=time.perf_counter() - t0,
                             cache="hit" if str(r.get("feature_report", {}).get("cache", "")).startswith("hit")
                             else "miss",
                             predicted=TX.dumps(r.get("predicted_log10_risk") or {})))
            if progress:
                progress(rows[-1])
    return rows


def component_rows(rows, choices: dict, column: str, risk: dict) -> list:
    """The model-component benchmark on the same rows as a replay: for each replayed (task, budget), the candidate the
    benchmark recorded in ``choices[(task_id, B)][column]`` (the model's best label-feasible candidate, without the
    deployed policy's OOD, preflight and margin rules), scored with the evaluation half (``risk[(task, protocol, B)]``).
    Every field describes the component (choice, status), not the policy; a missing choice or risk is unscored."""
    out = []
    for r in rows:
        c = (choices.get((r["task_id"], int(r["B"]))) or {}).get(column) or ""
        v = float(risk.get((r["task_id"], c, int(r["B"])), math.nan)) if c else math.nan
        out.append(dict(task_id=r["task_id"], base_id=r["base_id"], family=r.get("family"), n=r.get("n"),
                        B=int(r["B"]), status="model-component", accepted=True, choice=c, skipped="",
                        scored=bool(np.isfinite(v)), risk=v, seconds=math.nan, cache="n/a"))
    return out


def component_rows_from_predictions(rows, risk: dict, feasible: dict) -> list:
    """The model component of a bundle on the replayed rows when the confirmation's evaluation did not include its
    configuration: the bundle's own predictions (recorded per row by ``replay``) minimised over the candidates the
    labels mark feasible — the definition of the model-component benchmark — scored with the evaluation half."""
    out = []
    for r in rows:
        pred = TX.loads(r.get("predicted") or "{}") if isinstance(r.get("predicted"), str) else (r.get("predicted")
                                                                                                or {})
        cand = [k for k, v in pred.items() if feasible.get((r["task_id"], k, int(r["B"]))) and np.isfinite(v)]
        c = min(cand, key=lambda k: pred[k]) if cand else ""
        v = float(risk.get((r["task_id"], c, int(r["B"])), math.nan)) if c else math.nan
        out.append(dict(task_id=r["task_id"], base_id=r["base_id"], family=r.get("family"), n=r.get("n"),
                        B=int(r["B"]), status="model-component", accepted=True, choice=c, skipped="",
                        scored=bool(np.isfinite(v)), risk=v, seconds=math.nan, cache="n/a"))
    return out


def summarize(rows, baseline: dict, *, reps: int = 1000, seed: int = 0) -> dict:
    """Primary comparison of the replayed policy with a baseline (``baseline``: {(task_id, B): risk}, e.g. SBS-B's
    chosen risk on the evaluation half): base-circuit-weighted mean of log10(risk + 1e-6) and its paired difference
    with a percentile bootstrap over base circuits; arithmetic MSE; declared tails; counts. Undefined (NaN, with the
    reason) if any row is unscored."""
    from .evaluate import SEVERE_ABS, tail_metrics, wmean
    out = dict(rows=len(rows), base_circuits=len({r["base_id"] for r in rows}),
               unscored=sum(1 for r in rows if not r["scored"]),
               status={s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
               accepted=sum(1 for r in rows if r.get("accepted")),
               with_skipped_candidates=sum(1 for r in rows if r.get("skipped")),
               choice_counts={k: sum(1 for r in rows if r["choice"] == k) for k in sorted({r["choice"] for r in rows})},
               seconds_median=float(np.nanmedian([r["seconds"] for r in rows]))
               if rows and np.isfinite([r["seconds"] for r in rows]).any() else math.nan)
    miss = [r["seconds"] for r in rows if r.get("cache", "miss") == "miss" and np.isfinite(r["seconds"])]
    out["seconds_median_fresh"] = float(np.median(miss)) if miss else math.nan     # a full recomputation per call
    out["seconds_fresh_calls"] = len(miss)
    base = np.array([baseline.get((r["task_id"], r["B"]), math.nan) for r in rows])
    pol = np.array([r["risk"] for r in rows], float)
    g = np.array([r["base_id"] for r in rows])
    if out["unscored"] or not np.all(np.isfinite(base)) or not np.all(np.isfinite(pol)):
        out.update(primary="undefined: %d unscored rows, %d rows without a baseline risk, %d rows without a policy "
                   "risk" % (out["unscored"], int(np.sum(~np.isfinite(base))), int(np.sum(~np.isfinite(pol)))))
        return out
    lp, lb = np.log10(pol + EPS0), np.log10(base + EPS0)
    u = sorted(set(g.tolist()))
    mp = np.array([lp[g == k].mean() for k in u])
    mb = np.array([lb[g == k].mean() for k in u])
    d = mp - mb
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(u), size=(int(reps), len(u)))
    bs = d[picks].mean(1)
    out.update(primary="mean over base circuits of log10(MSE + 1e-6): policy − baseline (paired bootstrap)",
               policy_mean_log10=float(mp.mean()), baseline_mean_log10=float(mb.mean()),
               diff=float(d.mean()), diff_lo=float(np.quantile(bs, 0.025)), diff_hi=float(np.quantile(bs, 0.975)),
               ratio_geomean_mse_plus_floor=float(10 ** d.mean()),
               policy_mse=wmean(pol, g), baseline_mse=wmean(base, g),
               policy_tail=tail_metrics(pol, g), baseline_tail=tail_metrics(base, g), severe_abs_threshold=SEVERE_ABS)
    return out


def write(rows, summary: dict, out_dir: str, tag: str) -> None:
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "replay_%s.csv" % tag), index=False, encoding="utf-8")
    TX.write_json(os.path.join(out_dir, "replay_%s.json" % tag), summary)
