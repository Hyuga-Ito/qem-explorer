"""Full-policy evaluation of the shipped bundles on a confirmation set (1.8.0; review finding F1).

For each bundle, ``deploy.recommend`` itself — guard, features, out-of-distribution checks, preflight contract,
margin rule, fallback — is called on every (task, budget) of the confirmation labels and the returned candidate is
scored on the evaluation half (qemx.selection.replay). It is compared with SBS-B (the budget-only fixed policy
trained on the development labels, as in the confirmation's model-component evaluation) and, separately, the
model-component benchmark of the same configuration is summarized on the same rows. Run once, after the freeze.

    python scripts/replay_confirmation.py --labels runs/v18/confirm --out runs/v18/replay
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BUNDLES = (("qemx-rec-1.8.0", True), ("qemx-rec-1.8.0-deployable", False))


def confirmation_status(labels: str, bundle_meta: dict, prov: dict) -> tuple:
    """(is the once-only confirmation, reasons) — 1.9.0 (second review N4): the wording "once-only confirmation" is
    used only when the labels directory holds the once-only evaluation record (EVALUATED) of a freeze, the bundle was
    trained under that same freeze, and the replay evaluated the complete bound population (no subset, no exclusion).
    Otherwise the replay is exploratory, with the reasons."""
    from qemx import textio as TX
    why = []
    ev_p = os.path.join(labels, "EVALUATED")
    if not os.path.exists(ev_p):
        why.append("no once-only evaluation record (EVALUATED) in %s" % os.path.basename(os.path.normpath(labels)))
        ev = {}
    else:
        ev = TX.read_json(ev_p)
    bf = (bundle_meta.get("training") or {}).get("freeze") or {}
    ef = ev.get("freeze") or {}
    for k in ("policy", "config_digest", "dev_identity", "deploy_tau"):
        if ev and bf.get(k) != ef.get(k):
            why.append("the bundle's training freeze differs from the evaluated freeze in %s" % k)
    lab = prov.get("labels", {})
    if lab.get("subset") or lab.get("excluded"):
        why.append("not the complete bound population (%d excluded)" % len(lab.get("excluded") or []))
    if ev.get("data_digest") and lab.get("data_digest") and ev["data_digest"] != lab["data_digest"]:
        why.append("the labels' content changed since the once-only evaluation")
    return (not why), why


def sbsb_choices(meta: dict, tab) -> dict:
    """SBS-B's choice per (task, budget) from a bundle's fallback ranks (kept for scripts that import it; 1.9.0 checks
    use ``replay.sbsb_rule_choices``, which refits the rule on the evaluation's own training labels when present)."""
    from qemx.selection import replay as RP
    ranks = {int(b): r for b, r in (meta.get("fallback") or {}).get("ranks", {}).items()}
    return RP._first_feasible_choice(ranks, tab)


def main(argv=None) -> int:
    from qemx.selection import binding as BD, integrity as IN, replay as RP
    try:
        return _main(argv)
    except (IN.IncompleteRun, IN.CorruptedRun, RP.StaleTable, BD.IncompatibleRun, ValueError, OSError) as e:  # incl. UncheckedBaseline
        print("refused: %s" % e)
        return 2


def _main(argv=None) -> int:
    from qemx import textio as TX
    from qemx.selection import deploy as DP
    from qemx.selection import replay as RP
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--choices", default=None, help="model-component choices (default LABELS/choices_confirm.csv)")
    ap.add_argument("--bundle-dir", default=None,
                    help="directory holding the two bundles (default: the shipped bundles, qemx/selection/bundles)")
    ap.add_argument("--bundles", default=None, help="comma-separated NAME:sim|nosim pairs (default: the two 1.8.0 "
                    "bundles, simulator-feature with simulation allowed, deployable without)")
    ap.add_argument("--subset", default=None, help="exploratory only: a file with one task id per line to replay "
                    "instead of the complete bound population (needs --subset-reason)")
    ap.add_argument("--subset-reason", default="")
    a = ap.parse_args(argv)
    bundles = BUNDLES if not a.bundles else tuple((x.split(":")[0], x.split(":")[1] == "sim")
                                                  for x in a.bundles.split(","))
    subset = None
    if a.subset:
        subset = [ln.strip() for ln in open(a.subset, encoding="utf-8") if ln.strip()]
    ch = list(csv.DictReader(open(a.choices or os.path.join(a.labels, "choices_confirm.csv"), encoding="utf-8")))
    pop = RP.bound_population(a.labels, require_complete=subset is None, subset=subset, subset_reason=a.subset_reason)
    tab = pop["table"]
    risk = {(r.task_id, r.protocol, int(r.B)): float(r.mse_eval) for r in tab.itertuples()}
    key = {(r["task_id"], int(r["B"])): r for r in ch}
    missing = sorted(set((t, int(b)) for t, b in zip(tab.task_id, tab.B)) - set(key))
    if missing:
        raise ValueError("choices file lacks %d evaluated (task, budget) rows, e.g. %s" % (len(missing), missing[:2]))
    metas = [(name, DP.load_bundle(os.path.join(a.bundle_dir or DP.BUNDLE_DIR, name)).meta) for name, _ in bundles]
    rule, rule_source = RP.sbsb_rule_choices(a.labels, tab, metas)       # refuses when it cannot be checked
    checked = RP.check_sbsb_column(key, rule, what=os.path.basename(a.choices or "choices_confirm.csv"))
    baseline_check = dict(source=rule_source, rows_checked=checked, rows=len(key))
    print("SBS-B baseline column checked against the SBS-B rule (%s): %d/%d rows agree"
          % (rule_source, checked, len(key)))
    sbsb = {k: risk.get((k[0], r["SBS-B|choice"], k[1]), float("nan")) for k, r in key.items()}
    out = {}
    for name, sim in bundles:
        path = os.path.join(a.bundle_dir or DP.BUNDLE_DIR, name)
        b = DP.load_bundle(path)
        excluded, prov = [], {}
        rows = RP.replay(a.labels, path, allow_simulation=sim, excluded=excluded, subset=subset,
                         subset_reason=a.subset_reason, require_complete=subset is None, provenance=prov)
        s = RP.summarize(rows, sbsb)
        col = "%s|choice" % b.meta["configuration"]
        if ch and col in ch[0]:
            s["model_component_benchmark"] = RP.summarize(RP.component_rows(rows, key, col, risk), sbsb)
            s["model_component_configuration"] = b.meta["configuration"]
        else:          # the confirmation's declared comparators did not include this configuration (D39 lists the
            # 1.7.0 deployable set): its model component from the bundle's own predictions on the same rows, labelled
            feas = {(r.task_id, r.protocol, int(r.B)): RP._feasible(r.feasible) for r in tab.itertuples()}
            s["model_component_benchmark"] = RP.summarize(RP.component_rows_from_predictions(rows, risk, feas), sbsb)
            s["model_component_configuration"] = b.meta["configuration"]
            s["model_component_source"] = ("computed from the bundle's predictions on the replayed rows (not among the "
                                           "confirmation's declared comparators; reported, not claimed)")
        is_conf, why = confirmation_status(a.labels, b.meta, prov)
        s.update(bundle=name, allow_simulation=sim, policy=DP.POLICY, excluded_label_tasks=excluded,
                 provenance=prov, baseline_check=baseline_check, confirmation=is_conf, not_confirmation_because=why,
                 status_note=("once-only confirmation of the frozen policy; full deployed policy" if is_conf else
                              "exploratory replay of the full deployed policy (not the once-only confirmation: %s)"
                              % "; ".join(why)),
                 estimand=("mean over base circuits of log10(R + 1e-6), R the fresh-design average risk of the returned "
                           "candidate on the labels' evaluation half; policy minus SBS-B"))
        RP.write(rows, s, a.out, name)
        out[name] = {k: s.get(k) for k in ("rows", "base_circuits", "unscored", "status", "accepted",
                                           "with_skipped_candidates", "diff", "diff_lo", "diff_hi", "policy_mse",
                                           "baseline_mse", "seconds_median_fresh", "confirmation", "baseline_check")}
    TX.write_json(os.path.join(a.out, "replay_summary.json"), out)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
