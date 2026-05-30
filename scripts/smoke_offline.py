#!/usr/bin/env python
"""Offline smoke test — exercises the credibility-critical, dependency-light path.

No torch / MONAI / live LLM required. It drives geometry, the router (offline
rules), derive_metric (on a seeded segmentation), the S4d finding-card translator
(incl. the number guard), S5 aggregation, the S6 report template, and the full
in-process spine (with the Xue model monkeypatched), then asserts the CLAUDE.md
invariants hold:

    #2 numbers computed, sentences carry only the metric's digits
    #6 segment_t1 runs once; metrics derive from the cache
    #7 every measured card has value+threshold+reference; negatives produce cards
    #8 explicit abstain cards for PRD/SEF/PSY/TBI/LBD
    #9 the report obeys all 6 CDS principles (cds_violations == [])

Run:  python scripts/smoke_offline.py     (exit 0 = all pass)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["CRYCHIC_LLM_PROVIDER"] = "offline"  # force offline (templates/rules)
os.environ.pop("NEMOTRON_URL", None)
os.environ["CRYCHIC_OVERLAY_DIR"] = tempfile.mkdtemp(prefix="crychic_smoke_")

import numpy as np  # noqa: E402

from crychic import aggregate, cds_guard, geometry, segmentation, xue  # noqa: E402
from crychic.agent import reasoner, router  # noqa: E402
from crychic.schemas import (  # noqa: E402
    Conflict, Differential, FindingCard, ImagingCheck, Metric, MetricStatus,
    ReconClass,
)

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        _FAILS.append(msg)


def _fake_seg(t1: str, *, hippo_small: bool) -> None:
    """Seed a synthetic segmentation (realistic-ish volumes) into the cache."""
    L = np.zeros((100, 120, 100), np.int16)
    L[10:90, 10:110, 10:90] = 1
    L[40:43, 80:104, 40:60] = 2
    L[57:60, 80:104, 40:60] = 3
    if hippo_small:                       # atrophic hippocampus → abnormal Z
        L[30:36, 40:60, 30:40] = 10; L[64:70, 40:60, 30:40] = 11
    else:                                 # generous hippocampus → normal Z
        L[26:40, 38:62, 28:42] = 10; L[60:74, 38:62, 28:42] = 11
    L[20:80, 70:108, 45:70] = 20
    L[15:85, 25:45, 25:45] = 21
    chan = {1: "Cerebral-White-Matter", 2: "Left-Lateral-Ventricle",
            3: "Right-Lateral-Ventricle", 10: "Left-Hippocampus",
            11: "Right-Hippocampus", 20: "Left-Frontal-Lobe", 21: "Left-Temporal-Lobe"}
    segmentation.put_segmentation(t1, segmentation.SegResult(
        image=L.astype("float32"), labels=L, vox_mm3=1.0, channel_def=chan,
        tiv_mm3=float((L > 0).sum()), hippo_idx=[10, 11], vent_idx=[2, 3]))


def _diff(**probs) -> Differential:
    base = {l: 0.05 for l in
            ("NC", "MCI", "DE", "AD", "LBD", "VD", "PRD", "FTD", "NPH", "SEF",
             "PSY", "TBI", "ODE")}
    base.update(probs)
    stage = {k: base[k] for k in ("NC", "MCI", "DE")}
    etio = {k: base[k] for k in
            ("AD", "LBD", "VD", "PRD", "FTD", "NPH", "SEF", "PSY", "TBI", "ODE")}
    return Differential(stage_probs=stage, etiology_probs=etio, all_probs=base,
                        stage_top=max(stage, key=stage.get),
                        etiology_top=max(etio, key=etio.get),
                        n_clinical_features=40, input_id="SMOKE")


def test_geometry() -> None:
    print("\n[geometry]")
    _fake_seg("g.nii", hippo_small=True)
    seg = segmentation.get_segmentation("g.nii")
    ev = geometry.evans_like_index(seg.labels, seg.channel_def, seg.vox_mm3)
    check(ev.index is not None and 0.0 < ev.index < 1.0,
          f"Evans-like index in (0,1): {ev.index}")
    hz = geometry.hippocampus_z_from_seg(seg.labels, seg.channel_def, seg.vox_mm3,
                                         age=78, sex="F")
    check(hz.z is not None and hz.z < 0, f"atrophic hippo → negative Z: {hz.z}")


def test_derive_and_translate() -> None:
    print("\n[derive_metric + translate]  (#2, #6, #7)")
    _fake_seg("c.nii", hippo_small=True)
    m = segmentation.derive_metric("c.nii", ImagingCheck.HIPPO_Z, age=78, sex="F")
    check(m.status is MetricStatus.MEASURED and m.value is not None, "hippo metric measured")
    card = reasoner.translate(m, check=ImagingCheck.HIPPO_Z, t1_path="c.nii")
    check(card.metric.value is not None and card.metric.threshold is not None
          and bool(card.references), "measured card has value+threshold+reference (#7)")
    # #2: every number in the sentence is a number the metric carries.
    check(reasoner.sentence_within_number_budget(card.sentence, _measured_src(m)),
          "finding sentence introduces no digit absent from the metric (#2)")
    check(not reasoner.sentence_within_number_budget(card.sentence + " 99.9",
          _measured_src(m)), "number guard rejects a stray digit (#2)")
    # #6: derive_metric did not re-segment (UNAVAILABLE when never segmented).
    miss = segmentation.derive_metric("never.nii", ImagingCheck.HIPPO_Z)
    check(miss.status is MetricStatus.UNAVAILABLE and miss.value is None,
          "no segmentation → UNAVAILABLE, no fabricated value (#2/#6)")


def _measured_src(m: Metric) -> str:
    return f"{m.value} {m.threshold} {m.unit}"


def test_router_and_recon() -> None:
    print("\n[router + aggregate reconciliation]  (#5, #8)")
    diff = _diff(MCI=0.55, DE=0.30, AD=0.62, FTD=0.40, NPH=0.05, VD=0.25)
    plan = asyncio.run(router.route(diff, t1_path="c.nii", flair_path=None))
    check(set(c.value for c in plan.checks) == {"hippo_z", "evans"},
          "T1 present → free structural baseline runs (hippo_z + evans) (#6/#7)")
    check("fazekas" not in [c.value for c in plan.checks] and "VD" in plan.abstained,
          "no FLAIR → VD abstained, fazekas not run")
    check({"PRD", "SEF", "PSY", "TBI", "LBD", "FTD"}.issubset(set(plan.abstained)),
          "PRD/SEF/PSY/TBI/LBD/FTD abstained (#8)")

    # Hand-built cards to exercise all four reconciliation classes deterministically.
    # (reconcile() is generic over the etiology gates, so a hand-built FTD metric
    # still drives DISCORDANT even though FTD has no live imaging axis.)
    cards = [
        _card("AD", abnormal=True),      # high AD + abnormal → concordant
        _card("FTD", abnormal=False),    # high FTD + normal  → discordant
        _card("NPH", abnormal=True),     # low NPH + abnormal → incidental
    ]
    cards.append(reasoner.abstain_card("LBD", set()))  # high LBD + no axis → clinical_only
    diff2 = _diff(DE=0.6, MCI=0.3, AD=0.62, FTD=0.40, NPH=0.05, LBD=0.55)
    recon = {r.etiology: r.recon for r in aggregate.reconcile(diff2, cards)}
    check(recon.get("AD") is ReconClass.CONCORDANT, "AD → concordant")
    check(recon.get("FTD") is ReconClass.DISCORDANT, "FTD → discordant")
    check(recon.get("NPH") is ReconClass.INCIDENTAL, "NPH → incidental")
    check(recon.get("LBD") is ReconClass.CLINICAL_ONLY, "LBD → clinical_only")
    conflicts = aggregate.detect_conflicts(diff2, cards)
    check(any(c.conflict_id == "ventriculomegaly_without_nph" for c in conflicts),
          "ventriculomegaly-without-NPH conflict surfaced")


def _card(etiology: str, *, abnormal: bool) -> FindingCard:
    thr = {"AD": -1.5, "FTD": -1.5, "NPH": 0.30}[etiology]
    cmp_ = ">" if etiology == "NPH" else "<"
    val = (thr - 0.5) if (cmp_ == "<" and abnormal) or (cmp_ == ">" and not abnormal) \
        else (thr + 0.5)
    m = Metric(etiology=etiology, name=f"{etiology} metric", value=round(val, 2),
               threshold=thr, comparator=cmp_, abnormal=abnormal,
               status=MetricStatus.MEASURED, reference="ref", unit="SD")
    return reasoner.translate(m)


def test_report_cds() -> None:
    print("\n[report CDS compliance]  (#9)")
    diff = _diff(MCI=0.55, DE=0.30, AD=0.62)
    cards = [_card("AD", abnormal=True), _card("NPH", abnormal=False),
             reasoner.abstain_card("LBD", set())]
    unified = aggregate.aggregate("SMOKE", diff, _plan(), cards)
    md = reasoner.template_report(unified)
    viol = cds_guard.cds_violations(md)
    check(viol == [], f"report passes all 6 CDS principles: {viol or 'PASS'}")
    check(cds_guard.SIGN_OFF in md and "○" in md, "sign-off + option bullets present (#9)")


def _plan():
    from crychic.schemas import ImagingPlan
    return ImagingPlan(checks=[ImagingCheck.HIPPO_Z, ImagingCheck.EVANS],
                       rationale="test", fired_rules=["r"],
                       abstained=["LBD", "PRD", "SEF", "PSY", "TBI", "VD", "ODE"])


def test_full_spine() -> None:
    print("\n[full in-process spine]  (xue monkeypatched; no torch)")
    from crychic import pipeline
    from crychic.state import STORE, CaseInputs

    _fake_seg("spine.nii", hippo_small=True)
    diff = _diff(MCI=0.55, DE=0.30, AD=0.62, NPH=0.05, FTD=0.05, VD=0.10)

    def _fake_screen(clinical, mri=None, explain=True, out_dir=None):  # bypass torch/ADRD
        return diff
    xue.screen = _fake_screen

    state = STORE.create(CaseInputs(
        clinical={"NACCAGE": 74, "SEX": 2, "NACCMMSE": 22}, t1_path="spine.nii"))
    asyncio.run(pipeline.run_pipeline(state))

    if state.stage == "failed":
        print("    spine error:\n" + (state.error or "").rstrip()[-800:])
    check(state.stage == "complete" and state.report is not None,
          f"spine completed: stage={state.stage}")
    check("xue_predict" in state.completed_tools
          and "segment_t1" in state.completed_tools, "tools recorded (xue + segment)")
    pol = {c.polarity for c in state.cards}
    check({"supporting", "negative", "abstain"} <= pol,
          f"cards span supporting/negative/abstain: {sorted(pol)}")
    check(any(c.etiology == e and c.polarity == "abstain"
              for e in ("PRD", "SEF", "PSY", "TBI", "LBD") for c in state.cards),
          "abstain cards for the non-structural etiologies (#8)")
    if state.report:
        check(cds_guard.cds_violations(state.report.markdown) == [],
              "final report is CDS-compliant (#9)")


def main() -> int:
    test_geometry()
    test_derive_and_translate()
    test_router_and_recon()
    test_report_cds()
    test_full_spine()
    print("\n" + ("=" * 56))
    if _FAILS:
        print(f"SMOKE FAILED — {len(_FAILS)} assertion(s):")
        for f in _FAILS:
            print(f"  - {f}")
        return 1
    print("SMOKE PASSED — all invariant checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
