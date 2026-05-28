"""Stage 4 — Aggregator (pure Python rules, no LLM).

Two deterministic functions that turn typed evidence into a typed verdict:

* :func:`match_clinical_pattern` maps (Tier-1, Tier-2) onto one of 6 predefined
  AD-spectrum patterns (CLAUDE.md §2.5). First-match-wins over an order chosen
  by specificity, so the most distinctive picture (e.g. a CAA signature) is not
  swallowed by a broader one (e.g. "amyloid-negative impairment").
* :func:`detect_conflicts` surfaces evidence conflicts (CLAUDE.md §2.6) — an AD
  clinical syndrome without amyloid, amyloid without a clinical signal, atrophy
  without amyloid, mixed vascular+amyloid burden, or clinical severity out of
  proportion to imaging. **Conflicts are surfaced, never silently overridden.**

These are rules, not a model: the same evidence always yields the same pattern
and the same conflicts, which is what makes the downstream report auditable.
Every evidence string carries the value, the threshold it was judged against,
and (where relevant) the reference, so a claim can always be traced back.
"""

from __future__ import annotations

from .schemas import (
    ClinicalPattern,
    Conflict,
    ConflictSeverity,
    Tier1Result,
    Tier2Result,
)

# --- thresholds (kept named so the report can cite them) ---------------------
P_AD_SYNDROME = 0.50        # Tier-1 P(AD) at/above which we call it an AD syndrome
P_AD_ELEVATED = 0.30
P_MCI_ELEVATED = 0.40
P_VD_ELEVATED = 0.20
P_FTD_ELEVATED = 0.30
P_PSY_ELEVATED = 0.30
HIPPO_Z_ATROPHY = -1.5      # ≤ this = significant medial-temporal atrophy
HIPPO_Z_BENIGN = -1.0       # > this = no meaningful hippocampal atrophy
CENTILOID_POS = 20.0        # GAAIN positivity (Klunk 2015)


def _read(tier1: Tier1Result, tier2: Tier2Result) -> dict:
    """Flatten the handful of values the rules read, with safe defaults."""
    c = tier2.centiloid
    a = tier2.anatomy
    return {
        "p_ad": tier1.p_ad,
        "p_mci": tier1.p_mci,
        "p_vd": tier1.p_vd,
        "p_ftd": tier1.p("FTD"),
        "p_psy": tier1.p("PSY"),
        "stage": tier1.stage_top,
        "impaired": tier1.stage_top in ("MCI", "DE"),
        "amyloid_known": c is not None,
        "amyloid_pos": (c.positive if c is not None else None),
        "centiloid": (c.centiloid if c is not None else None),
        "hippo_z": (a.hippocampus_zscore if a is not None else None),
        "atrophy": (a.dominant_atrophy if a is not None else "none"),
    }


# ============================================================================ #
# Pattern matching
# ============================================================================ #

def match_clinical_pattern(tier1: Tier1Result, tier2: Tier2Result) -> ClinicalPattern:
    """Map the combined evidence onto one of the AD-spectrum patterns."""
    v = _read(tier1, tier2)
    has_atrophy = v["hippo_z"] is not None and v["hippo_z"] <= HIPPO_Z_ATROPHY
    vascular = v["p_vd"] >= P_VD_ELEVATED

    # 3 — Mixed AD + cerebrovascular.
    if v["amyloid_pos"] and vascular:
        return ClinicalPattern(
            pattern_id=3,
            name="Mixed AD + cerebrovascular",
            rationale="Amyloid positivity alongside a Tier-1 vascular signal "
                      "indicates a mixed substrate; cognitive impairment is "
                      "likely multifactorial.",
            supporting_evidence=[
                f"Amyloid-positive (Centiloid {v['centiloid']} ≥ {CENTILOID_POS}).",
                f"Vascular load: P(VD)={v['p_vd']:.2f} (≥{P_VD_ELEVATED}).",
            ],
            confidence="moderate",
        )

    # 1 — Typical amyloid-positive AD.
    if v["amyloid_pos"] and (has_atrophy or v["atrophy"] == "medial_temporal"):
        ev = [f"Amyloid-positive (Centiloid {v['centiloid']} ≥ {CENTILOID_POS}; Klunk 2015)."]
        if v["hippo_z"] is not None:
            ev.append(f"Hippocampal Z {v['hippo_z']} (≤ {HIPPO_Z_ATROPHY} = atrophy).")
        ev.append(f"Tier-1 P(AD)={v['p_ad']:.2f}, stage {v['stage']}.")
        return ClinicalPattern(
            pattern_id=1,
            name="Typical amyloid-positive Alzheimer's disease",
            rationale="Amyloid positivity with medial-temporal atrophy and a "
                      "concordant clinical profile fits a typical AD picture "
                      "(NIA-AA A+ with neurodegeneration).",
            supporting_evidence=ev,
            confidence="high" if v["p_ad"] >= P_AD_SYNDROME else "moderate",
        )

    # 5 — Frontotemporal-predominant (non-amyloid).
    if (v["atrophy"] == "frontotemporal" or v["p_ftd"] >= P_FTD_ELEVATED) and not v["amyloid_pos"]:
        return ClinicalPattern(
            pattern_id=5,
            name="Frontotemporal-predominant atrophy (non-amyloid)",
            rationale="A frontotemporal atrophy emphasis without amyloid positivity "
                      "points away from AD and toward an FTD-spectrum process.",
            supporting_evidence=[
                f"Tier-1 P(FTD)={v['p_ftd']:.2f} (≥{P_FTD_ELEVATED}).",
                f"Dominant atrophy: {v['atrophy']}.",
                ("Amyloid-negative." if v["amyloid_known"] else "Amyloid status unknown."),
            ],
            confidence="moderate",
        )

    # 6 — Pseudodementia / functional (no structural-amyloid substrate).
    benign_structure = v["hippo_z"] is None or v["hippo_z"] > HIPPO_Z_BENIGN
    if (v["p_psy"] >= P_PSY_ELEVATED and benign_structure
            and v["amyloid_pos"] is not True):
        return ClinicalPattern(
            pattern_id=6,
            name="Suspected pseudodementia / functional",
            rationale="Cognitive complaints without an amyloid or structural "
                      "substrate raise the possibility of a functional/affective "
                      "contributor; longitudinal follow-up is needed to exclude an "
                      "early neurodegenerative process.",
            supporting_evidence=[
                f"Tier-1 P(PSY)={v['p_psy']:.2f} (≥{P_PSY_ELEVATED}).",
                ("No significant hippocampal atrophy "
                 f"(Z {v['hippo_z']} > {HIPPO_Z_BENIGN})."
                 if v["hippo_z"] is not None else "No structural atrophy flagged."),
                ("Amyloid-negative." if v["amyloid_known"] else "Amyloid status unknown."),
            ],
            confidence="low",
        )

    # 2 — Amyloid-negative cognitive impairment (suspected non-AD); catch-all
    #     for an impaired patient who did not fit a more specific pattern.
    ev = [f"Cognitive stage {v['stage']}; P(MCI)={v['p_mci']:.2f}, P(AD)={v['p_ad']:.2f}."]
    if v["amyloid_known"]:
        ev.append(f"Amyloid-negative (Centiloid {v['centiloid']} < {CENTILOID_POS}).")
    else:
        ev.append("Amyloid status unknown (no PET quantified).")
    if v["hippo_z"] is not None:
        ev.append(f"Hippocampal Z {v['hippo_z']}.")
    return ClinicalPattern(
        pattern_id=2,
        name="Amyloid-negative cognitive impairment (suspected non-AD)",
        rationale="Cognitive impairment without confirmed amyloid suggests a "
                  "non-AD or suspected-non-amyloid-pathology (SNAP) process; the "
                  "differential stays broad.",
        supporting_evidence=ev,
        confidence="moderate" if v["impaired"] else "low",
    )


# ============================================================================ #
# Conflict detection
# ============================================================================ #

def detect_conflicts(tier1: Tier1Result, tier2: Tier2Result) -> list[Conflict]:
    """Surface evidence conflicts. Order is stable; the list may be empty."""
    v = _read(tier1, tier2)
    conflicts: list[Conflict] = []

    # AD clinical syndrome without amyloid confirmation.
    if v["amyloid_pos"] is False and v["p_ad"] >= P_AD_SYNDROME:
        conflicts.append(Conflict(
            conflict_id="amyloid_negative_ad_syndrome",
            name="Amyloid-negative AD syndrome",
            description="A strong Tier-1 AD signal is not supported by amyloid PET. "
                        "Consider AD mimics (LATE, hippocampal sclerosis, primary "
                        "tauopathy) or a false-negative amyloid read.",
            severity=ConflictSeverity.IMPORTANT,
            evidence=[
                f"P(AD)={v['p_ad']:.2f} (≥{P_AD_SYNDROME}).",
                f"Centiloid {v['centiloid']} < {CENTILOID_POS} (amyloid-negative).",
            ],
        ))

    # Amyloid present without a matching clinical signal.
    if v["amyloid_pos"] is True and v["p_ad"] < P_AD_ELEVATED and v["stage"] == "NC":
        conflicts.append(Conflict(
            conflict_id="subclinical_amyloid",
            name="Subclinical / preclinical amyloid",
            description="Amyloid is present without a concordant clinical AD signal — "
                        "compatible with preclinical AD. Not diagnostic on its own.",
            severity=ConflictSeverity.CAUTION,
            evidence=[
                f"Centiloid {v['centiloid']} ≥ {CENTILOID_POS} (amyloid-positive).",
                f"P(AD)={v['p_ad']:.2f} (<{P_AD_ELEVATED}); stage {v['stage']}.",
            ],
        ))

    # Hippocampal atrophy without amyloid.
    if (v["amyloid_pos"] is False and v["hippo_z"] is not None
            and v["hippo_z"] <= HIPPO_Z_ATROPHY):
        conflicts.append(Conflict(
            conflict_id="atrophy_without_amyloid",
            name="Hippocampal atrophy without amyloid",
            description="Medial-temporal atrophy without amyloid suggests a non-AD "
                        "neurodegenerative process (e.g. LATE, hippocampal sclerosis, "
                        "SNAP) rather than typical AD.",
            severity=ConflictSeverity.CAUTION,
            evidence=[
                f"Hippocampal Z {v['hippo_z']} (≤{HIPPO_Z_ATROPHY}).",
                f"Centiloid {v['centiloid']} < {CENTILOID_POS} (amyloid-negative).",
            ],
        ))

    # Mixed amyloid + vascular burden (Tier-1 signal only).
    if v["amyloid_pos"] is True and v["p_vd"] >= P_VD_ELEVATED:
        conflicts.append(Conflict(
            conflict_id="vascular_amyloid_overlap",
            name="Concurrent amyloid and vascular burden",
            description="Both amyloid and vascular markers are elevated; the "
                        "contribution of each to the cognitive picture cannot be "
                        "apportioned from this evidence alone.",
            severity=ConflictSeverity.CAUTION,
            evidence=[
                f"Centiloid {v['centiloid']} ≥ {CENTILOID_POS} (amyloid-positive).",
                f"P(VD)={v['p_vd']:.2f} (≥{P_VD_ELEVATED}).",
            ],
        ))

    # Clinical severity out of proportion to imaging.
    benign_imaging = (
        v["amyloid_pos"] is not True
        and (v["hippo_z"] is None or v["hippo_z"] > HIPPO_Z_BENIGN)
    )
    if v["stage"] == "DE" and benign_imaging:
        conflicts.append(Conflict(
            conflict_id="severity_imaging_mismatch",
            name="Clinical severity exceeds imaging burden",
            description="A dementia-level clinical stage is not matched by amyloid "
                        "or structural findings. Consider non-degenerative "
                        "contributors (mood, metabolic, medication) and follow-up.",
            severity=ConflictSeverity.CAUTION,
            evidence=[
                f"Stage {v['stage']} with benign imaging "
                f"(hippocampal Z {v['hippo_z']}, "
                f"amyloid {'negative' if v['amyloid_known'] else 'unknown'}).",
            ],
        ))

    return conflicts
