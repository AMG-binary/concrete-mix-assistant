"""
Recommendation engine — Concrete Mix Design Assistant (Phase 3)

Policy (mirrors the problem statement):
  1. COMPLIANCE FIRST — if cement < Table 5 minimum, raise cement to the
     minimum; if W/C > Table 5 maximum, cut water to the allowed maximum.
     These fixes target the violated CONSTRAINT itself (cement content or
     W/C ratio), not the strength number.
  2. STRENGTH SECOND — if the (possibly fixed) mix still predicts below the
     target grade's minimum strength, add cement: the highest-leverage
     ADJUSTABLE ingredient per the SHAP ranking (age is excluded because
     the app locks it at 28 days). The first delta estimate comes from the
     model's own local sensitivity (finite difference), then the engine
     VERIFIES by re-running the prediction and iterating in 10 kg/m3 steps
     up to a physical cap of 550 kg/m3 (just above the dataset max of 540).
  3. CONSTANT TOTAL WEIGHT — every adjustment is rebalanced through fine
     aggregate (the slack variable) so the mix still sums to the user's
     original total (~2400 kg/m3 per the PS note).
  4. HONEST LIMITS — if the target cannot be reached within the cap, the
     engine says so instead of pretending success.

Demo (run from the repo root, after Phase 1):
    python src/recommender.py
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    from src.compliance import IS_456_TABLE_5, verify_is456
    from src.model_handler import INGREDIENTS
except ImportError:  # allows `python src/recommender.py` from the repo root
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.compliance import IS_456_TABLE_5, verify_is456
    from src.model_handler import INGREDIENTS

CEMENT_CAP = 550.0        # physical cap, just above dataset max (540 kg/m3)
CEMENT_STEP = 10.0        # site-practical increment
SAFETY_MARGIN = 1.0       # MPa buffer for model uncertainty (test RMSE 4.39)
FINE_AGG_MIN = 500.0      # sanity bounds for the slack variable
FINE_AGG_MAX = 1050.0


def _rebalance(mix: dict, target_total: float, notes: list) -> None:
    """Fine aggregate is the slack variable: keep total mix weight constant."""
    others = sum(mix[k] for k in INGREDIENTS if k != "fine_aggregate")
    fine = target_total - others
    if fine < FINE_AGG_MIN:
        notes.append(
            f"Fine aggregate hit its lower bound ({FINE_AGG_MIN:.0f} kg/m3); "
            f"total mix weight rises above the original {target_total:.0f} kg/m3."
        )
        fine = FINE_AGG_MIN
    elif fine > FINE_AGG_MAX:
        notes.append(
            f"Fine aggregate hit its upper bound ({FINE_AGG_MAX:.0f} kg/m3); "
            f"total mix weight falls below the original {target_total:.0f} kg/m3."
        )
        fine = FINE_AGG_MAX
    mix["fine_aggregate"] = fine


def generate_recommendation(mix: dict, target_grade: str, model_handler) -> dict:
    """Return specific, verified ingredient adjustments for the target grade."""
    if target_grade not in IS_456_TABLE_5:
        raise ValueError(f"Unknown grade {target_grade!r}. Valid: {list(IS_456_TABLE_5)}")
    limits = IS_456_TABLE_5[target_grade]

    working = {k: float(mix[k]) for k in INGREDIENTS}
    target_total = sum(working.values())
    actions: list[str] = []
    notes: list[str] = []

    # ---- Step 1: fix violated IS 456 constraints (constraint-targeted) ----
    if working["cement"] < limits["min_cement"]:
        delta = limits["min_cement"] - working["cement"]
        working["cement"] = limits["min_cement"]
        _rebalance(working, target_total, notes)
        actions.append(
            f"Increase cement by +{delta:.1f} kg/m3 (to {limits['min_cement']:.0f}) "
            f"to meet the IS 456 minimum cement content for {target_grade}."
        )
    wc = (working["water"] / working["cement"]) if working["cement"] > 0 else float("inf")
    if wc > limits["max_wc"]:
        allowed = working["cement"] * limits["max_wc"]
        delta = working["water"] - allowed
        working["water"] = allowed
        _rebalance(working, target_total, notes)
        actions.append(
            f"Reduce water by -{delta:.1f} kg/m3 to bring W/C from {wc:.3f} down "
            f"to the IS 456 maximum of {limits['max_wc']:.2f} for {target_grade}."
        )

    pred = model_handler.predict(working)

    # ---- Step 2: fix strength shortfall (cement = primary SHAP lever) ----
    if pred < limits["min_strength"]:
        aim = limits["min_strength"] + SAFETY_MARGIN
        cement_before = working["cement"]

        # Local sensitivity of the model at this mix (includes rebalance effect)
        probe = dict(working)
        probe["cement"] = min(working["cement"] + CEMENT_STEP, CEMENT_CAP)
        _rebalance(probe, target_total, [])
        slope = 0.0
        if probe["cement"] > working["cement"]:
            slope = (model_handler.predict(probe) - pred) / (probe["cement"] - working["cement"])
        if slope > 0:
            est = min((aim - pred) / slope, CEMENT_CAP - working["cement"])
            est = math.ceil(est / CEMENT_STEP) * CEMENT_STEP  # round up to 10s
            if est > 0:
                working["cement"] += est
                _rebalance(working, target_total, notes)
                pred = model_handler.predict(working)

        # Verification loop: bounded +10 steps until target or cap
        while pred < aim and working["cement"] < CEMENT_CAP:
            working["cement"] = min(working["cement"] + CEMENT_STEP, CEMENT_CAP)
            _rebalance(working, target_total, notes)
            pred = model_handler.predict(working)

        added = working["cement"] - cement_before
        if pred >= limits["min_strength"]:
            actions.append(
                f"Increase cement by an additional +{added:.1f} kg/m3 (to "
                f"{working['cement']:.0f}) — the highest-leverage adjustable "
                f"ingredient per SHAP — to raise predicted strength to the "
                f"{target_grade} minimum of {limits['min_strength']:.0f} MPa."
            )
        else:
            actions.append(
                f"Could not reach {target_grade} even at the cement cap of "
                f"{CEMENT_CAP:.0f} kg/m3 (+{added:.1f} applied; best predicted "
                f"strength {pred:.2f} MPa). A redesign (water reduction, "
                f"admixtures, or a lower target grade) is required."
            )

    compliance = verify_is456(target_grade, pred, working["cement"], working["water"])
    return {
        "target_grade": target_grade,
        "original_mix": {k: float(mix[k]) for k in INGREDIENTS},
        "adjusted_mix": {k: round(v, 2) for k, v in working.items()},
        "actions": actions,
        "notes": notes,
        "updated_strength": round(pred, 2),
        "updated_compliance": compliance,
        "target_reached": compliance.strength_pass
        and compliance.cement_pass and compliance.wc_pass,
    }


if __name__ == "__main__":
    try:
        from src.model_handler import ModelHandler
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.model_handler import ModelHandler

    handler = ModelHandler()

    scenarios = [
        ("Weak mix vs M40 (compliance + strength fixes)", {
            "cement": 300.0, "slag": 0.0, "fly_ash": 0.0, "water": 190.0,
            "superplasticizer": 0.0, "coarse_aggregate": 1000.0, "fine_aggregate": 910.0,
        }, "M40"),
        ("Strong but non-compliant vs M30 (SCM-heavy, the decoupling case)", {
            "cement": 250.0, "slag": 200.0, "fly_ash": 100.0, "water": 170.0,
            "superplasticizer": 6.0, "coarse_aggregate": 1000.0, "fine_aggregate": 680.0,
        }, "M30"),
        ("Already sufficient vs M25 (no changes expected)", {
            "cement": 400.0, "slag": 100.0, "fly_ash": 0.0, "water": 160.0,
            "superplasticizer": 8.0, "coarse_aggregate": 1000.0, "fine_aggregate": 732.0,
        }, "M25"),
    ]

    for title, mix, grade in scenarios:
        before = handler.predict(mix)
        rec = generate_recommendation(mix, grade, handler)
        print(f"\n=== {title} ===")
        print(f"Initial prediction: {before:.2f} MPa | Target grade: {grade}")
        for a in rec["actions"] or ["No changes needed — mix is compliant and strong enough."]:
            print(f"  - {a}")
        for n in rec["notes"]:
            print(f"  note: {n}")
        print(f"Updated prediction: {rec['updated_strength']:.2f} MPa | "
              f"target_reached: {rec['target_reached']}")

