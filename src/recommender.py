"""Recommendation engine for the Concrete Mix Design Assistant.

Run from the repository root:
    python src/recommender.py
"""

from __future__ import annotations

import math

from src.compliance import IS_456_TABLE_5, verify_is456
from src.model_handler import INGREDIENTS

CEMENT_CAP = 550.0
CEMENT_STEP = 10.0
SAFETY_MARGIN = 1.0
FINE_AGG_MIN = 500.0
FINE_AGG_MAX = 1050.0


def rebalance_fine_aggregate(mix: dict, target_total: float, notes: list) -> None:
    """Use fine aggregate as the slack variable to preserve original total mass."""
    non_fine_total = sum(
        mix[ingredient]
        for ingredient in INGREDIENTS
        if ingredient != "fine_aggregate"
    )
    proposed_fine_aggregate = target_total - non_fine_total

    if proposed_fine_aggregate < FINE_AGG_MIN:
        proposed_fine_aggregate = FINE_AGG_MIN
        notes.append(
            f"Fine aggregate reached its lower bound ({FINE_AGG_MIN:.0f} kg/m3); "
            f"the final total may exceed the original {target_total:.0f} kg/m3."
        )
    elif proposed_fine_aggregate > FINE_AGG_MAX:
        proposed_fine_aggregate = FINE_AGG_MAX
        notes.append(
            f"Fine aggregate reached its upper bound ({FINE_AGG_MAX:.0f} kg/m3); "
            f"the final total may be below the original {target_total:.0f} kg/m3."
        )

    mix["fine_aggregate"] = proposed_fine_aggregate


def generate_recommendation(mix: dict, target_grade: str, model_handler) -> dict:
    """Generate specific and verified adjustments for an IS 456 target grade."""
    if target_grade not in IS_456_TABLE_5:
        valid_grades = ", ".join(IS_456_TABLE_5.keys())
        raise ValueError(
            f"Unknown target grade: {target_grade}. Valid grades: {valid_grades}."
        )

    missing = [ingredient for ingredient in INGREDIENTS if ingredient not in mix]
    if missing:
        raise ValueError(f"Missing mix ingredients: {missing}")

    limits = IS_456_TABLE_5[target_grade]
    original_mix = {ingredient: float(mix[ingredient]) for ingredient in INGREDIENTS}
    working_mix = dict(original_mix)
    original_total = sum(original_mix.values())
    actions = []
    notes = []

    # Stage 1: Repair deterministic IS 456 compliance failures first.
    if working_mix["cement"] < limits["min_cement"]:
        cement_delta = limits["min_cement"] - working_mix["cement"]
        working_mix["cement"] = limits["min_cement"]
        rebalance_fine_aggregate(working_mix, original_total, notes)
        actions.append(
            f"Increase cement by +{cement_delta:.1f} kg/m3 "
            f"(to {working_mix['cement']:.1f} kg/m3) to meet the IS 456 "
            f"minimum cement content for {target_grade} "
            f"({limits['min_cement']:.0f} kg/m3)."
        )

    current_wc = (
        working_mix["water"] / working_mix["cement"]
        if working_mix["cement"] > 0
        else float("inf")
    )

    if current_wc > limits["max_wc"]:
        maximum_water = working_mix["cement"] * limits["max_wc"]
        water_delta = working_mix["water"] - maximum_water
        working_mix["water"] = maximum_water
        rebalance_fine_aggregate(working_mix, original_total, notes)
        actions.append(
            f"Reduce water by -{water_delta:.1f} kg/m3 "
            f"(to {working_mix['water']:.1f} kg/m3) to bring W/C from "
            f"{current_wc:.3f} down to the IS 456 maximum of "
            f"{limits['max_wc']:.2f} for {target_grade}."
        )

    # Stage 2: Re-predict after all compliance-specific corrections.
    updated_prediction = model_handler.predict(working_mix)

    if actions and updated_prediction >= limits["min_strength"]:
        actions.append(
            f"After the IS 456 compliance adjustments, the verified predicted "
            f"28-day strength is {updated_prediction:.2f} MPa. This meets the "
            f"{target_grade} minimum of {limits['min_strength']:.0f} MPa, so no "
            f"additional strength adjustment is required."
        )

    # Stage 3: If strength still fails, use cement, the highest-leverage
    # adjustable feature in the model's SHAP analysis. Age is fixed at 28 days.
    if updated_prediction < limits["min_strength"]:
        target_strength = limits["min_strength"] + SAFETY_MARGIN
        cement_before_strength_fix = working_mix["cement"]

        probe_mix = dict(working_mix)
        probe_mix["cement"] = min(
            probe_mix["cement"] + CEMENT_STEP,
            CEMENT_CAP,
        )
        rebalance_fine_aggregate(probe_mix, original_total, [])

        local_slope = 0.0
        cement_probe_delta = probe_mix["cement"] - working_mix["cement"]
        if cement_probe_delta > 0:
            probe_prediction = model_handler.predict(probe_mix)
            local_slope = (probe_prediction - updated_prediction) / cement_probe_delta

        if local_slope > 0:
            estimated_delta = (target_strength - updated_prediction) / local_slope
            estimated_delta = max(0.0, estimated_delta)
            estimated_delta = min(
                estimated_delta,
                CEMENT_CAP - working_mix["cement"],
            )
            estimated_delta = math.ceil(estimated_delta / CEMENT_STEP) * CEMENT_STEP

            if estimated_delta > 0:
                working_mix["cement"] += estimated_delta
                rebalance_fine_aggregate(working_mix, original_total, notes)
                updated_prediction = model_handler.predict(working_mix)

        # Verify the recommendation against the actual model, in bounded steps.
        while (
            updated_prediction < target_strength
            and working_mix["cement"] < CEMENT_CAP
        ):
            working_mix["cement"] = min(
                working_mix["cement"] + CEMENT_STEP,
                CEMENT_CAP,
            )
            rebalance_fine_aggregate(working_mix, original_total, notes)
            updated_prediction = model_handler.predict(working_mix)

        cement_delta = working_mix["cement"] - cement_before_strength_fix
        if updated_prediction >= limits["min_strength"]:
            actions.append(
                f"Increase cement by an additional +{cement_delta:.1f} kg/m3 "
                f"(to {working_mix['cement']:.1f} kg/m3) to correct the "
                f"strength shortfall. The verified updated prediction is "
                f"{updated_prediction:.2f} MPa."
            )
        else:
            actions.append(
                f"The {target_grade} target could not be reached within the "
                f"cement cap of {CEMENT_CAP:.0f} kg/m3. Best verified predicted "
                f"strength: {updated_prediction:.2f} MPa. Redesign the mix or "
                f"select a lower target grade."
            )

    final_compliance = verify_is456(
        target_grade=target_grade,
        predicted_strength=updated_prediction,
        cement=working_mix["cement"],
        water=working_mix["water"],
    )

    return {
        "target_grade": target_grade,
        "original_mix": {key: round(value, 2) for key, value in original_mix.items()},
        "adjusted_mix": {key: round(value, 2) for key, value in working_mix.items()},
        "actions": actions,
        "notes": notes,
        "updated_strength": round(updated_prediction, 2),
        "updated_compliance": final_compliance,
        "target_reached": final_compliance.overall_pass,
    }


if __name__ == "__main__":
    from src.model_handler import ModelHandler

    handler = ModelHandler()
    scenarios = [
        (
            "Weak mix vs M40 (compliance + strength fixes)",
            {
                "cement": 300.0,
                "slag": 0.0,
                "fly_ash": 0.0,
                "water": 190.0,
                "superplasticizer": 0.0,
                "coarse_aggregate": 1000.0,
                "fine_aggregate": 910.0,
            },
            "M40",
        ),
        (
            "Strong but non-compliant vs M30 (SCM-heavy)",
            {
                "cement": 250.0,
                "slag": 200.0,
                "fly_ash": 100.0,
                "water": 170.0,
                "superplasticizer": 6.0,
                "coarse_aggregate": 1000.0,
                "fine_aggregate": 680.0,
            },
            "M30",
        ),
        (
            "Already sufficient vs M25 (no changes expected)",
            {
                "cement": 400.0,
                "slag": 100.0,
                "fly_ash": 0.0,
                "water": 160.0,
                "superplasticizer": 8.0,
                "coarse_aggregate": 1000.0,
                "fine_aggregate": 732.0,
            },
            "M25",
        ),
    ]

    for title, scenario_mix, grade in scenarios:
        initial_prediction = handler.predict(scenario_mix)
        recommendation = generate_recommendation(scenario_mix, grade, handler)

        print(f"\n=== {title} ===")
        print(f"Initial prediction: {initial_prediction:.2f} MPa | Target grade: {grade}")
        for action in recommendation["actions"] or [
            "No changes needed - mix is compliant and strong enough."
        ]:
            print(f"  - {action}")
        for note in recommendation["notes"]:
            print(f"  note: {note}")
        print(
            f"Updated prediction: {recommendation['updated_strength']:.2f} MPa "
            f"| target_reached: {recommendation['target_reached']}"
        )
