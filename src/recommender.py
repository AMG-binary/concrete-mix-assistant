"""
Recommendation engine — Concrete Mix Design Assistant

Workflow:
1. Fix IS 456 compliance failures first:
   - Cement below the grade-specific minimum → increase cement.
   - W/C above the grade-specific maximum → reduce water.
2. Re-run the ML model after those corrections.
3. If strength is still below the target grade, increase cement in bounded
   10 kg/m³ increments and re-run the model after every adjustment.
4. Keep total mix mass near the original value by using fine aggregate as
   the balancing ingredient.
5. Return exact deltas, the adjusted mix, updated prediction, and verified
   IS 456 compliance.

Run from the repository root:
    python src/recommender.py
"""

from __future__ import annotations

import math
from pathlib import Path

from src.compliance import IS_456_TABLE_5, verify_is456
from src.model_handler import INGREDIENTS


CEMENT_CAP = 550.0
CEMENT_STEP = 10.0
SAFETY_MARGIN = 1.0

FINE_AGG_MIN = 500.0
FINE_AGG_MAX = 1050.0


def rebalance_fine_aggregate(mix: dict, target_total: float, notes: list) -> None:
    """
    Keeps the modified mix close to its initial total mass.

    Fine aggregate is used as the balancing variable because it is an
    adjustable constituent and is not directly part of the IS 456 W/C check.
    """
    non_fine_total = sum(
        mix[ingredient]
        for ingredient in INGREDIENTS
        if ingredient != "fine_aggregate"
    )

    proposed_fine_aggregate = target_total - non_fine_total

    if proposed_fine_aggregate < FINE_AGG_MIN:
        notes.append(
            f"Fine aggregate reached its lower bound of "
            f"{FINE_AGG_MIN:.0f} kg/m³. The final total mix weight may be "
            f"above the original {target_total:.0f} kg/m³."
        )
        proposed_fine_aggregate = FINE_AGG_MIN

    elif proposed_fine_aggregate > FINE_AGG_MAX:
        notes.append(
            f"Fine aggregate reached its upper bound of "
            f"{FINE_AGG_MAX:.0f} kg/m³. The final total mix weight may be "
            f"below the original {target_total:.0f} kg/m³."
        )
        proposed_fine_aggregate = FINE_AGG_MAX

    mix["fine_aggregate"] = proposed_fine_aggregate


def generate_recommendation(
    mix: dict,
    target_grade: str,
    model_handler,
) -> dict:
    """
    Produce a specific, re-predicted recommendation for the selected target.

    Parameters
    ----------
    mix:
        Dictionary with all seven adjustable ingredient quantities in kg/m³.
    target_grade:
        One of M20, M25, M30, M35, M40.
    model_handler:
        A loaded ModelHandler instance with a predict(mix) method.

    Returns
    -------
    dict:
        Original and adjusted mix, list of actions, prediction, compliance,
        notes, and an overall target_reached boolean.
    """
    if target_grade not in IS_456_TABLE_5:
        valid_grades = ", ".join(IS_456_TABLE_5.keys())
        raise ValueError(
            f"Unknown target grade '{target_grade}'. "
            f"Choose one of: {valid_grades}."
        )

    limits = IS_456_TABLE_5[target_grade]

    working_mix = {
        ingredient: float(mix[ingredient])
        for ingredient in INGREDIENTS
    }

    original_mix = dict(working_mix)
    original_total = sum(working_mix.values())

    actions = []
    notes = []

    # ---------------------------------------------------------------
    # Stage 1: Cement-content compliance.
    # ---------------------------------------------------------------
    if working_mix["cement"] < limits["min_cement"]:
        cement_delta = limits["min_cement"] - working_mix["cement"]

        working_mix["cement"] = limits["min_cement"]

        rebalance_fine_aggregate(
            mix=working_mix,
            target_total=original_total,
            notes=notes,
        )

        actions.append(
            f"Increase cement by +{cement_delta:.1f} kg/m³ "
            f"(from {original_mix['cement']:.1f} to "
            f"{working_mix['cement']:.1f} kg/m³) to satisfy the IS 456 "
            f"minimum cement content for {target_grade} "
            f"({limits['min_cement']:.0f} kg/m³)."
        )

    # ---------------------------------------------------------------
    # Stage 2: Water-cement-ratio compliance.
    # W/C is water ÷ cement only, per the problem statement.
    # ---------------------------------------------------------------
    if working_mix["cement"] <= 0:
        current_wc = float("inf")
    else:
        current_wc = working_mix["water"] / working_mix["cement"]

    if current_wc > limits["max_wc"]:
        maximum_water = (
            working_mix["cement"] * limits["max_wc"]
        )

        water_delta = working_mix["water"] - maximum_water

        working_mix["water"] = maximum_water

        rebalance_fine_aggregate(
            mix=working_mix,
            target_total=original_total,
            notes=notes,
        )

        actions.append(
            f"Reduce water by -{water_delta:.1f} kg/
