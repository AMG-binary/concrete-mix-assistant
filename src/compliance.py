"""
IS 456:2000 compliance layer — Concrete Mix Design Assistant
Concreate Club ML Inductions, IIT Indore

Civil-engineering rules, deliberately DECOUPLED from the ML strength
prediction, exactly as the problem statement mandates:

  * W/C ratio = Water / Cement ONLY. Fly ash and blast furnace slag are
    NOT binder for this calculation — they are excluded from the
    denominator, consistent with IS 456:2000 Table 5 as written.

  * IS 456 Table 5 (values required by the PS — implement exactly):

        Grade | Min strength (MPa) | Min cement (kg/m3) | Max W/C
        ------|--------------------|--------------------|--------
        M20   |        20          |        300         |  0.55
        M25   |        25          |        300         |  0.50
        M30   |        30          |        320         |  0.45
        M35   |        35          |        340         |  0.45
        M40   |        40          |        360         |  0.40

  * Key engineering case the app must catch: a mix can predict ADEQUATE
    strength (e.g. strength boosted by slag/fly ash) while still FAILING
    IS 456 compliance (too little plain cement, or W/C too high). The
    three verdicts — strength, cement content, W/C ratio — are therefore
    reported separately so the user never confuses a strength failure
    with a compliance failure.

Self-test:
    python src/compliance.py
"""

from dataclasses import dataclass, asdict, field

# IS 456:2000 Table 5 — problem-statement values, implemented exactly
IS_456_TABLE_5 = {
    "M20": {"min_strength": 20.0, "min_cement": 300.0, "max_wc": 0.55},
    "M25": {"min_strength": 25.0, "min_cement": 300.0, "max_wc": 0.50},
    "M30": {"min_strength": 30.0, "min_cement": 320.0, "max_wc": 0.45},
    "M35": {"min_strength": 35.0, "min_cement": 340.0, "max_wc": 0.45},
    "M40": {"min_strength": 40.0, "min_cement": 360.0, "max_wc": 0.40},
}
GRADES = list(IS_456_TABLE_5)


@dataclass
class ComplianceResult:
    """Separate verdicts so the UI can display strength vs compliance failures distinctly."""
    target_grade: str
    predicted_grade: str
    predicted_strength: float
    strength_pass: bool
    cement_pass: bool
    wc_pass: bool
    actual_wc: float
    min_cement_required: float
    max_wc_allowed: float
    overall_pass: bool = field(init=False)

    def __post_init__(self) -> None:
        self.overall_pass = self.strength_pass and self.cement_pass and self.wc_pass

    def to_dict(self) -> dict:
        return asdict(self)


def classify_grade(strength: float) -> str:
    """Map a predicted strength (MPa) to its IS 456 grade band."""
    if strength >= 40.0:
        return "M40+"
    if strength >= 35.0:
        return "M35"
    if strength >= 30.0:
        return "M30"
    if strength >= 25.0:
        return "M25"
    if strength >= 20.0:
        return "M20"
    return "Below Grade (<M20)"


def wc_ratio(water: float, cement: float) -> float:
    """Water-cement ratio: water / cement only. Guards division by zero."""
    if cement <= 0:
        return float("inf")
    return water / cement


def verify_is456(
    target_grade: str,
    predicted_strength: float,
    cement: float,
    water: float,
) -> ComplianceResult:
    """Check a mix against IS 456 Table 5 for the user's target grade."""
    if target_grade not in IS_456_TABLE_5:
        raise ValueError(f"Unknown grade {target_grade!r}. Valid grades: {GRADES}")
    limits = IS_456_TABLE_5[target_grade]
    ratio = wc_ratio(water, cement)

    return ComplianceResult(
        target_grade=target_grade,
        predicted_grade=classify_grade(predicted_strength),
        predicted_strength=round(predicted_strength, 2),
        strength_pass=predicted_strength >= limits["min_strength"],
        cement_pass=cement >= limits["min_cement"],
        wc_pass=ratio <= limits["max_wc"],
        actual_wc=round(ratio, 3),
        min_cement_required=limits["min_cement"],
        max_wc_allowed=limits["max_wc"],
    )


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    # 1) Fully compliant M25 mix
    r = verify_is456("M25", 26.5, 310.0, 150.0)          # wc = 0.484 <= 0.50
    assert r.strength_pass and r.cement_pass and r.wc_pass and r.overall_pass

    # 2) Strong mix that FAILS W/C compliance — the decoupling case.
    #    Strength 27 MPa passes M25, but wc = 180/300 = 0.60 > 0.50 fails.
    r = verify_is456("M25", 27.0, 300.0, 180.0)
    assert r.strength_pass and not r.wc_pass and not r.overall_pass

    # 3) Strength shortfall with compliant cement and W/C
    r = verify_is456("M30", 28.0, 330.0, 145.0)          # wc = 0.439 <= 0.45
    assert not r.strength_pass and r.cement_pass and r.wc_pass

    # 4) Grade classification boundaries
    assert classify_grade(40.0) == "M40+"
    assert classify_grade(39.9) == "M35"
    assert classify_grade(25.0) == "M25"
    assert classify_grade(19.9) == "Below Grade (<M20)"

    # 5) Zero-cement guard (invalid input must not crash with ZeroDivisionError)
    assert wc_ratio(180.0, 0.0) == float("inf")

    print("All IS 456:2000 self-tests passed.")
