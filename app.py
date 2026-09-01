"""
Concrete Mix Design Assistant — Streamlit application
Concreate Club ML Inductions, IIT Indore

Flow (per the problem statement):
  1. The user enters 7 mix ingredients (kg/m3) and a target IS 456 grade.
     Age is FIXED at 28 days — the app never asks for it.
  2. ML layer: XGBoost predicts the 28-day compressive strength (MPa).
  3. IS 456 layer: Table 5 checks (min cement, max W/C) displayed
     SEPARATELY from the strength prediction, so a strength failure is
     never confused with a compliance failure.
  4. If the mix falls short, the recommender proposes specific deltas and
     the app re-runs the prediction to VERIFY the fix.

Run:  streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.compliance import IS_456_TABLE_5, classify_grade, verify_is456
from src.model_handler import INGREDIENTS, ModelHandler
from src.recommender import generate_recommendation

st.set_page_config(
    page_title="Concrete Mix Design Assistant", page_icon="🏗️", layout="wide"
)

LABELS = {
    "cement": "Cement (kg/m³)",
    "slag": "Blast Furnace Slag (kg/m³)",
    "fly_ash": "Fly Ash (kg/m³)",
    "water": "Water (kg/m³)",
    "superplasticizer": "Superplasticizer (kg/m³)",
    "coarse_aggregate": "Coarse Aggregate (kg/m³)",
    "fine_aggregate": "Fine Aggregate (kg/m³)",
}
STEPS = {
    "cement": 5.0, "slag": 5.0, "fly_ash": 5.0, "water": 1.0,
    "superplasticizer": 0.5, "coarse_aggregate": 5.0, "fine_aggregate": 5.0,
}
# Generous caps (~ dataset max + margin) to block absurd entries gracefully
MAX_VALUES = {
    "cement": 600.0, "slag": 400.0, "fly_ash": 300.0, "water": 300.0,
    "superplasticizer": 40.0, "coarse_aggregate": 1300.0, "fine_aggregate": 1100.0,
}
DEFAULT_MIX = {
    "cement": 350.0, "slag": 100.0, "fly_ash": 0.0, "water": 190.0,
    "superplasticizer": 8.0, "coarse_aggregate": 1000.0, "fine_aggregate": 752.0,
}  # sums to 2400 kg/m3
EXAMPLES = {
    "Custom mix": None,
    "Example: weak mix (needs fixes)": {
        "cement": 300.0, "slag": 0.0, "fly_ash": 0.0, "water": 190.0,
        "superplasticizer": 0.0, "coarse_aggregate": 1000.0, "fine_aggregate": 910.0,
    },
    "Example: strong but non-compliant (SCM-heavy)": {
        "cement": 250.0, "slag": 200.0, "fly_ash": 100.0, "water": 170.0,
        "superplasticizer": 6.0, "coarse_aggregate": 1000.0, "fine_aggregate": 680.0,
    },
    "Example: compliant M25 mix": {
        "cement": 400.0, "slag": 100.0, "fly_ash": 0.0, "water": 160.0,
        "superplasticizer": 8.0, "coarse_aggregate": 1000.0, "fine_aggregate": 732.0,
    },
}


@st.cache_resource(show_spinner="Loading trained model...")
def load_model() -> ModelHandler:
    """Cached so the joblib artifact is read once, not on every interaction."""
    return ModelHandler()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(Path("models/metrics.json").read_text())


def render_sidebar() -> tuple[dict, str, bool]:
    with st.sidebar:
        st.header("Mix Inputs")

        example = st.selectbox("Quick-load an example", list(EXAMPLES))
        if EXAMPLES[example] and st.session_state.get("_loaded_example") != example:
            for key, value in EXAMPLES[example].items():
                st.session_state[key] = value
            st.session_state["_loaded_example"] = example

        mix = {}
        for name in INGREDIENTS:
            mix[name] = st.number_input(
                LABELS[name],
                min_value=0.0,
                max_value=MAX_VALUES[name],
                step=STEPS[name],
                value=DEFAULT_MIX[name],
                key=name,
                format="%.1f",
            )

        total = sum(mix.values())
        st.markdown(f"**Total mix weight:** {total:.0f} kg/m³")
        if 2300.0 <= total <= 2500.0:
            st.success("Within the typical range (~2400 kg/m³)")
        else:
            st.warning("Typical concrete sums to ~2400 kg/m³ — adjust the aggregates to get closer.")

        st.info("Curing age is fixed at **28 days** (standard test age).")
        target_grade = st.selectbox("Target IS 456 grade", list(IS_456_TABLE_5), index=1)
        evaluate = st.button("Evaluate Mix", type="primary", use_container_width=True)
    return mix, target_grade, evaluate


def render_prediction(strength: float, target_grade: str) -> None:
    st.subheader("1 · ML Strength Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted 28-day strength", f"{strength:.2f} MPa")
    col2.metric("Classified grade", classify_grade(strength))
    col3.metric("Your target grade", target_grade)


def render_compliance(comp, mix: dict, target_grade: str) -> None:
    st.subheader("2 · IS 456:2000 Compliance Check")
    st.caption(
        "Checked independently of the ML prediction — a mix can be strong enough "
        "and still fail IS 456 durability requirements."
    )
    checks = [
        ("Strength", f"predicted {comp.predicted_strength:.2f} MPa vs minimum "
                     f"{IS_456_TABLE_5[target_grade]['min_strength']:.0f} MPa", comp.strength_pass),
        ("Cement content", f"{mix['cement']:.0f} kg/m³ vs minimum "
                           f"{comp.min_cement_required:.0f} kg/m³", comp.cement_pass),
        ("Water/cement ratio", f"{comp.actual_wc:.3f} vs maximum "
                               f"{comp.max_wc_allowed:.2f} (water ÷ cement only)", comp.wc_pass),
    ]
    for label, detail, passed in checks:
        if passed:
            st.success(f"✅ **{label}** — {detail}")
        else:
            st.error(f"❌ **{label}** — {detail}")


def render_recommendation(mix: dict, target_grade: str, model: ModelHandler) -> None:
    st.subheader("3 · Recommended Adjustments")
    st.caption(
        "Primary lever chosen by model feature importance (SHAP); every fix is "
        "verified by re-running the prediction with the adjusted values."
    )
    rec = generate_recommendation(mix, target_grade, model)

    for action in rec["actions"]:
        st.markdown(f"- {action}")
    for note in rec["notes"]:
        st.caption(f"⚠️ {note}")

    rows = []
    for name in INGREDIENTS:
        before, after = mix[name], rec["adjusted_mix"][name]
        delta = after - before
        rows.append({
            "Ingredient": LABELS[name],
            "Original (kg/m³)": f"{before:.1f}",
            "Adjusted (kg/m³)": f"{after:.1f}",
            "Change": f"{delta:+.1f}" if abs(delta) > 1e-6 else "—",
        })
    st.table(pd.DataFrame(rows))

    final = rec["updated_compliance"]
    if final.overall_pass:
        st.success(
            f"✅ **Verified:** the adjusted mix meets all {target_grade} requirements "
            f"— predicted strength {rec['updated_strength']:.2f} MPa, "
            f"W/C {final.actual_wc:.3f}, cement {rec['adjusted_mix']['cement']:.0f} kg/m³."
        )
    else:
        st.warning(
            f"⚠️ Best achievable within adjustment limits: "
            f"{rec['updated_strength']:.2f} MPa — see the notes above."
        )


def main() -> None:
    st.title("🏗️ Concrete Mix Design Assistant")
    st.caption(
        "Enter a concrete mix to get its predicted 28-day compressive strength, an "
        "IS 456:2000 grade classification and compliance check, and specific "
        "adjustment recommendations if it falls short of your target grade."
    )

    try:
        model = load_model()
        metrics = load_metrics()
    except FileNotFoundError as exc:
        st.error(f"Model artifacts not found: {exc} — run `python train_model.py` first.")
        st.stop()

    st.caption(
        f"Model: **{metrics['model_type']}** · Test R² **{metrics['test_r2']}** · "
        f"RMSE **{metrics['test_rmse']} MPa** · "
        f"{int(metrics['n_train'])}/{int(metrics['n_test'])} train/test split · "
        f"random seed **{metrics['random_state']}** (reproducible)"
    )

    mix, target_grade, evaluate = render_sidebar()

    if not evaluate:
        st.info("Enter your mix on the left, then press **Evaluate Mix**.")
        return

    try:
        strength = model.predict(mix)
        comp = verify_is456(target_grade, strength, mix["cement"], mix["water"])

        render_prediction(strength, target_grade)
        render_compliance(comp, mix, target_grade)

        if comp.overall_pass:
            st.success(f"🎉 This mix meets all requirements for **{target_grade}** — no changes needed.")
        else:
            render_recommendation(mix, target_grade, model)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
    except Exception as exc:  # last-resort guard: the app must never crash
        st.error(f"Something went wrong while evaluating the mix: {exc}")

    st.divider()
    st.caption(
        "Dataset: UCI Concrete Compressive Strength (Yeh, 1998) — 1030 samples, "
        "8 features. Trained with scikit-learn / XGBoost; artifacts in `models/`."
    )


if __name__ == "__main__":
    main()

