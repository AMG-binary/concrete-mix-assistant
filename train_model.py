"""
Phase 1 — Model Training & Artifact Export
Concrete Mix Design Assistant | Concreate Club ML Inductions, IIT Indore

Pipeline (matches the workflow blueprint):
  1. Load the UCI Concrete Compressive Strength dataset (.xls) and
     normalise the messy original column names to canonical snake_case.
  2. Sanity checks + quick EDA (missing values, duplicates, correlations).
  3. 80/20 train-test split with random_state=42 (seed is stated in the
     app UI, as required by the problem statement).
  4. Model comparison — RandomForest vs XGBoost — selected purely by
     5-fold cross-validated R2 on the TRAINING set. The test set is
     touched once, at the end, for final reporting (no leakage).
  5. Export deployment artifacts:
        models/model.joblib              (fitted model + feature contract)
        models/metrics.json              (R2 / RMSE + reproducibility info)
        models/feature_importance.json   (drives the recommender engine)

Usage (Google Colab or local):
    pip install -r requirements.txt
    python train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBRegressor

# ----------------------------------------------------------- constants
RANDOM_STATE = 42        # reproducibility seed — MUST be stated in the app UI
TEST_SIZE = 0.20         # PS requires a minimum 80/20 split
DATA_CANDIDATES = [Path("data/Concrete_Data.xls"), Path("Concrete_Data.xls")]
MODEL_DIR = Path("models")

FEATURES = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age",
]
TARGET = "strength"


# ----------------------------------------------------------- data loading
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map the verbose UCI column names onto canonical snake_case names."""
    mapping = {}
    for col in df.columns:
        c = col.lower()
        if "strength" in c or "mpa" in c:
            mapping[col] = "strength"
        elif "cement" in c:
            mapping[col] = "cement"
        elif "slag" in c:
            mapping[col] = "slag"
        elif "fly" in c:
            mapping[col] = "fly_ash"
        elif "water" in c:
            mapping[col] = "water"
        elif "superplastic" in c:
            mapping[col] = "superplasticizer"
        elif "coarse" in c:
            mapping[col] = "coarse_aggregate"
        elif "fine" in c:
            mapping[col] = "fine_aggregate"
        elif "age" in c:
            mapping[col] = "age"
    return df.rename(columns=mapping)


def load_data() -> pd.DataFrame:
    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Could not find Concrete_Data.xls in {[str(p) for p in DATA_CANDIDATES]}. "
            "Upload the UCI dataset file first."
        )
    df = pd.read_excel(path)  # legacy .xls -> requires the xlrd engine
    df = rename_columns(df)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Column renaming failed — missing: {missing}")
    print(f"[data] loaded {len(df)} rows x {df.shape[1]} cols from {path}")
    return df


# ----------------------------------------------------------- EDA
def run_eda(df: pd.DataFrame) -> None:
    print("\n=== EDA ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {int(df.isna().sum().sum())} (readme says 0 — verify)")
    dups = int(df.duplicated().sum())
    print(f"Duplicate rows: {dups} (kept — repeated lab measurements; "
          f"dropping them would also change the 1030-row count the PS references)")
    print("\nDescriptive statistics:")
    print(df.describe().T.round(2))

    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    print("\nCorrelation with compressive strength (ascending):")
    for name, val in corr.items():
        print(f"  {name:<18} {val:+.3f}")

    n28 = int((df["age"] == 28).sum())
    print(f"\nRows measured at age = 28 days: {n28} "
          "(the app will predict at this fixed age)")


# ----------------------------------------------------------- training
def make_grids():
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={
            "n_estimators": [400, 800],
            "max_depth": [None, 12],
            "min_samples_leaf": [1, 2],
        },
        cv=cv, scoring="r2", n_jobs=-1,
    )

    xgb_search = GridSearchCV(
        XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
            objective="reg:squarederror",
        ),
        param_grid={
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.08],
            "max_depth": [3, 5],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [1.0],
        },
        cv=cv, scoring="r2", n_jobs=-1,
    )
    return {"RandomForest": rf_search, "XGBoost": xgb_search}


def evaluate_on_test(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "test_r2": round(float(r2_score(y_test, pred)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
    }


def shap_analysis(model, X_test) -> None:
    """Optional directional attribution (used for the video / recommender story)."""
    try:
        import shap
    except ImportError:
        print("\n[shap] not installed — skipping (pip install shap to enable)")
        return
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_test)
    mean_abs = np.abs(values).mean(axis=0)
    ranking = sorted(zip(FEATURES, mean_abs.round(4)), key=lambda t: -t[1])
    out = MODEL_DIR / "shap_importance.json"
    out.write_text(json.dumps({f: float(v) for f, v in ranking}, indent=2))
    print(f"\n[shap] mean |SHAP| ranking saved -> {out}")
    for f, v in ranking:
        print(f"  {f:<18} {v:.4f}")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()
    run_eda(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n=== Split (seed={RANDOM_STATE}) ===")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("\n=== Model selection (5-fold CV on TRAIN only) ===")
    searches = make_grids()
    results, fitted = {}, {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        results[name] = {
            "cv_r2_mean": round(float(search.best_score_), 4),
            "best_params": search.best_params_,
        }
        fitted[name] = search.best_estimator_
        print(f"{name:<14} CV R2 = {search.best_score_:.4f}  {search.best_params_}")

    print("\n=== Final test-set reporting (used once) ===")
    for name, model in fitted.items():
        results[name].update(evaluate_on_test(model, X_test, y_test))
        print(f"{name:<14} test R2 = {results[name]['test_r2']:.4f} | "
              f"RMSE = {results[name]['test_rmse']:.3f} MPa")

    # Winner chosen by cross-validated R2, never by test-set peeking
    winner_name = max(results, key=lambda k: results[k]["cv_r2_mean"])
    winner = fitted[winner_name]
    print(f"\nDeploying: {winner_name} (best CV R2)")

    # ---- artifact 1: model + feature contract --------------------------
    joblib.dump(
        {"model": winner, "features": FEATURES, "model_type": winner_name},
        MODEL_DIR / "model.joblib",
    )

    # ---- artifact 2: metrics (surfaced in the Streamlit UI) ------------
    metrics = {
        "model_type": winner_name,
        "model_params": winner.get_params(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_r2_mean": results[winner_name]["cv_r2_mean"],
        "test_r2": results[winner_name]["test_r2"],
        "test_rmse": results[winner_name]["test_rmse"],
        "comparison": results,
        "age_policy": "Trained on all ages; app predicts at fixed age=28 days.",
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # ---- artifact 3: feature importance (recommender input) ------------
    # NOTE: cast numpy float32 -> native Python float before json.dumps
    importance = {f: round(float(v), 4) for f, v in zip(FEATURES, winner.feature_importances_)}
    importance = dict(sorted(importance.items(), key=lambda t: -t[1]))
    (MODEL_DIR / "feature_importance.json").write_text(json.dumps(importance, indent=2))
    print(f"\nFeature importance ({winner_name}):")
    for f, v in importance.items():
        print(f"  {f:<18} {v:.4f}")

    # ---- demo prediction at the app's fixed operating point ------------
    demo_mix = pd.DataFrame(
        [{
            "cement": 350.0, "slag": 100.0, "fly_ash": 0.0, "water": 190.0,
            "superplasticizer": 8.0, "coarse_aggregate": 1000.0,
            "fine_aggregate": 752.0, "age": 28,   # sums to ~2400 kg/m3
        }]
    )[FEATURES]
    demo_pred = float(winner.predict(demo_mix)[0])
    print(f"\nDemo mix (2400 kg/m3, age 28) predicted strength: {demo_pred:.2f} MPa")

    shap_analysis(winner, X_test)

    print(f"\nArtifacts written to {MODEL_DIR}/: model.joblib, metrics.json, "
          f"feature_importance.json")
    print("Next: wire these into src/compliance.py + the Streamlit app.")


if __name__ == "__main__":
    main()

