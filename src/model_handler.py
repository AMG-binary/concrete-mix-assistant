"""
Model inference layer — Concrete Mix Design Assistant
Loads the Phase 1 artifacts and exposes a single validated predict() call.

Contract (per the problem statement):
  * The app accepts 7 mix ingredients only; age is ALWAYS injected as 28 days.
  * Inputs are validated (finite, non-negative) so bad UI input fails with a
    clear ValueError instead of a silent garbage prediction. The Streamlit
    app catches these and shows clean error messages.
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/model.joblib")
FIXED_AGE = 28.0  # PS: all app predictions are at the fixed 28-day point

INGREDIENTS = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate",
]


class ModelHandler:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run train_model.py first (Phase 1)."
            )
        payload = joblib.load(path)
        self.model = payload["model"]
        self.features = payload["features"]  # exact training column-order contract
        self.model_type = payload.get("model_type", "unknown")

    def predict(self, mix: dict) -> float:
        """Predict 28-day compressive strength (MPa) for a 7-ingredient mix dict."""
        row = {}
        for name in self.features:
            if name == "age":
                row["age"] = FIXED_AGE
                continue
            if name not in mix:
                raise ValueError(f"Missing ingredient: {name!r}")
            raw = mix[name]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid value for {name!r}: {raw!r}")
            if not math.isfinite(value):
                raise ValueError(f"Invalid value for {name!r}: {raw!r}")
            if value < 0:
                raise ValueError(f"{name!r} must be non-negative, got {value}")
            row[name] = value
        X = pd.DataFrame([row])[self.features]  # enforce training column order
        return float(self.model.predict(X)[0])


if __name__ == "__main__":
    handler = ModelHandler()
    demo = {
        "cement": 350.0, "slag": 100.0, "fly_ash": 0.0, "water": 190.0,
        "superplasticizer": 8.0, "coarse_aggregate": 1000.0, "fine_aggregate": 752.0,
    }
    print(f"Model: {handler.model_type} | demo prediction: {handler.predict(demo):.2f} MPa")
