"""Production-style inference utilities for loan propensity prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURE_NAMES_PATH, MODEL_PATH


def load_artifacts(
    model_path: str | Path = MODEL_PATH,
    feature_names_path: str | Path = FEATURE_NAMES_PATH,
):
    """Load trained model and feature names from disk."""
    model_path = Path(model_path)
    feature_names_path = Path(feature_names_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names artifact not found: {feature_names_path}")

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)

    return model, feature_names


def prepare_customer_input(customer_data: dict | pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Validate and order customer input for model prediction."""
    if isinstance(customer_data, dict):
        customer_df = pd.DataFrame([customer_data])
    elif isinstance(customer_data, pd.DataFrame):
        customer_df = customer_data.copy()
    else:
        raise TypeError("customer_data must be a dictionary or pandas DataFrame.")

    missing_features = [feature for feature in feature_names if feature not in customer_df.columns]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    return customer_df[feature_names].copy()


def predict_customer(
    customer_data: dict | pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    feature_names_path: str | Path = FEATURE_NAMES_PATH,
) -> dict:
    """Predict loan propensity for one or more customer records."""
    model, feature_names = load_artifacts(model_path, feature_names_path)
    customer_df = prepare_customer_input(customer_data, feature_names)

    predictions = model.predict(customer_df)
    probabilities = model.predict_proba(customer_df)[:, 1]

    if len(predictions) == 1:
        return {
            "prediction": int(predictions[0]),
            "loan_probability": round(float(probabilities[0]), 4),
            "risk_label": "High Propensity" if predictions[0] == 1 else "Low Propensity",
        }

    return {
        "predictions": predictions.astype(int).tolist(),
        "loan_probabilities": [round(float(prob), 4) for prob in probabilities],
    }


if __name__ == "__main__":
    sample_customer = {
        "Age": 45,
        "CustomerSince": 20,
        "HighestSpend": 67,
        "MonthlyAverageSpend": 2.5,
        "Mortgage": 0,
        "Level": 2,
        "HiddenScore": 3,
        "Security": 0,
        "FixedDepositAccount": 1,
        "InternetBanking": 1,
        "CreditCard": 1,
    }

    result = predict_customer(sample_customer)
    print(result)
