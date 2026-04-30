"""
ml_tool.py
----------
Inference helper for the saved car-price pipeline model.
"""

import pickle

import pandas as pd


with open("LinearRegressionModel.pkl", "rb") as f:
    MODEL = pickle.load(f)


REQUIRED_FEATURES = [
    "name",
    "company",
    "year",
    "kms_driven",
    "fuel_type",
]

DEFAULTS = {
    "name": "Corolla",
    "company": "Toyota",
    "year": 2020,
    "kms_driven": 30000,
    "fuel_type": "Petrol",
}


def predict_car_price(features_dict: dict) -> dict:
    try:
        filled = {**DEFAULTS, **features_dict}
        df = pd.DataFrame([filled])[REQUIRED_FEATURES]
        price = float(MODEL.predict(df)[0])
        price = max(price, 10000.0)

        provided = len([key for key in REQUIRED_FEATURES if key in features_dict])
        if provided >= 5:
            confidence = "High (all model features provided)"
        elif provided >= 3:
            confidence = "Medium (some defaults used)"
        else:
            confidence = "Low (many defaults used)"

        return {
            "predicted_price": round(price, 2),
            "confidence_note": confidence,
            "input_used": filled,
        }
    except Exception as e:
        return {
            "predicted_price": None,
            "confidence_note": "Prediction failed",
            "error": str(e),
            "input_used": features_dict,
        }


if __name__ == "__main__":
    result = predict_car_price(
        {
            "name": "City",
            "company": "Honda",
            "year": 2022,
            "kms_driven": 15000,
            "fuel_type": "Petrol",
        }
    )

    print(f"Predicted Price : Rs. {result['predicted_price']}")
    print(f"Confidence      : {result['confidence_note']}")
    if result.get("error"):
        print(f"Error           : {result['error']}")
