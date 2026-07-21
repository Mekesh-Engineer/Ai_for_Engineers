#!/usr/bin/env python3
"""
predict_house_value.py — Interactive House Price Prediction
============================================================
Loads the pre-trained Multiple Linear Regression model and scaler,
accepts user input for 8 housing features, and predicts the
median house value in dollars.

Usage:
    python predict_house_value.py
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Feature configuration ──────────────────────────────────────────────────
# The user's pre-trained model expects raw features in this exact order:
RAW_FEATURE_NAMES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "multiple_linear_regression_model.joblib")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "raw_feature_scaler.joblib")


# ── Build scaler from training data ────────────────────────────────────────
def build_raw_feature_scaler():
    """
    The California Housing dataset from sklearn provides averaged features
    (AveRooms, AveOccup, etc.). The user's model was trained on raw features
    (total_rooms, total_bedrooms, households, population).

    This function reconstructs the raw feature DataFrame from the sklearn
    dataset and fits a StandardScaler on it so new input can be scaled
    consistently.
    """
    housing = fetch_california_housing()
    X = housing.data  # columns: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    # sklearn feature order: 0-MedInc, 1-HouseAge, 2-AveRooms, 3-AveBedrms,
    #                        4-Population, 5-AveOccup, 6-Latitude, 7-Longitude

    # Reconstruct raw features from the averaged ones:
    #   AveRooms   = total_rooms / households       → total_rooms   = AveRooms * households
    #   AveBedrms  = total_bedrooms / households     → total_bedrooms = AveBedrms * households
    #   AveOccup   = population / households         → households     = population / AveOccup

    med_inc    = X[:, 0]
    house_age  = X[:, 1]
    ave_rooms  = X[:, 2]
    ave_bedrms = X[:, 3]
    population = X[:, 4]
    ave_occup  = X[:, 5]
    latitude   = X[:, 6]
    longitude  = X[:, 7]

    households      = population / ave_occup
    total_rooms     = ave_rooms * households
    total_bedrooms  = ave_bedrms * households

    # Build raw feature matrix in the order the model expects
    raw_X = np.column_stack([
        longitude,
        latitude,
        house_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        med_inc,
    ])

    scaler = StandardScaler()
    scaler.fit(raw_X)

    # Save for reuse
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print("[INFO] Raw-feature scaler fitted and saved.")
    return scaler


def get_scaler():
    """Load or create the scaler for raw features."""
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    else:
        return build_raw_feature_scaler()


# ── Interactive prediction ─────────────────────────────────────────────────
def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    scaler = get_scaler()

    # Collect user input
    print("\nEnter values for the following features to get a house value prediction:")
    values = []
    for feature in RAW_FEATURE_NAMES:
        while True:
            try:
                val = float(input(f"Enter value for '{feature}': "))
                values.append(val)
                break
            except ValueError:
                print("  Please enter a valid number.")

    # Build input DataFrame
    input_df = pd.DataFrame([values], columns=RAW_FEATURE_NAMES)
    print("\nReceived input data:")
    print(input_df.to_string())

    # Scale
    input_scaled = scaler.transform(input_df.values)
    scaled_df = pd.DataFrame(input_scaled, columns=RAW_FEATURE_NAMES)
    print("\n\nScaled input data:")
    print(scaled_df.to_string())

    # Predict
    prediction = model.predict(input_scaled)[0]
    print(f"\n\nPredicted Median House Value: ${prediction:,.2f}")


if __name__ == "__main__":
    main()
