# -*- coding: utf-8 -*-
"""
data_loader.py
--------------
DataLoaderModule for Experiment 5: Sequential Data Modeling using Vanilla RNN.
Handles loading configuration, generating/loading hourly household/industrial
energy consumption time-series data (2,160 hourly records), and printing data summaries.
"""

import os
import json
import numpy as np
import pandas as pd


def load_config(config_path: str = "config/hyperparameters.json") -> dict:
    """Load hyperparameters JSON configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_energy_consumption_dataset(num_samples: int = 2160, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic hourly household energy consumption time-series dataset (kWh)
    featuring 24-hour diurnal patterns, morning/evening load peaks, 168-hour weekly
    seasonality (weekday vs. weekend shifts), temperature-dependent HVAC load,
    high-demand appliance spikes, and random observation noise.

    Parameters
    ----------
    num_samples : int
        Number of hourly timestamps to generate (default: 2,160 hours = 90 days).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Datetime', 'Energy_Consumption_kWh', 'Temperature_C',
        'Humidity_Pct', 'Hour_of_Day', 'Day_of_Week', 'Is_Weekend'].
    """
    np.random.seed(seed)
    datetime_range = pd.date_range(start="2024-01-01 00:00:00", periods=num_samples, freq="h")
    t = np.arange(num_samples)
    hours = datetime_range.hour.values
    dayofweek = datetime_range.dayofweek.values
    is_weekend = (dayofweek >= 5).astype(int)

    # 1. Base Standby Baseload (refrigeration, router, standby electronics)
    base_load = 1.20  # kWh

    # 2. 24-Hour Diurnal Cycle (Morning peak 07:00-09:00, Evening peak 18:00-22:00, Night trough 01:00-05:00)
    morning_peak = 1.40 * np.exp(-((hours - 8.0) ** 2) / 3.5)
    evening_peak = 2.20 * np.exp(-((hours - 20.0) ** 2) / 7.0)
    midday_activity = 0.50 * np.sin(np.pi * (np.clip(hours, 9, 17) - 9) / 8.0)
    diurnal_pattern = morning_peak + evening_peak + midday_activity

    # 3. Weekly 168-Hour Seasonality & Weekend Modulation
    # Weekend load is distributed more evenly across the day with later morning rise
    weekend_morning_shift = 0.60 * is_weekend * np.exp(-((hours - 11.0) ** 2) / 8.0)
    weekday_office_suppression = -0.30 * (1 - is_weekend) * ((hours >= 10) & (hours <= 16)).astype(float)
    weekly_modulation = weekend_morning_shift + weekday_office_suppression

    # 4. Ambient Temperature (°C) & HVAC Heating/Cooling Response
    # Winter-to-spring progression (5°C to 18°C) + daily temperature swing
    seasonal_temp = 7.0 + 8.0 * (t / num_samples)
    daily_temp_swing = 4.5 * np.sin(2 * np.pi * (hours - 14.0) / 24.0)
    temperature_c = seasonal_temp + daily_temp_swing + np.random.normal(0, 0.6, size=num_samples)
    temperature_c = np.round(temperature_c, 2)

    # Heating demand kicks in when temperature falls below 12°C
    heating_demand = 0.08 * np.maximum(0.0, 14.0 - temperature_c)

    # 5. Humidity (%)
    humidity_pct = np.clip(65.0 - 15.0 * np.sin(2 * np.pi * hours / 24.0) + np.random.normal(0, 3.0, size=num_samples), 35.0, 95.0)
    humidity_pct = np.round(humidity_pct, 1)

    # 6. Intermittent High-Draw Appliance Events (e.g., EV charging, washing machine, oven)
    appliance_events = (np.random.rand(num_samples) < 0.04).astype(int)
    appliance_draw = appliance_events * np.random.uniform(1.2, 2.8, size=num_samples)

    # 7. Stochastic Observation Noise
    noise = np.random.normal(scale=0.08, size=num_samples)

    # Composite Hourly Energy Consumption (kWh)
    energy_kwh = base_load + diurnal_pattern + weekly_modulation + heating_demand + appliance_draw + noise
    energy_kwh = np.clip(energy_kwh, 0.40, None)  # Ensure minimum physical power threshold

    df = pd.DataFrame({
        "Datetime": datetime_range,
        "Energy_Consumption_kWh": np.round(energy_kwh, 3),
        "Temperature_C": temperature_c,
        "Humidity_Pct": humidity_pct,
        "Hour_of_Day": hours,
        "Day_of_Week": dayofweek,
        "Is_Weekend": is_weekend
    })

    return df


def get_or_create_energy_data(config: dict) -> pd.DataFrame:
    """
    Load raw CSV if present, otherwise generate and save `household_energy_consumption.csv`.

    Parameters
    ----------
    config : dict
        Hyperparameters configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Loaded energy consumption DataFrame.
    """
    raw_path = config["dataset"]["raw_csv_path"]
    num_samples = config["dataset"].get("num_samples", 2160)
    seed = config["training"].get("random_seed", 42)

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"[DataLoader] Dataset file not found at '{raw_path}'. Generating synthetic hourly energy consumption dataset ({num_samples} samples)...")
        df = generate_energy_consumption_dataset(num_samples=num_samples, seed=seed)
        df.to_csv(raw_path, index=False)
        print(f"[DataLoader] Saved new dataset to '{raw_path}'")
    else:
        df = pd.read_csv(raw_path)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        print(f"[DataLoader] Loaded existing dataset from '{raw_path}' ({len(df)} rows).")

    return df


def explore_time_series_summary(df: pd.DataFrame, target_col: str = "Energy_Consumption_kWh") -> None:
    """Print energy time-series dataset statistics and temporal property summary."""
    print("\n---------------- Time Series Summary (Energy Consumption) ----------------")
    print(f"  Total Timestamp Count : {len(df):,} hourly records")
    print(f"  Start Datetime        : {df['Datetime'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End Datetime          : {df['Datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target Feature        : '{target_col}'")
    print(f"  Energy Min Value      : {df[target_col].min():.3f} kWh")
    print(f"  Energy Max Value      : {df[target_col].max():.3f} kWh")
    print(f"  Energy Mean +- Std     : {df[target_col].mean():.3f} +- {df[target_col].std():.3f} kWh")
    print(f"  Null Values Count     : {df[target_col].isnull().sum()}")
    print("-------------------------------------------------------------------------\n")


if __name__ == "__main__":
    cfg = load_config()
    data = get_or_create_energy_data(cfg)
    explore_time_series_summary(data)

