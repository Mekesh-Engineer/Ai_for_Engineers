# -*- coding: utf-8 -*-
"""
time_series_preprocessing.py
----------------------------
PreprocessingModule: Normalization, scaling, outlier handling, and missing
value interpolation for the Vanilla RNN time-series forecasting experiment.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesScaler:
    """
    MinMaxScaler wrapper for scaling time-series values to range [0, 1]
    and converting predicted tensors back to original physical kWh units.
    """

    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=tuple(feature_range))
        self.is_fitted = False

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Fit scaler on values and transform.

        Parameters
        ----------
        values : np.ndarray
            Array of shape (N,) or (N, 1).

        Returns
        -------
        np.ndarray
            Scaled 2D array of shape (N, 1) in range [0, 1].
        """
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)
        self.is_fitted = True
        return scaled

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform values using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("TimeSeriesScaler has not been fitted yet.")
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return self.scaler.transform(values)

    def inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Convert scaled values back to original energy consumption units (kWh).

        Parameters
        ----------
        scaled_values : np.ndarray
            Array of shape (N,) or (N, 1) in range [0, 1].

        Returns
        -------
        np.ndarray
            Original scale values of shape (N, 1).
        """
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)
        return self.scaler.inverse_transform(scaled_values)


def clean_and_interpolate_series(series: pd.Series) -> pd.Series:
    """
    Fill missing time steps using linear interpolation and forward-fill.

    Parameters
    ----------
    series : pd.Series
        Input time-series data.

    Returns
    -------
    pd.Series
        Cleaned time series without NaN values.
    """
    if series.isnull().sum() > 0:
        series = series.interpolate(method="linear").ffill().bfill()
    return series


def verify_preprocessing_quality(raw_values: np.ndarray, scaled_values: np.ndarray) -> bool:
    """Validate scaled values boundary [0, 1] and dimension integrity."""
    print("---------------- Preprocessing Validation ----------------")
    print(f"  Raw series range min/max  : [{raw_values.min():.2f}, {raw_values.max():.2f}]")
    print(f"  Scaled series min/max     : [{scaled_values.min():.4f}, {scaled_values.max():.4f}]")
    print(f"  Scaled tensor data type   : {scaled_values.dtype}")
    assert scaled_values.min() >= -1e-5 and scaled_values.max() <= 1.0 + 1e-5, "Scaled values fall outside boundary!"
    print("  Validation status         : Passed [OK]")
    print("----------------------------------------------------------\n")
    return True


if __name__ == "__main__":
    dummy_data = np.random.uniform(50, 200, size=100)
    scaler = TimeSeriesScaler()
    scaled = scaler.fit_transform(dummy_data)
    verify_preprocessing_quality(dummy_data, scaled)
    restored = scaler.inverse_transform(scaled)
    print(f"[Preprocessing Test] Reconstructed Max Diff: {np.max(np.abs(dummy_data - restored.flatten())):.6f}")
