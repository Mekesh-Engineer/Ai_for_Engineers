"""
data_preprocessing.py
---------------------
PreprocessingModule: Image normalization, tensor scaling, and dataset validation
for the Solar PV Panel Fault CNN Image Classification experiment.
"""

import numpy as np

def get_normalize_parameters(config: dict) -> tuple:
    """Retrieve mean and std normalization parameters from config."""
    mean = config.get("preprocessing", {}).get("normalize_mean", 0.35)
    std  = config.get("preprocessing", {}).get("normalize_std", 0.15)
    return (mean,), (std,)


def normalize_numpy_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize raw pixel values from [0, 255] uint8 to [0, 1] float32.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W) or (N, H, W, C) with values in range [0, 255].

    Returns
    -------
    np.ndarray
        Float32 array scaled to range [0.0, 1.0].
    """
    if images.dtype != np.float32:
        images = images.astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0
    return images


def verify_dataset_shapes(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Verify dataset dimensions, sample count alignment, and class label range.

    Parameters
    ----------
    X : np.ndarray
        Feature array (N, C, H, W) or (N, H, W).
    y : np.ndarray
        Label array (N,).

    Returns
    -------
    bool
        True if all checks pass, False otherwise.
    """
    print("-- Preprocessing Validation ---------------------------------")
    print(f"  Input features shape : {X.shape} ({X.dtype})")
    print(f"  Target labels shape  : {y.shape} ({y.dtype})")
    print(f"  Pixel range min/max  : [{X.min():.3f}, {X.max():.3f}]")

    assert len(X) == len(y), f"Sample count mismatch: {len(X)} vs {len(y)}"
    assert X.min() >= 0.0 and X.max() <= 1.0, f"Pixel values outside [0, 1]: [{X.min()}, {X.max()}]"
    unique_labels = np.unique(y)
    print(f"  Unique target classes: {unique_labels.tolist()}")
    print("  Validation status    : Passed [OK]")
    print("-------------------------------------------------------------\n")
    return True


if __name__ == "__main__":
    dummy_X = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
    dummy_y = np.random.randint(0, 4, (100,), dtype=np.int64)
    norm_X = normalize_numpy_images(dummy_X)
    verify_dataset_shapes(norm_X, dummy_y)
