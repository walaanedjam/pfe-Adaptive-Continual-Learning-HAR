"""
Signal preprocessing for IMU-based HAR.

Pipeline (following Amrani 2025):
  1. Resample to TARGET_HZ (50 Hz)
  2. Convert units to m/s²
  3. Remove gravity via 5th-order 0.5 Hz Butterworth low-pass filter
  4. Slide windows of WINDOW_SEC seconds with OVERLAP overlap
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample


TARGET_HZ   = 50          # Hz — standard for HAR (Ravi et al. 2005)
WINDOW_SEC  = 3           # seconds
OVERLAP     = 0.5         # 50 % overlap
WINDOW_SIZE = TARGET_HZ * WINDOW_SEC          # 150 samples
STEP_SIZE   = int(WINDOW_SIZE * (1 - OVERLAP))  # 75 samples
G           = 9.80665     # m/s²


# ---------------------------------------------------------------------------
# 1. Resampling
# ---------------------------------------------------------------------------

def resample_signal(signal: np.ndarray, original_hz: int) -> np.ndarray:
    """Resample signal from original_hz to TARGET_HZ using Fourier method.

    Args:
        signal: (T, C) array
        original_hz: original acquisition frequency

    Returns:
        (T_new, C) resampled array
    """
    if original_hz == TARGET_HZ:
        return signal

    n_original = signal.shape[0]
    n_new = int(n_original * TARGET_HZ / original_hz)
    return resample(signal, n_new, axis=0)


# ---------------------------------------------------------------------------
# 2. Unit conversion
# ---------------------------------------------------------------------------

def convert_g_to_ms2(signal: np.ndarray) -> np.ndarray:
    """Convert accelerometer data from g to m/s²."""
    return signal * G


def convert_ms2_to_g(signal: np.ndarray) -> np.ndarray:
    """Convert accelerometer data from m/s² to g."""
    return signal / G


# ---------------------------------------------------------------------------
# 3. Gravity removal
# ---------------------------------------------------------------------------

def _butter_lowpass(cutoff: float, fs: int, order: int = 5):
    nyq = fs / 2.0
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype="low", analog=False)


def remove_gravity(signal: np.ndarray, fs: int = TARGET_HZ,
                   cutoff: float = 0.5) -> np.ndarray:
    """Estimate and subtract gravity using a 5th-order Butterworth low-pass.

    The gravity component is assumed to live below 0.5 Hz (Amrani 2025).

    Args:
        signal: (T, C) accelerometer array in m/s²
        fs: sampling frequency of the signal
        cutoff: low-pass cutoff frequency in Hz

    Returns:
        (T, C) gravity-free signal
    """
    b, a = _butter_lowpass(cutoff, fs)
    gravity = filtfilt(b, a, signal, axis=0)
    return signal - gravity


# ---------------------------------------------------------------------------
# 4. Sliding window segmentation
# ---------------------------------------------------------------------------

def sliding_windows(signal: np.ndarray,
                    window_size: int = WINDOW_SIZE,
                    step_size: int = STEP_SIZE) -> np.ndarray:
    """Segment a signal into overlapping windows.

    Args:
        signal: (T, C) array
        window_size: number of samples per window
        step_size: stride between windows

    Returns:
        (N, window_size, C) array of N windows
    """
    T = signal.shape[0]
    starts = range(0, T - window_size + 1, step_size)
    windows = [signal[s: s + window_size] for s in starts]
    return np.stack(windows, axis=0) if windows else np.empty((0, window_size, signal.shape[1]))


def sliding_windows_with_labels(signal: np.ndarray,
                                 labels: np.ndarray,
                                 window_size: int = WINDOW_SIZE,
                                 step_size: int = STEP_SIZE):
    """Segment signal into windows and assign majority label to each window.

    Args:
        signal: (T, C) array
        labels: (T,) integer label array (sample-level)
        window_size: samples per window
        step_size: stride

    Returns:
        windows: (N, window_size, C)
        window_labels: (N,) majority label per window
    """
    T = signal.shape[0]
    starts = list(range(0, T - window_size + 1, step_size))

    if not starts:
        return (np.empty((0, window_size, signal.shape[1])),
                np.empty((0,), dtype=np.int64))

    windows = np.stack([signal[s: s + window_size] for s in starts])

    # Majority vote over each window's label sequence
    window_labels = np.array([
        np.bincount(labels[s: s + window_size].astype(int)).argmax()
        for s in starts
    ], dtype=np.int64)

    return windows, window_labels


# ---------------------------------------------------------------------------
# 5. Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_signal(signal: np.ndarray,
                      original_hz: int,
                      unit: str = "ms2",
                      has_gravity: bool = True) -> np.ndarray:
    """Apply the full preprocessing pipeline to a raw IMU signal.

    Args:
        signal: (T, C) raw accelerometer data
        original_hz: original sampling frequency
        unit: "g" or "ms2" — current unit of measurement
        has_gravity: whether gravity is present in the signal

    Returns:
        (T_new, C) preprocessed signal at TARGET_HZ in m/s² without gravity
    """
    # Step 1 — resample
    signal = resample_signal(signal, original_hz)

    # Step 2 — convert to m/s²
    if unit == "g":
        signal = convert_g_to_ms2(signal)

    # Step 3 — remove gravity (always applied; if gravity is absent the filter
    # output is near-zero so subtraction is harmless — matches Amrani 2025)
    signal = remove_gravity(signal, fs=TARGET_HZ)

    return signal


def compute_magnitude(signal: np.ndarray) -> np.ndarray:
    """Compute L2 magnitude across channels.

    Args:
        signal: (..., C) array

    Returns:
        (..., 1) magnitude array
    """
    return np.linalg.norm(signal, axis=-1, keepdims=True)


def normalize_signal(signal: np.ndarray,
                     mean: np.ndarray = None,
                     std: np.ndarray = None):
    """Z-score normalize a signal.

    If mean/std are not provided, they are computed from the signal itself
    (use only for single-dataset normalization; for cross-dataset use the
    training-set statistics).

    Returns:
        normalized signal, mean, std
    """
    if mean is None:
        mean = signal.mean(axis=0, keepdims=True)
    if std is None:
        std = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std, mean, std
