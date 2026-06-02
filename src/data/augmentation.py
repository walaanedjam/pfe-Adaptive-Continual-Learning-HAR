"""
Data augmentation for IMU time-series.

All transforms operate on (T, C) or (B, T, C) numpy arrays.
Used during training to reduce overfitting and improve accuracy.

Techniques (from literature on sensor-based HAR):
  1. Gaussian noise        — simulate sensor noise
  2. Amplitude scaling     — simulate different sensor sensitivities
  3. Time-warp             — simulate different movement speeds
  4. Channel dropout       — robustness to missing channels
  5. Window slicing        — randomly crop a sub-window of same length
"""

from __future__ import annotations
import numpy as np


def add_noise(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add zero-mean Gaussian noise."""
    return x + np.random.randn(*x.shape).astype(np.float32) * sigma


def scale_amplitude(x: np.ndarray,
                    min_scale: float = 0.8,
                    max_scale: float = 1.2) -> np.ndarray:
    """Randomly scale each channel independently."""
    scales = np.random.uniform(min_scale, max_scale, size=(1, x.shape[-1]))
    return (x * scales).astype(np.float32)


def time_warp(x: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Random time-warping via smooth distortion of the time axis.
    Uses cumulative sum of a smooth random curve to warp time.
    """
    T, C = x.shape
    # Smooth warp curve via cumsum of N(0,sigma)
    warp = np.random.randn(T).astype(np.float32) * sigma
    warp = np.cumsum(warp)
    warp -= warp.min()
    warp /= (warp.max() + 1e-8)
    # Map to [0, T-1] uniformly
    orig = np.linspace(0, 1, T)
    warped_t = warp * (T - 1)
    # Interpolate each channel
    out = np.stack(
        [np.interp(warped_t, np.arange(T), x[:, c]) for c in range(C)],
        axis=-1,
    ).astype(np.float32)
    return out


def channel_dropout(x: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Zero out each channel with probability p (simulates sensor failure)."""
    mask = (np.random.rand(x.shape[-1]) > p).astype(np.float32)
    return (x * mask[np.newaxis, :]).astype(np.float32)


def window_slice(x: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """Randomly crop a fraction of the window, then resize back."""
    T, C = x.shape
    crop_len = max(1, int(T * crop_ratio))
    start = np.random.randint(0, T - crop_len + 1)
    sliced = x[start: start + crop_len]
    # Resize back to T via linear interpolation
    out = np.stack(
        [np.interp(np.linspace(0, 1, T),
                   np.linspace(0, 1, crop_len),
                   sliced[:, c])
         for c in range(C)],
        axis=-1,
    ).astype(np.float32)
    return out


def flip_horizontal(x: np.ndarray) -> np.ndarray:
    """Reverse time — valid augmentation since many activities are symmetric."""
    return x[::-1].copy().astype(np.float32)


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

class IMUAugmenter:
    """
    Stochastic augmentation pipeline for IMU windows.

    Each augmentation is applied independently with probability p_apply.
    Safe defaults: minimal distortion, preserves activity semantics.

    Args:
        p_apply:      probability of applying each individual transform
        noise_sigma:  std of Gaussian noise
        scale_range:  (min, max) amplitude scaling
        warp_sigma:   time-warp distortion strength
        dropout_p:    per-channel zero-out probability
        crop_ratio:   fraction of window kept in window_slice
        use_flip:     include horizontal flip (time reversal)
    """

    def __init__(
        self,
        p_apply:     float = 0.5,
        noise_sigma: float = 0.01,
        scale_range: tuple = (0.85, 1.15),
        warp_sigma:  float = 0.05,
        dropout_p:   float = 0.1,
        crop_ratio:  float = 0.9,
        use_flip:    bool  = False,
    ):
        self.p_apply     = p_apply
        self.noise_sigma = noise_sigma
        self.scale_range = scale_range
        self.warp_sigma  = warp_sigma
        self.dropout_p   = dropout_p
        self.crop_ratio  = crop_ratio
        self.use_flip    = use_flip

    def _apply(self, transform, x):
        if np.random.rand() < self.p_apply:
            return transform(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (T, C) float32 array

        Returns:
            augmented (T, C) float32 array
        """
        x = self._apply(lambda a: add_noise(a, self.noise_sigma), x)
        x = self._apply(lambda a: scale_amplitude(a, *self.scale_range), x)
        x = self._apply(lambda a: time_warp(a, self.warp_sigma), x)
        x = self._apply(lambda a: channel_dropout(a, self.dropout_p), x)
        x = self._apply(lambda a: window_slice(a, self.crop_ratio), x)
        if self.use_flip:
            x = self._apply(flip_horizontal, x)
        return x

    def augment_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply to a batch (N, T, C) — processes each sample independently."""
        return np.stack([self(X[i]) for i in range(len(X))], axis=0)
