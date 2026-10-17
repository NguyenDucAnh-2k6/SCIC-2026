from typing import Tuple
import numpy as np
from sklearn.metrics import brier_score_loss

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)
    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(labels[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)

def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Brier score for probabilistic accuracy."""
    return float(brier_score_loss(labels, probs))

class TemperatureScaling:
    """Post-processing calibration via scalar temperature T."""
    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-4, temperature)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))
