from typing import List, Dict, Tuple
import torch
import numpy as np

def compute_probe_mastery(
    model: torch.nn.Module,
    hidden_state: torch.Tensor,
    probe_indices: List[int],
    n_observed: int = 0
) -> Tuple[float, float]:
    """
    Implements Algorithm 1 (Probe Mastery and Uncertainty Calculation).
    
    Args:
        model: Trained GRUDKT or compatible estimator.
        hidden_state: Hidden vector for the student [num_layers, 1, d_hidden].
        probe_indices: Fixed list of question indices representing skill k.
        n_observed: Number of historical interactions observed for this student.

    Returns:
        (mastery, uncertainty):
            mastery: Expected probability of answering probe questions correctly.
            uncertainty: Combined variance and sample size uncertainty.
    """
    if not probe_indices:
        return 0.5, 1.0

    model.eval()
    with torch.no_grad():
        # Last hidden state projection
        h_last = hidden_state[-1]  # [1, d_hidden]
        logits = model.out_proj(h_last).squeeze(0)  # [num_questions + 1]
        probs = torch.sigmoid(logits).cpu().numpy()

    probe_probs = [float(probs[idx]) for idx in probe_indices if idx < len(probs)]
    if not probe_probs:
        return 0.5, 1.0

    mastery = float(np.mean(probe_probs))
    sample_uncertainty = 1.0 / np.sqrt(1.0 + max(0, n_observed))
    probe_variance = float(np.var(probe_probs))
    total_uncertainty = float(sample_uncertainty + probe_variance)

    return mastery, total_uncertainty
