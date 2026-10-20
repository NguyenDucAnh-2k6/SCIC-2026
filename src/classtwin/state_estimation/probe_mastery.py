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
        h_last = hidden_state[-1]  # [1, d_hidden]
        # Fast sliced projection: chỉ nhân chiếu lên tập probe_indices thay vì toàn bộ 12,280 chiều
        w_sub = model.out_proj.weight[probe_indices]
        b_sub = model.out_proj.bias[probe_indices]
        logits_sub = torch.matmul(h_last, w_sub.t()) + b_sub
        probs = torch.sigmoid(logits_sub).squeeze(0).cpu().numpy()

    mastery = float(np.mean(probs))
    sample_uncertainty = 1.0 / np.sqrt(1.0 + max(0, n_observed))
    probe_variance = float(np.var(probs))
    total_uncertainty = float(sample_uncertainty + probe_variance)

    return mastery, total_uncertainty
