import torch
import numpy as np
from src.classtwin.state_estimation.bkt import BayesianKnowledgeTracing
from src.classtwin.state_estimation.gru_dkt import GRUDKT
from src.classtwin.state_estimation.probe_mastery import compute_probe_mastery
from src.classtwin.state_estimation.calibration import compute_ece, compute_brier

def test_bkt_updates():
    bkt = BayesianKnowledgeTracing(p_l0=0.2, p_t=0.1, p_g=0.25, p_s=0.1)
    # Successive correct answers should increase mastery monotonically
    m1 = bkt.update_posterior(0.2, is_correct=True)
    m2 = bkt.update_posterior(m1, is_correct=True)
    assert m1 > 0.2
    assert m2 > m1

def test_gru_dkt_forward():
    model = GRUDKT(num_questions=10, d_model=32, d_hidden=64)
    q = torch.tensor([[1, 2, 3]], dtype=torch.long)
    resp = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    el = torch.tensor([[9.0, 10.0, 9.5]], dtype=torch.float32)

    logits, hidden = model(q, resp, el)
    assert logits.shape == (1, 3, 11)
    assert hidden.shape == (1, 1, 64)

def test_probe_mastery_computation():
    model = GRUDKT(num_questions=10, d_model=32, d_hidden=64)
    hidden = torch.zeros(1, 1, 64)
    probes = [1, 2, 3, 4]
    mastery, uncertainty = compute_probe_mastery(model, hidden, probes, n_observed=5)
    assert 0.0 <= mastery <= 1.0
    assert uncertainty > 0.0

def test_calibration_metrics():
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    ece = compute_ece(probs, labels)
    brier = compute_brier(probs, labels)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 1.0
