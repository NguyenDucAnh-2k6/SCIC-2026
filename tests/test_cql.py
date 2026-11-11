import torch
import numpy as np
from src.classtwin.policies.cql import DiscreteCQL
from src.classtwin.policies.action_space import get_valid_action_mask, PEDAGOGICAL_ACTIONS

def test_action_masking():
    mask = get_valid_action_mask(time_remaining_minutes=2.0, current_cumulative_workload=0.0)
    assert len(mask) == len(PEDAGOGICAL_ACTIONS)
    assert sum(mask) >= 1

def test_cql_train_step_12dim():
    cql = DiscreteCQL(d_state=12, num_actions=5, d_hidden=32)
    s = torch.randn(8, 12)
    a = torch.randint(0, 5, (8,))
    r = torch.randn(8)
    next_s = torch.randn(8, 12)
    dur = torch.full((8,), 300.0)
    d = torch.zeros(8)
    masks = torch.ones(8, 5)

    metrics = cql.train_batch(s, a, r, next_s, dur, d, masks)
    assert "total_loss" in metrics
    assert "bellman_loss" in metrics
    assert "cql_loss" in metrics
    assert not np.isnan(metrics["total_loss"])

def test_cql_train_step_32dim():
    cql = DiscreteCQL(d_state=32, num_actions=5, d_hidden=32)
    s = torch.randn(8, 32)
    a = torch.randint(0, 5, (8,))
    r = torch.randn(8)
    next_s = torch.randn(8, 32)
    dur = torch.full((8,), 300.0)
    d = torch.zeros(8)
    masks = torch.ones(8, 5)

    metrics = cql.train_batch(s, a, r, next_s, dur, d, masks)
    assert "total_loss" in metrics
    assert not np.isnan(metrics["total_loss"])
