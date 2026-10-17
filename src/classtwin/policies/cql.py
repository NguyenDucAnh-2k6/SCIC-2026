import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
from .action_space import PEDAGOGICAL_ACTIONS, DEFAULT_DURATIONS_MINUTES

class DiscreteQNetwork(nn.Module):
    """Multi-Layer Perceptron Q-Network mapping state vector x to Q-values for all discrete actions."""
    def __init__(self, d_state: int = 12, num_actions: int = 5, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, num_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class DiscreteCQL:
    """
    Implements Algorithm 3 (Discrete Conservative Q-Learning with Action Masking and Semi-Markov Discounting).
    """
    def __init__(
        self,
        d_state: int = 12,
        num_actions: int = len(PEDAGOGICAL_ACTIONS),
        d_hidden: int = 128,
        lr: float = 0.0003,
        cql_alpha: float = 1.0,
        gamma_base: float = 0.99,
        tau_seconds: float = 60.0,
        polyak_tau: float = 0.005,
        device: str = "cpu"
    ):
        self.num_actions = num_actions
        self.cql_alpha = cql_alpha
        self.gamma_base = gamma_base
        self.tau_seconds = tau_seconds
        self.polyak_tau = polyak_tau
        self.device = torch.device(device)

        self.q_net = DiscreteQNetwork(d_state, num_actions, d_hidden).to(self.device)
        self.target_q_net = DiscreteQNetwork(d_state, num_actions, d_hidden).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)

    def select_action(
        self,
        state: np.ndarray,
        valid_mask: List[int],
        deterministic: bool = True
    ) -> Tuple[int, str]:
        """Chooses best pedagogical action satisfying valid mask."""
        self.q_net.eval()
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_net(s_t).squeeze(0).cpu().numpy()

        # Apply action mask: set masked actions to large negative value
        masked_q = np.copy(q_vals)
        for i, m in enumerate(valid_mask):
            if m == 0:
                masked_q[i] = -1e9

        if deterministic:
            best_idx = int(np.argmax(masked_q))
        else:
            # Softmax over valid actions
            exp_q = np.exp(masked_q - np.max(masked_q))
            probs = exp_q / np.sum(exp_q)
            best_idx = int(np.random.choice(len(probs), p=probs))

        return best_idx, PEDAGOGICAL_ACTIONS[best_idx]

    def train_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        durations_seconds: torch.Tensor,
        dones: torch.Tensor,
        next_action_masks: torch.Tensor
    ) -> Dict[str, float]:
        """
        Executes one batch gradient step for Conservative Q-Learning.
        """
        self.q_net.train()
        self.target_q_net.eval()

        # Semi-Markov discounting: gamma = gamma_base ** (delta_t / tau_0)
        gammas = torch.pow(self.gamma_base, durations_seconds / self.tau_seconds).unsqueeze(-1)

        # 1. Compute Bellman Target
        with torch.no_grad():
            next_q_all = self.target_q_net(next_states)  # [B, num_actions]
            # Mask out invalid next actions
            mask_penalty = (1.0 - next_action_masks) * 1e9
            masked_next_q = next_q_all - mask_penalty
            max_next_q, _ = torch.max(masked_next_q, dim=-1, keepdim=True)
            y_td = rewards.unsqueeze(-1) + gammas * (1.0 - dones.unsqueeze(-1)) * max_next_q

        # 2. Current Q estimate
        current_q_all = self.q_net(states)  # [B, num_actions]
        current_q = current_q_all.gather(1, actions.unsqueeze(-1))

        # 3. Standard Bellman Loss
        bellman_loss = 0.5 * torch.mean((current_q - y_td) ** 2)

        # 4. CQL Conservative Loss (penalize out-of-distribution high Q values)
        # logsumexp over actions - Q(s, a_data)
        logsumexp_q = torch.logsumexp(current_q_all, dim=-1, keepdim=True)
        cql_loss = torch.mean(logsumexp_q - current_q)

        total_loss = bellman_loss + self.cql_alpha * cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak soft update of target network
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.polyak_tau * param.data + (1.0 - self.polyak_tau) * target_param.data)

        return {
            "total_loss": float(total_loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "cql_loss": float(cql_loss.item()),
            "mean_q": float(current_q.mean().item())
        }
