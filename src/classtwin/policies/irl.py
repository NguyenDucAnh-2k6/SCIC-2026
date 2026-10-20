import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

class LatentRewardNet(nn.Module):
    """
    Deep Neural Latent Reward Function R_psi(s, a).
    Maps 12-dimensional classroom telemetry state s and discrete action a to a scalar reward.
    Learned from expert teacher pedagogical demonstrations via Maximum Entropy IRL.
    """
    def __init__(self, d_state: int = 12, num_actions: int = 5, d_hidden: int = 64):
        super().__init__()
        self.d_state = d_state
        self.num_actions = num_actions
        
        # State encoder
        self.state_net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, d_hidden)
        
        # Combined latent reward head
        self.reward_head = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, d_state]
            action: [batch_size] long tensor
        Returns:
            latent_reward: [batch_size, 1]
        """
        s_feat = self.state_net(state)
        a_feat = self.action_embed(action)
        combined = torch.cat([s_feat, a_feat], dim=-1)
        return self.reward_head(combined)

class MaximumEntropyIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning optimizer.
    Recovers the implicit latent reward function underlying expert pedagogical interventions.
    """
    def __init__(self, d_state: int = 12, num_actions: int = 5, lr: float = 1e-3):
        self.reward_net = LatentRewardNet(d_state, num_actions)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=lr, weight_decay=1e-4)

    def compute_latent_reward(self, state: np.ndarray, action_idx: int) -> float:
        """Computes scalar latent reward for a given state and action."""
        self.reward_net.eval()
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            a_t = torch.tensor([action_idx], dtype=torch.long)
            r = self.reward_net(s_t, a_t)
            return float(r.item())

    def train_step(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        simulated_states: torch.Tensor,
        simulated_actions: torch.Tensor
    ) -> float:
        """
        Performs one gradient step of MaxEnt-IRL:
        Maximizes likelihood of expert trajectories while penalizing random/exploratory rollouts.
        """
        self.reward_net.train()
        self.optimizer.zero_grad()

        # Expert rewards should be higher
        r_expert = self.reward_net(expert_states, expert_actions)
        r_simulated = self.reward_net(simulated_states, simulated_actions)

        # Cross-entropy / MaxEnt partition approximation loss
        loss = -torch.mean(r_expert) + torch.logsumexp(r_simulated, dim=0).mean()
        # L2 regularizer on reward magnitude
        loss += 0.01 * (torch.mean(r_expert ** 2) + torch.mean(r_simulated ** 2))

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.reward_net.state_dict(), path)

    def load(self, path: Path):
        self.reward_net.load_state_dict(torch.load(path, map_location="cpu"))
        self.reward_net.eval()

def generate_expert_demonstrations(env, num_episodes: int = 20, temperature: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates rich, balanced pedagogical demonstration trajectories from a
    Stochastic Expert Pedagogical Demonstrator (Probabilistic Multi-Criteria Softmax):
    - Replaces rigid deterministic if-else heuristics to eliminate Mode Collapse in MaxEnt IRL.
    - Balances curriculum delivery (lecture), active retrieval (quiz), collaborative peer scaffolding
      (groupwork), guided reflection (discussion), and physiological restoration (break).
    - Incorporates pedagogical anti-monotony penalties to ensure curriculum diversity across all 5 actions.
    """
    try:
        from classtwin.policies.action_space import PEDAGOGICAL_ACTIONS, DEFAULT_DURATIONS_MINUTES
    except ImportError:
        from src.classtwin.policies.action_space import PEDAGOGICAL_ACTIONS, DEFAULT_DURATIONS_MINUTES

    states, actions = [], []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        step_in_ep = 0
        last_act = None

        while not done:
            m_mean = float(state[0])
            q25 = float(state[1])
            q75 = float(state[2])
            gap = max(0.0, q75 - q25)
            att = float(state[5])
            fatigue = float(state[6])
            co2_norm = float(state[7])

            # Pedagogical Utility Scores for each action
            utilities = {
                # Lecture: Highest value early in lesson and when class mastery is low
                "lecture": (
                    2.5 * max(0.0, 0.60 - m_mean)
                    + (1.5 if step_in_ep <= 1 else 0.0)
                    - 1.2 * fatigue
                    - (2.0 if last_act == "lecture" else 0.0)
                ),
                # Quiz: Formative retrieval practice after learning or when mastery is moderate/high
                "quiz": (
                    1.6 * m_mean
                    + (1.5 if last_act in ["lecture", "discussion"] else 0.2)
                    - 0.8 * fatigue
                    - (2.5 if last_act == "quiz" else 0.0)
                ),
                # Groupwork: Peer scaffolding effective when knowledge disparity (gap) is pronounced
                "groupwork": (
                    2.0 * gap
                    + 1.0 * att
                    - 1.5 * fatigue
                    - (2.5 if last_act == "groupwork" else 0.0)
                ),
                # Discussion: Interactive deepening between lecture and assessment
                "discussion": (
                    1.2 * m_mean
                    + 1.0 * att
                    - 1.0 * fatigue
                    - (2.0 if last_act == "discussion" else 0.0)
                ),
                # Break: Restorative reset triggered by fatigue or high CO2; heavily penalized at start or consecutively
                "break": (
                    4.0 * max(0.0, fatigue - 0.38)
                    + 3.0 * max(0.0, co2_norm - 0.90)
                    - (4.0 if last_act == "break" else 0.0)
                    - (3.0 if step_in_ep <= 1 else 0.0)
                )
            }

            scores = np.array([utilities[a] for a in PEDAGOGICAL_ACTIONS], dtype=np.float32)
            # Softmax with pedagogical temperature
            scaled_scores = (scores - np.max(scores)) / max(1e-4, temperature)
            exp_scores = np.exp(scaled_scores)
            probs = exp_scores / np.sum(exp_scores)

            act_idx = int(np.random.choice(len(PEDAGOGICAL_ACTIONS), p=probs))
            act = PEDAGOGICAL_ACTIONS[act_idx]
            dur = DEFAULT_DURATIONS_MINUTES[act]

            states.append(state)
            actions.append(act_idx)

            next_state, _, done, _ = env.step(act, duration_minutes=dur)
            state = next_state
            last_act = act
            step_in_ep += 1

    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)
