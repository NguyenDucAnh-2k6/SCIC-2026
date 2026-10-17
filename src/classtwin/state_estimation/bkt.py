from typing import List, Dict, Any, Optional

class BayesianKnowledgeTracing:
    """
    Standard 4-parameter Bayesian Knowledge Tracing (BKT) baseline:
    - P(L0): Prior probability of initial mastery
    - P(T): Transition probability (learning rate)
    - P(G): Guess probability
    - P(S): Slip probability
    """
    def __init__(
        self,
        p_l0: float = 0.2,
        p_t: float = 0.1,
        p_g: float = 0.25,
        p_s: float = 0.1
    ):
        self.p_l0 = p_l0
        self.p_t = p_t
        self.p_g = p_g
        self.p_s = p_s

    def update_posterior(self, p_l_prev: float, is_correct: bool) -> float:
        """Computes posterior mastery P(L_t | observation) and projects ahead to P(L_{t+1})."""
        if is_correct:
            numerator = p_l_prev * (1.0 - self.p_s)
            denominator = numerator + (1.0 - p_l_prev) * self.p_g
        else:
            numerator = p_l_prev * self.p_s
            denominator = numerator + (1.0 - p_l_prev) * (1.0 - self.p_g)

        p_l_given_obs = numerator / max(1e-8, denominator)
        # Transition step: P(L_{t+1}) = P(L_t | obs) + (1 - P(L_t | obs)) * P(T)
        p_l_next = p_l_given_obs + (1.0 - p_l_given_obs) * self.p_t
        return float(min(1.0, max(0.0, p_l_next)))

    def predict_correct(self, p_l: float) -> float:
        """P(C_{t+1}) = P(L_t) * (1 - S) + (1 - P(L_t)) * G"""
        return p_l * (1.0 - self.p_s) + (1.0 - p_l) * self.p_g

    def evaluate_sequence(self, responses: List[bool]) -> List[float]:
        """Runs filtering over sequence of responses and returns mastery timeline."""
        p_l = self.p_l0
        mastery_history = [p_l]
        for resp in responses:
            p_l = self.update_posterior(p_l, resp)
            mastery_history.append(p_l)
        return mastery_history
