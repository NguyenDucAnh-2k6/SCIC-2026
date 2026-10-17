import torch
import torch.nn as nn
from typing import Tuple, Optional

class GRUDKT(nn.Module):
    """
    Deep Knowledge Tracing using Gated Recurrent Units (GRU)
    with elapsed time projection and question embeddings.
    """
    def __init__(
        self,
        num_questions: int,
        d_model: int = 64,
        d_hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_questions = num_questions
        self.d_model = d_model
        self.d_hidden = d_hidden

        self.q_embedding = nn.Embedding(num_questions + 1, d_model, padding_idx=0)
        self.elapsed_proj = nn.Linear(1, d_model // 2)

        # Input dimension: d_model (question) + 1 (binary response) + d_model // 2 (elapsed log time)
        in_dim = d_model + 1 + (d_model // 2)
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_hidden, num_questions + 1)

    def forward(
        self,
        question_ids: torch.Tensor,
        responses: torch.Tensor,
        elapsed_logs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            question_ids: [batch_size, seq_len]
            responses: [batch_size, seq_len]
            elapsed_logs: [batch_size, seq_len]
            hidden_state: [num_layers, batch_size, d_hidden]
        Returns:
            logits: [batch_size, seq_len, num_questions + 1]
            last_hidden: [num_layers, batch_size, d_hidden]
        """
        q_emb = self.q_embedding(question_ids)  # [B, L, d_model]
        resp = responses.unsqueeze(-1)          # [B, L, 1]
        el_emb = self.elapsed_proj(elapsed_logs.unsqueeze(-1))  # [B, L, d_model // 2]

        x = torch.cat([q_emb, resp, el_emb], dim=-1)  # [B, L, in_dim]
        gru_out, last_hidden = self.gru(x, hidden_state)  # [B, L, d_hidden]
        gru_out = self.dropout(gru_out)
        logits = self.out_proj(gru_out)  # [B, L, num_questions + 1]

        return logits, last_hidden

    def step_single(
        self,
        q_idx: int,
        response: float,
        elapsed_ms: float,
        hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single recurrent step for online inference."""
        self.eval()
        with torch.no_grad():
            q_t = torch.tensor([[q_idx]], dtype=torch.long, device=hidden_state.device)
            resp_t = torch.tensor([[response]], dtype=torch.float32, device=hidden_state.device)
            import math
            el_t = torch.tensor([[math.log1p(max(0.0, elapsed_ms))]], dtype=torch.float32, device=hidden_state.device)
            logits, next_hidden = self.forward(q_t, resp_t, el_t, hidden_state)
            probs = torch.sigmoid(logits.squeeze(1))  # [1, num_questions + 1]
            return probs, next_hidden
