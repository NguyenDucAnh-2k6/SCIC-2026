import math
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for Deep Knowledge Tracing sequences.
    Each item contains:
    - question_ids: [seq_len]
    - responses: [seq_len] (0 or 1)
    - elapsed_log: [seq_len]
    - target_responses: [seq_len]
    - mask: [seq_len] (1 for valid steps, 0 for padded)
    """
    def __init__(
        self,
        student_sequences: List[List[Dict[str, Any]]],
        max_seq_len: int = 200,
        q_to_idx: Optional[Dict[str, int]] = None
    ):
        self.max_seq_len = max_seq_len
        self.q_to_idx = q_to_idx or {}
        self.data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._build_tensors(student_sequences)

    def _build_tensors(self, student_sequences: List[List[Dict[str, Any]]]) -> None:
        for seq in student_sequences:
            if len(seq) < 2:
                continue
            trimmed = seq[-self.max_seq_len:]
            seq_len = len(trimmed)

            q_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
            y_tensor = torch.zeros(self.max_seq_len, dtype=torch.float32)
            el_tensor = torch.zeros(self.max_seq_len, dtype=torch.float32)
            target_tensor = torch.zeros(self.max_seq_len, dtype=torch.float32)
            mask_tensor = torch.zeros(self.max_seq_len, dtype=torch.float32)

            for t in range(seq_len - 1):
                item = trimmed[t]
                next_item = trimmed[t + 1]

                qid = str(item.get("question_id", ""))
                q_idx = self.q_to_idx.get(qid, 0)
                y_val = 1.0 if item.get("is_correct") is True else 0.0
                el_ms = max(0.0, float(item.get("elapsed_ms") or 0.0))

                next_y_val = 1.0 if next_item.get("is_correct") is True else 0.0

                q_tensor[t] = q_idx
                y_tensor[t] = y_val
                el_tensor[t] = math.log1p(el_ms)
                target_tensor[t] = next_y_val
                mask_tensor[t] = 1.0

            self.data.append((q_tensor, y_tensor, el_tensor, target_tensor, mask_tensor))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx]

def create_sequence_dataloader(
    student_sequences: List[List[Dict[str, Any]]],
    q_to_idx: Dict[str, int],
    batch_size: int = 64,
    max_seq_len: int = 200,
    shuffle: bool = True
) -> DataLoader:
    dataset = SequenceDataset(student_sequences, max_seq_len=max_seq_len, q_to_idx=q_to_idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
