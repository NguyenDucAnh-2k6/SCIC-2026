from typing import Dict, Any, List, Tuple
import numpy as np
from .action_space import PEDAGOGICAL_ACTIONS

class RuleBasedPolicy:
    """
    Standard pedagogical rule-based / heuristic policy baseline.
    Decides actions based on transparent pedagogical criteria.
    """
    def __init__(self, fatigue_threshold: float = 0.60, weak_ratio_threshold: float = 0.30):
        self.fatigue_threshold = fatigue_threshold
        self.weak_ratio_threshold = weak_ratio_threshold

    def select_action(self, state_dict: Dict[str, float], valid_mask: List[int]) -> Tuple[str, str]:
        """
        Returns (action_name, rationale_string).
        """
        mean_fatigue = state_dict.get("mean_fatigue", 0.2)
        rho_low = state_dict.get("rho_low", 0.2)
        mean_attention = state_dict.get("mean_attention", 0.8)
        mean_mastery = state_dict.get("mean_mastery", 0.5)

        # Priority 1: High cognitive fatigue -> Break
        if mean_fatigue > self.fatigue_threshold and valid_mask[4] == 1:
            return "break", f"Mức độ mệt mỏi nhận thức cao ({mean_fatigue:.1%}). Đề xuất giải lao 3 phút."

        # Priority 2: Struggling students ratio high -> Discussion / Review
        if rho_low > self.weak_ratio_threshold and valid_mask[3] == 1:
            return "discussion", f"Tỷ lệ sinh viên chưa nắm bài cao ({rho_low:.1%}). Đề xuất thảo luận giải đáp thắc mắc."

        # Priority 3: Good mastery but declining attention -> Groupwork activity
        if mean_mastery > 0.65 and mean_attention < 0.50 and valid_mask[2] == 1:
            return "groupwork", f"Độ hiểu bài tốt ({mean_mastery:.1%}) nhưng chú ý giảm. Đề xuất bài tập nhóm tương tác."

        # Priority 4: Knowledge check quiz
        if mean_attention > 0.60 and valid_mask[1] == 1:
            return "quiz", "Kiểm tra mức độ tiếp thu bằng câu hỏi ngắn tương tác."

        # Default fallback: Continue lecture
        return "lecture", "Tiến trình lớp học ổn định, tiếp tục thuyết giảng lý thuyết."
