"""
ASSISTments Dataset Adapter for ClassTwin
Processes real classroom cohort data (classes of 30-60 students, teacher IDs, hints, scaffolds)
from ASSISTments Skill Builder benchmarks and adapts them to ClassTwin simulator format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from classtwin.data.schemas import AssistmentsRecord, ClassroomCohort

class AssistmentsAdapter:
    """
    Adapter to load and process ASSISTments classroom-level benchmark logs.
    """
    def __init__(self, csv_path: str = "data/assistments/ASSISTments_skill_builder_data.csv"):
        self.csv_path = Path(csv_path)
        self._df: Optional[pd.DataFrame] = None

    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load ASSISTments dataset with optional row limit.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"ASSISTments file not found at {self.csv_path}")
        
        cols = [
            "order_id", "assignment_id", "user_id", "problem_id",
            "correct", "attempt_count", "ms_first_response",
            "student_class_id", "teacher_id", "skill_id", "skill_name",
            "hint_count", "hint_total"
        ]
        self._df = pd.read_csv(self.csv_path, usecols=cols, nrows=nrows, low_memory=False)
        return self._df

    def get_classroom_cohorts(self, min_students: int = 30, max_students: int = 65) -> List[ClassroomCohort]:
        """
        Find classroom cohorts that match typical lecture/seminar sizes (e.g. 30-65 students).
        """
        if self._df is None:
            self.load_data()

        cohorts = []
        class_groups = self._df.groupby(["student_class_id", "teacher_id"])
        
        for (class_id, teacher_id), group in class_groups:
            student_ids = group["user_id"].unique().tolist()
            student_count = len(student_ids)
            
            if min_students <= student_count <= max_students:
                skills = group["skill_name"].dropna().unique().tolist()
                skill_ids = [int(s) for s in group["skill_id"].dropna().unique().tolist()]
                
                cohorts.append(ClassroomCohort(
                    class_id=int(class_id),
                    teacher_id=int(teacher_id),
                    student_ids=student_ids,
                    student_count=student_count,
                    skill_ids=skill_ids[:10],
                    skill_names=skills[:10],
                    total_attempts=len(group),
                    mean_first_attempt_accuracy=float(group["correct"].mean()),
                    mean_hints_per_problem=float(group["hint_count"].mean()),
                    mean_response_time_ms=float(group["ms_first_response"].dropna().mean()) if not group["ms_first_response"].dropna().empty else 0.0
                ))
                
        # Sort by total attempts descending
        cohorts.sort(key=lambda c: c.total_attempts, reverse=True)
        return cohorts

    def extract_cohort_matrix(self, class_id: int) -> pd.DataFrame:
        """
        Extract tabular interaction matrix for a specific cohort.
        """
        if self._df is None:
            self.load_data()
        
        sub = self._df[self._df["student_class_id"] == class_id].copy()
        sub.sort_values(by="order_id", inplace=True)
        return sub
