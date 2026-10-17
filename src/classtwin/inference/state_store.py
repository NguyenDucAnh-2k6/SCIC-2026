from typing import Dict, Any, Optional
import time
import numpy as np

class ClassroomStateStore:
    """
    In-memory / Redis-compatible cache storing live student and classroom states.
    Ensures sub-second retrieval for real-time online inference.
    """
    def __init__(self):
        # class_id -> { student_id -> { "hidden": np.ndarray, "last_ts": int, "n_obs": int } }
        self._student_states: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # class_id -> { "last_action": str, "cumulative_workload": float, "as_of": int, "n_enrolled": int }
        self._class_metadata: Dict[str, Dict[str, Any]] = {}
        # decision_id -> decision_payload
        self._decisions_log: Dict[str, Dict[str, Any]] = {}

    def get_or_create_student_state(
        self,
        class_id: str,
        student_id: str,
        d_hidden: int = 128
    ) -> Dict[str, Any]:
        if class_id not in self._student_states:
            self._student_states[class_id] = {}
        if student_id not in self._student_states[class_id]:
            self._student_states[class_id][student_id] = {
                "hidden": np.zeros(d_hidden, dtype=np.float32),
                "last_ts": 0,
                "n_obs": 0,
                "mastery_estimate": 0.5
            }
        return self._student_states[class_id][student_id]

    def update_student_state(
        self,
        class_id: str,
        student_id: str,
        hidden: np.ndarray,
        ts_ms: int,
        mastery: float
    ) -> None:
        state = self.get_or_create_student_state(class_id, student_id, len(hidden))
        state["hidden"] = hidden
        state["last_ts"] = ts_ms
        state["n_obs"] += 1
        state["mastery_estimate"] = mastery

    def get_class_summary(self, class_id: str, default_enrolled: int = 60) -> Dict[str, Any]:
        students = self._student_states.get(class_id, {})
        n_observed = len(students)
        n_enrolled = self._class_metadata.get(class_id, {}).get("n_enrolled", default_enrolled)

        if n_observed == 0:
            return {
                "class_id": class_id,
                "coverage": 0.0,
                "n_observed": 0,
                "n_enrolled": n_enrolled,
                "mean_mastery": 0.5,
                "q25_mastery": 0.5,
                "q75_mastery": 0.5,
                "uncertainty": 1.0,
                "as_of_ts": int(time.time() * 1000)
            }

        masteries = [s["mastery_estimate"] for s in students.values()]
        coverage = min(1.0, n_observed / max(1, n_enrolled))

        return {
            "class_id": class_id,
            "coverage": coverage,
            "n_observed": n_observed,
            "n_enrolled": n_enrolled,
            "mean_mastery": float(np.mean(masteries)),
            "q25_mastery": float(np.percentile(masteries, 25)),
            "q75_mastery": float(np.percentile(masteries, 75)),
            "uncertainty": float(1.0 - coverage),
            "as_of_ts": int(time.time() * 1000)
        }

    def record_decision(self, decision_id: str, payload: Dict[str, Any]) -> None:
        self._decisions_log[decision_id] = payload

    def get_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        return self._decisions_log.get(decision_id)
