import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .schemas import Attempt, Event, ContentItem, ProvenanceEnum

def extract_user_id_from_path(member_path: str) -> str:
    """Extracts student identifier like 'u246317' from member path 'KT1/u246317.csv'."""
    filename = Path(member_path).stem
    return filename

def adapt_kt1_row(
    row: Dict[str, str],
    student_id: str,
    source_row: int,
    correct_answers_map: Optional[Dict[str, str]] = None
) -> Tuple[Optional[Attempt], Optional[Event]]:
    """
    Converts a raw EdNet KT1 CSV row to canonical Attempt and Event models.
    Raw fields: timestamp, solving_id, question_id, user_answer, elapsed_time.
    """
    try:
        ts_ms = int(row.get("timestamp", 0))
        solving_id = int(row.get("solving_id", 0))
        question_id = str(row.get("question_id", "")).strip()
        user_ans = str(row.get("user_answer", "")).strip().lower()
        elapsed_ms = int(row.get("elapsed_time", 0)) if row.get("elapsed_time") else None

        bundle_instance_id = f"{student_id}_{solving_id}"
        attempt_id = f"{student_id}_{ts_ms}_{question_id}"
        event_id = f"evt_{attempt_id}"

        is_correct = None
        if correct_answers_map and question_id in correct_answers_map:
            correct_ans = correct_answers_map[question_id].strip().lower()
            is_correct = (user_ans == correct_ans)

        attempt = Attempt(
            attempt_id=attempt_id,
            student_id=student_id,
            question_id=question_id,
            bundle_instance_id=bundle_instance_id,
            attempt_index=1,
            observed_ts_ms=ts_ms,
            is_correct=is_correct,
            elapsed_ms=elapsed_ms,
            timing_semantics="question_level"
        )

        event = Event(
            event_id=event_id,
            student_id=student_id,
            event_ts_ms=ts_ms,
            source_row=source_row,
            event_type="respond",
            item_id=question_id,
            response=user_ans,
            provenance=ProvenanceEnum.OBSERVED
        )

        return attempt, event
    except Exception:
        return None, None

def adapt_question_row(row: Dict[str, str]) -> Optional[ContentItem]:
    """
    Converts questions.csv row to ContentItem.
    Fields: question_id, bundle_id, correct_answer, part, tags.
    """
    try:
        q_id = str(row.get("question_id", "")).strip()
        if not q_id:
            return None
        bundle_id = str(row.get("bundle_id", "")).strip() or None
        correct_ans = str(row.get("correct_answer", "")).strip().lower() or None
        part = int(row.get("part", 0)) if row.get("part") else None
        
        tags_raw = str(row.get("tags", "")).strip()
        skill_ids = []
        if tags_raw and tags_raw != "-1":
            for t in re.split(r"[;,]", tags_raw):
                t_str = t.strip()
                if t_str.isdigit():
                    skill_ids.append(int(t_str))

        return ContentItem(
            item_id=q_id,
            item_type="question",
            bundle_id=bundle_id,
            skill_ids=skill_ids,
            part=part,
            correct_answer=correct_ans
        )
    except Exception:
        return None
