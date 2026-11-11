"""
EdNet Schema Adapters (KT1 - KT4):
Converts raw EdNet CSV rows across all 4 tiers (KT1-KT4) and questions.csv
into canonical Attempt, Event, and ContentItem models.

Tiers:
- KT1: Question response interaction tracks (question_id, solving_id, elapsed_time)
- KT2: Explanation & review views (bundle_id, item_id, enter, respond, submit)
- KT3: Lecture video consumption (video_id, enter, quit, play, pause, seek)
- KT4: Full multi-action session trajectories across question, video, and review activities
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from classtwin.data.schemas import Attempt, Event, ContentItem, ProvenanceEnum

def extract_user_id_from_path(member_path: str) -> str:
    """Extracts student identifier like 'u246317' from member path 'KT1/u246317.csv'."""
    filename = Path(member_path).stem
    return filename

def adapt_kt1_row(
    row: Dict[str, Any],
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
        question_id = str(row.get("question_id", row.get("item_id", ""))).strip()
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

def adapt_kt2_row(
    row: Dict[str, Any],
    student_id: str,
    source_row: int,
    correct_answers_map: Optional[Dict[str, str]] = None
) -> Tuple[Optional[Attempt], Optional[Event]]:
    """
    Converts a raw EdNet KT2 CSV row (explanation consultations & diagnostic events).
    Raw fields: timestamp, action_type, item_id, source, user_answer, platform.
    """
    try:
        ts_ms = int(row.get("timestamp", 0))
        action_type = str(row.get("action_type", "enter")).strip().lower()
        item_id = str(row.get("item_id", "")).strip()
        user_ans = str(row.get("user_answer", "")).strip().lower() if row.get("user_answer") else None
        platform = str(row.get("platform", "")).strip() or None
        source = str(row.get("source", "")).strip() or None

        event_id = f"evt_kt2_{student_id}_{ts_ms}_{source_row}"
        event = Event(
            event_id=event_id,
            student_id=student_id,
            event_ts_ms=ts_ms,
            source_row=source_row,
            event_type=action_type,
            item_id=item_id,
            response=user_ans,
            source=source,
            platform=platform,
            provenance=ProvenanceEnum.OBSERVED
        )

        attempt = None
        # If this action is a question response in KT2 (e.g. item_id starts with 'q')
        if action_type == "respond" and item_id.startswith("q"):
            is_correct = None
            if correct_answers_map and item_id in correct_answers_map:
                is_correct = (user_ans == correct_answers_map[item_id].strip().lower())
            attempt = Attempt(
                attempt_id=f"{student_id}_{ts_ms}_{item_id}",
                student_id=student_id,
                question_id=item_id,
                bundle_instance_id=f"{student_id}_kt2",
                attempt_index=1,
                observed_ts_ms=ts_ms,
                is_correct=is_correct,
                elapsed_ms=None
            )

        return attempt, event
    except Exception:
        return None, None

def adapt_kt3_row(
    row: Dict[str, Any],
    student_id: str,
    source_row: int
) -> Tuple[Optional[Attempt], Optional[Event]]:
    """
    Converts a raw EdNet KT3 CSV row (video lecture consumption).
    Raw fields: timestamp, action_type, item_id, source, user_answer, platform.
    """
    try:
        ts_ms = int(row.get("timestamp", 0))
        action_type = str(row.get("action_type", "enter")).strip().lower()
        item_id = str(row.get("item_id", "")).strip()
        source = str(row.get("source", "lecture")).strip() or None
        platform = str(row.get("platform", "mobile")).strip() or None

        event_id = f"evt_kt3_{student_id}_{ts_ms}_{source_row}"
        event = Event(
            event_id=event_id,
            student_id=student_id,
            event_ts_ms=ts_ms,
            source_row=source_row,
            event_type=action_type,
            item_id=item_id,
            response=None,
            source=source,
            platform=platform,
            provenance=ProvenanceEnum.OBSERVED
        )
        return None, event
    except Exception:
        return None, None

def adapt_kt4_row(
    row: Dict[str, Any],
    student_id: str,
    source_row: int,
    correct_answers_map: Optional[Dict[str, str]] = None
) -> Tuple[Optional[Attempt], Optional[Event]]:
    """
    Converts a raw EdNet KT4 CSV row (full multi-action session trajectories).
    Raw fields: timestamp, action_type, item_id, cursor_time, user_answer, platform.
    """
    try:
        ts_ms = int(row.get("timestamp", 0))
        action_type = str(row.get("action_type", "enter")).strip().lower()
        item_id = str(row.get("item_id", "")).strip()
        user_ans = str(row.get("user_answer", "")).strip().lower() if row.get("user_answer") else None
        platform = str(row.get("platform", "")).strip() or None

        event_id = f"evt_kt4_{student_id}_{ts_ms}_{source_row}"
        event = Event(
            event_id=event_id,
            student_id=student_id,
            event_ts_ms=ts_ms,
            source_row=source_row,
            event_type=action_type,
            item_id=item_id,
            response=user_ans,
            platform=platform,
            provenance=ProvenanceEnum.OBSERVED
        )

        attempt = None
        if action_type == "respond" and item_id.startswith("q"):
            is_correct = None
            if correct_answers_map and item_id in correct_answers_map:
                is_correct = (user_ans == correct_answers_map[item_id].strip().lower())
            attempt = Attempt(
                attempt_id=f"{student_id}_{ts_ms}_{item_id}",
                student_id=student_id,
                question_id=item_id,
                bundle_instance_id=f"{student_id}_kt4",
                attempt_index=1,
                observed_ts_ms=ts_ms,
                is_correct=is_correct,
                elapsed_ms=None
            )

        return attempt, event
    except Exception:
        return None, None

def adapt_ednet_row(
    row: Dict[str, Any],
    student_id: str,
    source_row: int,
    tier: Optional[str] = None,
    correct_answers_map: Optional[Dict[str, str]] = None
) -> Tuple[Optional[Attempt], Optional[Event]]:
    """
    Unified adapter automatically delegating across KT1, KT2, KT3, and KT4 rows.
    """
    if tier == "KT1" or "solving_id" in row or "elapsed_time" in row:
        return adapt_kt1_row(row, student_id=student_id, source_row=source_row, correct_answers_map=correct_answers_map)
    elif tier == "KT3" or ("action_type" in row and str(row.get("item_id", "")).startswith("e")):
        return adapt_kt3_row(row, student_id=student_id, source_row=source_row)
    elif tier == "KT2":
        return adapt_kt2_row(row, student_id=student_id, source_row=source_row, correct_answers_map=correct_answers_map)
    elif tier == "KT4" or "cursor_time" in row:
        return adapt_kt4_row(row, student_id=student_id, source_row=source_row, correct_answers_map=correct_answers_map)
    else:
        return adapt_kt1_row(row, student_id=student_id, source_row=source_row, correct_answers_map=correct_answers_map)

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
