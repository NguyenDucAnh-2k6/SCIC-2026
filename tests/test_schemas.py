from src.classtwin.data.schemas import Attempt, Event, ContentItem, TeacherDecision, ProvenanceEnum
from src.classtwin.data.adapters import adapt_kt1_row, adapt_question_row, extract_user_id_from_path

def test_extract_user_id():
    path = "KT1/u246317.csv"
    assert extract_user_id_from_path(path) == "u246317"

def test_adapt_kt1_row():
    raw_row = {
        "timestamp": "1535196434651",
        "solving_id": "1",
        "question_id": "q8098",
        "user_answer": "b",
        "elapsed_time": "20000"
    }
    ans_map = {"q8098": "b"}
    attempt, event = adapt_kt1_row(raw_row, student_id="u246317", source_row=0, correct_answers_map=ans_map)
    assert attempt is not None
    assert attempt.student_id == "u246317"
    assert attempt.is_correct is True
    assert attempt.elapsed_ms == 20000
    assert event.event_type == "respond"

def test_adapt_question_row():
    raw_row = {
        "question_id": "q8098",
        "bundle_id": "b123",
        "correct_answer": "b",
        "part": "5",
        "tags": "12;45;78"
    }
    item = adapt_question_row(raw_row)
    assert item is not None
    assert item.item_id == "q8098"
    assert item.skill_ids == [12, 45, 78]
    assert item.correct_answer == "b"
