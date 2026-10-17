from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ProvenanceEnum(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    INFERRED = "inferred"
    SIMULATED = "simulated"

class ContentItem(BaseModel):
    item_id: str
    item_type: str = "question"  # "question" or "lecture"
    bundle_id: Optional[str] = None
    skill_ids: List[int] = Field(default_factory=list)
    part: Optional[int] = None
    correct_answer: Optional[str] = None
    available_at_ms: Optional[int] = None

class Event(BaseModel):
    event_id: str
    student_id: str
    event_ts_ms: int
    source_row: int
    event_type: str = "respond"  # "respond", "enter", "quit"
    item_id: Optional[str] = None
    response: Optional[str] = None
    source: Optional[str] = None
    platform: Optional[str] = None
    provenance: ProvenanceEnum = ProvenanceEnum.OBSERVED

class Attempt(BaseModel):
    attempt_id: str
    student_id: str
    question_id: str
    bundle_instance_id: str
    attempt_index: int = 1
    observed_ts_ms: int
    answered_at_ms: Optional[int] = None
    is_correct: Optional[bool] = None
    elapsed_ms: Optional[int] = None
    timing_semantics: str = "question_level"
    quality_flags: List[str] = Field(default_factory=list)

class StudentSession(BaseModel):
    session_id: str
    student_id: str
    start_ms: int
    end_ms: int
    boundary_reason: str = "gap_gt_30m"
    split: str = "train"

class TeacherDecision(BaseModel):
    decision_id: str
    class_id: str
    state_ref: str
    suggested_action: str
    executed_action: Optional[str] = None
    planned_duration_s: int
    actual_duration_s: Optional[int] = None
    accepted_status: str = "PENDING"  # "ACCEPTED", "MODIFIED", "REJECTED"
    override_reason: Optional[str] = None
    policy_version: str = "cql_v1"
    support_score: float = 1.0

class TransitionRecord(BaseModel):
    episode_id: str
    step: int
    state_ref: str
    executed_action: str
    reward: Optional[float] = None
    reward_components: dict = Field(default_factory=dict)
    next_state_ref: str
    duration_s: int
    terminated: bool = False
    truncated: bool = False
    data_origin: ProvenanceEnum = ProvenanceEnum.SIMULATED
    simulator_version: Optional[str] = "classtwin_sim_v1"
