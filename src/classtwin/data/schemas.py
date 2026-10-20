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

class DatasetRoleEnum(str, Enum):
    EDNET_KT1_FOUNDATION = "ednet_kt1_foundation"         # Knowledge Tracing representation & response dynamics
    EDNET_KT2_KT3_ATTENTION = "ednet_kt2_kt3_attention"   # Video/lecture consumption & attention decay
    EDNET_KT4_FATIGUE = "ednet_kt4_fatigue"               # Session fatigue & cognitive break dynamics
    ASSISTMENTS_CLASSROOM = "assistments_classroom"       # Real teacher-class cohorts, hints & scaffolding

class AssistmentsRecord(BaseModel):
    order_id: int
    assignment_id: int
    user_id: int
    assistment_id: Optional[int] = None
    problem_id: int
    original: Optional[int] = 1
    correct: int = Field(description="1: correct on first attempt, 0: incorrect")
    attempt_count: int = Field(default=1, description="Number of attempts on this problem")
    ms_first_response: Optional[int] = Field(default=None, description="Latency to first response in ms")
    tutor_mode: Optional[str] = None
    answer_type: Optional[str] = None
    sequence_id: Optional[int] = None
    student_class_id: int = Field(description="Classroom cohort identifier")
    position: Optional[int] = None
    type: Optional[str] = None
    base_sequence_id: Optional[int] = None
    skill_id: Optional[int] = None
    skill_name: Optional[str] = None
    teacher_id: int = Field(description="Teacher responsible for the cohort")
    school_id: Optional[int] = None
    hint_count: int = Field(default=0, description="Number of hints requested by student")
    hint_total: Optional[int] = 0
    overlap_time: Optional[int] = None
    template_id: Optional[int] = None
    answer_id: Optional[int] = None
    answer_text: Optional[str] = None
    first_action: Optional[int] = None
    bottom_hint: Optional[int] = None
    opportunity: Optional[int] = None
    opportunity_original: Optional[int] = None

class ClassroomCohort(BaseModel):
    class_id: int
    teacher_id: int
    student_ids: List[int] = Field(default_factory=list)
    student_count: int
    skill_ids: List[int] = Field(default_factory=list)
    skill_names: List[str] = Field(default_factory=list)
    total_attempts: int = 0
    mean_first_attempt_accuracy: float = 0.0
    mean_hints_per_problem: float = 0.0
    mean_response_time_ms: float = 0.0

