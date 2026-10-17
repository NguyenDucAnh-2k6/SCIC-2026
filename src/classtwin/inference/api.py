from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from .state_store import ClassroomStateStore
from .engine import DecisionSupportEngine
from ..twin.classroom_env import ClassroomDigitalTwinEnv
from ..twin.scenarios import counterfactual_twin_rollout

app = FastAPI(
    title="ClassTwin REST API",
    description="Online Decision Support System for Educators • SCIC 2026",
    version="1.0.0"
)

# Global singleton runtime state
store = ClassroomStateStore()
engine = DecisionSupportEngine(state_store=store)

class EventBatchRequest(BaseModel):
    events: List[Dict[str, Any]]
    schema_version: str = "v1.0"

class RecommendationRequest(BaseModel):
    class_id: str
    topic: str = "general"
    time_remaining_minutes: float = 45.0
    current_workload: float = 3.0
    run_simulation: bool = True

class TeacherDecisionRequest(BaseModel):
    decision_id: str
    status: str = Field(..., description="'ACCEPTED', 'MODIFIED', or 'REJECTED'")
    executed_action: Optional[str] = None
    actual_duration_s: Optional[int] = None
    override_reason: Optional[str] = None

class SimulationRequest(BaseModel):
    class_id: str
    candidate_actions: List[str]
    duration_minutes: float = 5.0
    num_samples: int = 10

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "classtwin_api", "version": "1.0.0"}

@app.post("/v1/events:batch")
def ingest_events_batch(req: EventBatchRequest):
    accepted_ids = []
    for ev in req.events:
        student_id = ev.get("student_id")
        class_id = ev.get("class_id", "default_class")
        if student_id:
            # Simulate state update
            is_corr = ev.get("is_correct", True)
            curr = store.get_or_create_student_state(class_id, student_id)
            prev_m = curr.get("mastery_estimate", 0.5)
            new_m = min(1.0, max(0.0, prev_m + (0.05 if is_corr else -0.05)))
            store.update_student_state(
                class_id=class_id,
                student_id=student_id,
                hidden=curr["hidden"],
                ts_ms=ev.get("event_ts_ms", 0),
                mastery=new_m
            )
            accepted_ids.append(ev.get("event_id", f"ev_{student_id}"))
    return {
        "status": "success",
        "accepted_count": len(accepted_ids),
        "accepted_ids": accepted_ids
    }

@app.get("/v1/classes/{class_id}/state")
def get_class_state(class_id: str):
    summary = store.get_class_summary(class_id)
    return summary

@app.post("/v1/recommendations")
def get_recommendation(req: RecommendationRequest):
    rec = engine.generate_recommendation(
        class_id=req.class_id,
        topic=req.topic,
        time_remaining_minutes=req.time_remaining_minutes,
        current_workload=req.current_workload,
        run_simulation_rollout=req.run_simulation
    )
    return rec

@app.post("/v1/teacher-decisions")
def record_teacher_decision(req: TeacherDecisionRequest):
    dec = store.get_decision(req.decision_id)
    if not dec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Decision ID {req.decision_id} not found."
        )
    dec["teacher_status"] = req.status
    dec["executed_action"] = req.executed_action or dec.get("recommended_action")
    dec["actual_duration_s"] = req.actual_duration_s or dec.get("planned_duration_s")
    dec["override_reason"] = req.override_reason
    store.record_decision(req.decision_id, dec)
    return {
        "status": "recorded",
        "decision_id": req.decision_id,
        "recorded_action": dec["executed_action"]
    }

@app.post("/v1/simulations")
def simulate_counterfactuals(req: SimulationRequest):
    summary = store.get_class_summary(req.class_id)
    env = ClassroomDigitalTwinEnv(cohort_size=summary["n_enrolled"], seed=42)
    env.reset()

    results = []
    for act in req.candidate_actions:
        stats = counterfactual_twin_rollout(
            base_env=env,
            candidate_action=act,
            duration_minutes=req.duration_minutes,
            num_mc_samples=req.num_samples
        )
        results.append(stats)

    return {
        "class_id": req.class_id,
        "simulations": results
    }
