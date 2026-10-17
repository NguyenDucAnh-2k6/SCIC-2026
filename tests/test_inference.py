from src.classtwin.inference.state_store import ClassroomStateStore
from src.classtwin.inference.engine import DecisionSupportEngine

def test_state_store_and_fallback():
    store = ClassroomStateStore()
    engine = DecisionSupportEngine(state_store=store, min_coverage_threshold=0.60)

    # Empty class -> Low coverage -> triggers fallback safely
    rec = engine.generate_recommendation(
        class_id="class_test_101",
        topic="calculus",
        time_remaining_minutes=40.0,
        run_simulation_rollout=False
    )
    assert rec["evidence_level"] == "heuristic_fallback"
    assert rec["requires_teacher_decision"] is True
    assert "decision_id" in rec

def test_engine_with_active_students():
    store = ClassroomStateStore()
    engine = DecisionSupportEngine(state_store=store, min_coverage_threshold=0.50)

    # Add 40 active students to satisfy coverage
    import numpy as np
    for i in range(40):
        store.update_student_state(
            class_id="class_active_202",
            student_id=f"u_{i}",
            hidden=np.random.randn(64).astype(np.float32),
            ts_ms=1000 + i,
            mastery=0.75
        )

    rec = engine.generate_recommendation(
        class_id="class_active_202",
        topic="algorithms",
        time_remaining_minutes=30.0,
        run_simulation_rollout=True
    )
    assert rec["coverage"] > 0.50
    assert "simulation_gain" in rec
    assert rec["latency_ms"] >= 0.0
