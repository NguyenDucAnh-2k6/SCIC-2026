"""
MultiSourceDataManager Facade:
Re-exports the unified DataOrchestrator from classtwin.data.orchestrator.
"""

from classtwin.data.orchestrator import DataOrchestrator, MultiSourceDataManager

__all__ = ["DataOrchestrator", "MultiSourceDataManager"]
