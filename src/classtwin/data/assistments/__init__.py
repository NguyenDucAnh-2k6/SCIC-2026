"""
ASSISTments Dataset Package for ClassTwin:
Dedicated modules for parsing 250 real classroom cohorts, teacher structures, and student baseline vectors.
"""

from .adapter import AssistmentsAdapter
from .extractor import ClassroomCohortExtractor

__all__ = ["AssistmentsAdapter", "ClassroomCohortExtractor"]

