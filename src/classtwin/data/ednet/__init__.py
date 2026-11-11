"""
EdNet Dataset Package for ClassTwin:
Dedicated modules for reading, streaming, adapting, preprocessing, and extracting DKT sequence features.
"""

from .loader import StreamingZipReader
from .adapter import adapt_kt1_row, adapt_question_row, extract_user_id_from_path
from .preprocessor import assign_student_split, ShardWriter
from .extractor import (
    FastSequenceDataset,
    SequenceDataset,
    create_sequence_dataloader,
    load_sequences_from_shards,
    deterministic_irt_label,
    compute_as_of_features,
    get_ednet_behavioral_priors
)

__all__ = [
    "StreamingZipReader",
    "adapt_kt1_row",
    "adapt_question_row",
    "extract_user_id_from_path",
    "assign_student_split",
    "ShardWriter",
    "FastSequenceDataset",
    "SequenceDataset",
    "create_sequence_dataloader",
    "load_sequences_from_shards",
    "deterministic_irt_label",
    "compute_as_of_features",
    "get_ednet_behavioral_priors"
]

