"""Read-only harness trace discovery, normalization, partitioning and golden eligibility.

Privacy contract:
- Raw harness transcripts are read at their origin and never copied or rewritten.
- Public artifacts (repo manifests) carry only hashes, counts and schema facts.
- Private artifacts go to an explicit output root outside the Git repository and
  contain credential-redacted, bounded text; this is redaction, not a claim of
  complete anonymization.
"""

from .common import TraceError
from .discovery import discover
from .session import normalize_sessions
from .identity import build_task_groups
from .partition import partition_tasks
from .golden import evaluate_golden_eligibility

__all__ = [
    "TraceError",
    "discover",
    "normalize_sessions",
    "build_task_groups",
    "partition_tasks",
    "evaluate_golden_eligibility",
]
