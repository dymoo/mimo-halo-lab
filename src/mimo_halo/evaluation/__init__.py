"""Evaluation subpackage: paired task-outcome statistical analysis.

Statistics over JSONL task-outcome records produced by external runners.
No inference, no sandbox, no runner logic lives here.
"""

from typing import Any

__all__ = ["EvaluationError", "analyze", "exact_mcnemar_p", "load_records", "main"]


def __getattr__(name: str) -> Any:
    # Lazy import so `python -m mimo_halo.evaluation.paired` does not
    # import the module twice (which emits a RuntimeWarning).
    if name in __all__:
        from mimo_halo.evaluation import paired

        return getattr(paired, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")