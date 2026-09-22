"""Fail-closed error hierarchy for candidate construction (mimo_halo.build).

Every refusal path in the construction pipeline raises a subclass of
:class:`BuildError`. The named subclasses are the refusal table of
``tests/test_build.py``: callers catch the specific condition, never a
generic failure, and nothing in this package warns-and-continues.
"""


class BuildError(Exception):
    """Base fail-closed error for candidate construction."""


class SourcePathError(BuildError):
    """An input path escaped the verified source root, or source bytes failed
    verification against the pinned inventory (refusal: any input path
    outside the verified source root)."""


class RequantError(BuildError):
    """Second-generation quantization was attempted on source values without
    an explicit recorded override in the recipe's ``second_gen_applied``
    ledger (refusal: requantization of an mxfp4 tensor without an explicit
    recorded override)."""


class PassthroughQuantError(BuildError):
    """Second-generation quantization was routed at a recipe class marked
    passthrough (``second_gen=false``/``bit_exact=true``), or the
    ``second_gen_applied`` ledger names such a class (refusal: second-gen
    quant silently applied to a recipe class marked passthrough)."""


class AccountingError(BuildError):
    """Byte accounting did not close exactly against the recipe's predicted
    size (refusal: byte accounting that does not close exactly)."""


class PackingEvidenceError(BuildError):
    """An mxfp4 tensor was dispatched without its ``weight_scale`` sibling in
    the source shard (refusal: missing packing-evidence sibling)."""
