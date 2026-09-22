"""Baseline modules for published compression checkpoints.

Each module archives small public metadata for one third-party baseline,
normalizes it to a stable ``schema_version=1`` JSON contract, and exposes
a CLI. Normalized baselines are consumed by
``mimo_halo.baselines.compare`` (owned separately); importers never edit
shared interfaces.
"""