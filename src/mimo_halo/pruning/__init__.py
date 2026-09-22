"""Prune-map planning for the MiMo MoE expert stack.

Public entry point is :func:`main`, a dry-run planning CLI::

    PYTHONPATH=src python -m mimo_halo.pruning.maps \
        --inventory inventory.json --retained-count 160 --seed 7 \
        --output map.json --candidate-out candidate.json

or, for an external actual selection (original expert ids, order preserved)::

    PYTHONPATH=src python -m mimo_halo.pruning.maps \
        --inventory inventory.json --selection candidate.json --output map.json

The CLI is planning-only: it never reads weights, never slices tensors, and
never claims checkpoint reload. Generated selections are deterministic
shape-only placeholders explicitly marked as NOT REAP/HOPE quality maps.
Consumes inventory JSON (``schema_version=1``) directly; imports no
inventory-module internals.

``mimo_halo.pruning.maps`` is the runnable module; names are exposed here
lazily (PEP 562) so ``python -m`` execution stays warning-free.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .maps import (
        EXPERT_PROJECTIONS,
        SCHEMA_VERSION,
        PruneMapError,
        build_prune_map,
        collect_moe_structure,
        generate_selection,
        load_inventory,
        load_selection_file,
        main,
        parse_external_selection,
    )

__all__ = [
    "EXPERT_PROJECTIONS",
    "SCHEMA_VERSION",
    "PruneMapError",
    "build_prune_map",
    "collect_moe_structure",
    "generate_selection",
    "load_inventory",
    "load_selection_file",
    "main",
    "parse_external_selection",
]


def __getattr__(name: str):
    if name in __all__:
        from . import maps

        return getattr(maps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
