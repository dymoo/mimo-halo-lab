"""Model artifact inventory for MiMo-V2.6-Flash-RL.

Public entry point: the :mod:`mimo_halo.models.inventory` CLI
(``python -m mimo_halo.models.inventory fetch|inventory|memory``).
No names are re-exported here: eagerly importing the submodule during
package init made ``python -m mimo_halo.models.inventory`` execute it twice
and emit a runpy RuntimeWarning on every run, and nothing imports these
names from the package root.
"""
