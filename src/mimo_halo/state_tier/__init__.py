"""NVMe state-tier package.

Owned by the state-store module: ``store.py`` implements the blocking opaque
put/get/inspect CLI (documented in docs/state-store.md). The retention policy
module (``retention.py``) is an independent sibling; this package performs no
imports at package-load time so either module can be exercised alone.
"""
