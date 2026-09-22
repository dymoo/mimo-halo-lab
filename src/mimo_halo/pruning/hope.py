"""HOPE: Higher-order Pruning of Experts — stdlib statistics, exact selector, CLI.

Implements the objective of Tseng, Kaul, Zancano, Xia, Soatto, "Higher-order
pruning of experts in mixture-of-experts language models", arXiv:2609.18916v1
(2026-09-16), restricted to what is verifiable from that paper:

* first-order score (REAP connection, §2.2): ``S_REAP = mean_over_active(g_k * ||f_k||)``
* interaction matrix (Eq. 3 / Appendix A.7, conditional normalization):
  ``F_ij = (1/|X_ij|) * sum_{x in X_ij} g_i g_j ||f_i|| ||f_j||`` with
  ``X_ij = {x : i,j both routed on x}`` and ``F_ij = 0`` when ``X_ij`` is empty.
  The diagonal is therefore ``E[(g_k ||f_k||)^2]`` (contribution variance kept),
  which differs from REAP's squared ``E[g_k ||f_k||]`` — this is the paper's
  stated distinction (§3.5, Fig. S4).
* prune-set QP (Theorem 2): ``min_{p in {0,1}^E, sum p = |P|} p^T F p`` where
  ``p_k = 1`` means expert k is PRUNED.

Solver policy (documented in docs/hope-objective.md):

* ``exhaustive`` — complete enumeration of all C(E,|P|) fixed-size sets.
  Exact global optimum, returned with a certificate, for 8 <= E <= 16
  (the toy-oracle range required by the project contract; smaller E is also
  correct, larger E is refused rather than silently approximated).
* ``scipy-slsqp-relaxation-top-round`` — continuous relaxation on the capped
  simplex + top-|P| rounding, mirroring the paper's §3.6 recipe ("a standard
  QP solver", then round), with seeded restarts and an optional deterministic
  binary 1-swap refinement. ALWAYS labeled ``exact=false`` /
  ``global_optimal=false``. F is entrywise non-negative but NOT assumed PSD.
  Requires scipy; a missing dependency raises :class:`HopeDependencyError`
  with the install command — there is never a silent fallback between
  objectives.

Statistics contract: one :class:`HopeStats` per MoE layer accumulates routed
selected-expert ids, gate weights, and expert output norms from real batches
(see reap_adapter.py for the REAP one-pass hook). Accumulators are plain
Python floats/ints: additive across batch boundaries and process resumes
(:meth:`HopeStats.merge`), deterministically serializable, and validated on
load by count-safe invariants (pair-count symmetry, diagonal == activation
count, Fréchet co-activation bounds, gate bounds, finiteness).

CLI (actual algorithms, no mock results)::

    PYTHONPATH=src python -m mimo_halo.pruning.hope observe-fixture \
        --config configs/hope.json --out scratch/hope-fixture
    PYTHONPATH=src python -m mimo_halo.pruning.hope select \
        --config configs/hope.json \
        --stats scratch/hope-fixture/uniform.json --budget 4 --out selection.json
    PYTHONPATH=src python -m mimo_halo.pruning.hope toys \
        --config configs/hope.json
    PYTHONPATH=src python -m mimo_halo.pruning.hope merge \
        --out merged.json part1.json part2.json

Telemetry: this module performs no network I/O of any kind.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = 1
PAPER_REF = "arXiv:2609.18916v1"
PAPER_TITLE = (
    "Higher-order pruning of experts in mixture-of-experts language models"
)
STATS_KIND = "hope-stats"
SELECTION_KIND = "hope-selection"
TOYS_KIND = "hope-toys"
EXACT_MAX_EXPERTS = 16
RELAXATION_METHOD = "scipy-slsqp-relaxation-top-round"
EXHAUSTIVE_METHOD = "exhaustive"
SCIPY_INSTALL_HINT = (
    "scipy is an optional dependency of the HOPE relaxation selector; "
    "install it with: python -m pip install 'scipy>=1.14'"
)
GATE_MAX_SLACK = 1e-6


class HopeError(Exception):
    """Base class for every HOPE failure."""


class HopeValidationError(HopeError):
    """A statistic, count, id, gate, norm, or document failed validation."""


class HopeDependencyError(HopeError):
    """An explicitly required optional dependency is unavailable."""


class HopeSelectionError(HopeError):
    """A selection request is out of contract (size, budget, method)."""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _require_int(value: object, where: str) -> int:
    if type(value) is not int:
        raise HopeValidationError(f"{where}: expected int, got {type(value).__name__}")
    return value


def _require_count(value: object, where: str) -> int:
    got = _require_int(value, where)
    if got < 0:
        raise HopeValidationError(f"{where}: count must be >= 0, got {got}")
    return got


def _require_float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HopeValidationError(
            f"{where}: expected number, got {type(value).__name__}"
        )
    got = float(value)
    if not math.isfinite(got):
        raise HopeValidationError(f"{where}: value must be finite, got {got!r}")
    return got


class HopeStats:
    """Per-layer HOPE accumulators over routed (selected-id, gate, norm) rows.

    Every valid routing row contributes exactly ``top_k`` selections.  Fields:

    ``total_rows``
        number of accumulated routing rows (valid, non-padding tokens).
    ``act_count[k]`` / ``pair_count[i][j]``
        activation counts; ``pair_count`` diagonal equals ``act_count`` and the
        off-diagonal entries are TRUE co-activation counts ``|X_ij|`` — never
        the marginal outer-sum that REAP's ``pairwise_expert_frequency``
        stores (those two are not interchangeable).
    ``pair_sum[i][j]``
        ``sum_{x in X_ij} a_i a_j`` with ``a_k = g_k * ||f_k||`` (diagonal:
        ``sum a_k^2``).
    ``first_sum[k]``, ``gate_sum[k]``, ``norm_sum[k]``
        sums over rows where k was selected.

    Capability partitions (``capabilities``) accumulate full sub-statistics for
    labeled subsets of rows; unlabeled rows only exist at the top level, so
    every capability count is bounded by its parent (count-safe on load).
    """

    __slots__ = (
        "n_experts",
        "top_k",
        "total_rows",
        "act_count",
        "pair_count",
        "pair_sum",
        "first_sum",
        "gate_sum",
        "norm_sum",
        "capabilities",
    )

    def __init__(self, n_experts: int, top_k: int | None = None) -> None:
        if type(n_experts) is not int or n_experts < 1:
            raise HopeValidationError(f"n_experts must be a positive int, got {n_experts!r}")
        if top_k is not None:
            if type(top_k) is not int or not 1 <= top_k <= n_experts:
                raise HopeValidationError(
                    f"top_k must be an int in [1, {n_experts}], got {top_k!r}"
                )
        self.n_experts = n_experts
        self.top_k = top_k
        self.total_rows = 0
        e = n_experts
        self.act_count = [0] * e
        self.pair_count = [[0] * e for _ in range(e)]
        self.pair_sum = [[0.0] * e for _ in range(e)]
        self.first_sum = [0.0] * e
        self.gate_sum = [0.0] * e
        self.norm_sum = [0.0] * e
        self.capabilities: dict[str, HopeStats] = {}

    # -- accumulation -------------------------------------------------------

    def add_batch(
        self,
        selected: Sequence[Sequence[int]],
        gates: Sequence[Sequence[float]],
        norms: Sequence[Sequence[float]],
        *,
        capability: str | None = None,
    ) -> None:
        """Accumulate one routed batch.

        ``selected[t][r]`` is the routed expert id at row t, position r;
        ``gates`` and ``norms`` are aligned with it.  Validation is strict:
        fixed ``top_k`` per row, ids int and unique within a row and inside
        ``[0, n_experts)``, gates finite in ``[0, 1+slack]``, norms finite and
        non-negative.  This rejects the failure modes of recording raw-logit
        top-k instead of actual routed ids (out-of-range/duplicated ids) and
        of feeding unnormalized gates.
        """
        if capability is not None:
            if not isinstance(capability, str) or not capability:
                raise HopeValidationError("capability must be a non-empty string")
        rows = len(selected)
        if len(gates) != rows or len(norms) != rows:
            raise HopeValidationError(
                f"batch shape mismatch: {rows} selected rows, "
                f"{len(gates)} gate rows, {len(norms)} norm rows"
            )
        if rows == 0:
            return
        width = self.top_k
        if width is None:
            first_len = len(selected[0])
            if type(first_len) is not int or not 1 <= first_len <= self.n_experts:
                raise HopeValidationError(
                    f"row width must be in [1, {self.n_experts}], got {first_len!r}"
                )
            width = first_len
        # Phase 1: validate the WHOLE batch before touching any accumulator so
        # a rejected batch leaves the statistics completely unchanged.
        prepared: list[tuple[list[int], list[float], list[float], list[float]]] = []
        for t in range(rows):
            row = selected[t]
            gate_row = gates[t]
            norm_row = norms[t]
            if len(row) != width or len(gate_row) != width or len(norm_row) != width:
                raise HopeValidationError(
                    f"row {t}: expected width top_k={width}, got "
                    f"selected={len(row)} gates={len(gate_row)} norms={len(norm_row)}"
                )
            ids: list[int] = []
            a_vals: list[float] = []
            gate_vals: list[float] = []
            norm_vals: list[float] = []
            for r in range(width):
                expert = row[r]
                if type(expert) is not int:
                    raise HopeValidationError(
                        f"row {t} pos {r}: expert id must be int, got {type(expert).__name__}"
                    )
                if not 0 <= expert < self.n_experts:
                    raise HopeValidationError(
                        f"row {t} pos {r}: expert id {expert} outside [0, {self.n_experts})"
                    )
                if expert in ids:
                    raise HopeValidationError(
                        f"row {t}: duplicate expert id {expert} (not actual routed ids?)"
                    )
                ids.append(expert)
                gate = _require_float(gate_row[r], f"row {t} pos {r} gate")
                if gate < 0.0 or gate > 1.0 + GATE_MAX_SLACK:
                    raise HopeValidationError(
                        f"row {t} pos {r}: gate {gate} outside [0, 1]"
                    )
                norm = _require_float(norm_row[r], f"row {t} pos {r} norm")
                if norm < 0.0:
                    raise HopeValidationError(
                        f"row {t} pos {r}: expert output norm {norm} must be >= 0"
                    )
                a_vals.append(gate * norm)
                gate_vals.append(gate)
                norm_vals.append(norm)
            prepared.append((ids, a_vals, gate_vals, norm_vals))
        # Phase 2: commit (validation passed for every row).
        if self.top_k is None:
            self.top_k = width
        cap = None
        if capability is not None:
            cap = self.capabilities.get(capability)
            if cap is None:
                cap = HopeStats(self.n_experts, width)
                self.capabilities[capability] = cap
        for ids, a_vals, gate_vals, norm_vals in prepared:
            self._accumulate_row(ids, a_vals, gate_vals, norm_vals)
            self.total_rows += 1
            if cap is not None:
                cap._accumulate_row(ids, a_vals, gate_vals, norm_vals)
                cap.total_rows += 1

    def _accumulate_row(
        self, ids: list[int], a_vals: list[float], gates: list[float], norms: list[float]
    ) -> None:
        k = len(ids)
        for r in range(k):
            expert = ids[r]
            self.act_count[expert] += 1
            self.pair_count[expert][expert] += 1
            a = a_vals[r]
            self.pair_sum[expert][expert] += a * a
            self.first_sum[expert] += a
            self.gate_sum[expert] += gates[r]
            self.norm_sum[expert] += norms[r]
        for r in range(k):
            i = ids[r]
            ai = a_vals[r]
            for s in range(r + 1, k):
                j = ids[s]
                product = ai * a_vals[s]
                # Symmetric accumulation keeps pair_count/pair_sum exactly
                # symmetric — an invariant checked on load.
                self.pair_count[i][j] += 1
                self.pair_count[j][i] += 1
                self.pair_sum[i][j] += product
                self.pair_sum[j][i] += product

    def merge(self, other: HopeStats) -> HopeStats:
        """Add another accumulator of the same layer (streaming / resume)."""
        if not isinstance(other, HopeStats):
            raise HopeValidationError(f"cannot merge {type(other).__name__}")
        other.validate()
        if other.n_experts != self.n_experts:
            raise HopeValidationError(
                f"merge n_experts mismatch: {self.n_experts} vs {other.n_experts}"
            )
        if self.top_k is None:
            self.top_k = other.top_k
        elif other.top_k is not None and other.top_k != self.top_k:
            raise HopeValidationError(
                f"merge top_k mismatch: {self.top_k} vs {other.top_k}"
            )
        self.total_rows += other.total_rows
        for i in range(self.n_experts):
            self.act_count[i] += other.act_count[i]
            self.first_sum[i] += other.first_sum[i]
            self.gate_sum[i] += other.gate_sum[i]
            self.norm_sum[i] += other.norm_sum[i]
            row_pc = self.pair_count[i]
            row_ps = self.pair_sum[i]
            row_pc_o = other.pair_count[i]
            row_ps_o = other.pair_sum[i]
            for j in range(self.n_experts):
                row_pc[j] += row_pc_o[j]
                row_ps[j] += row_ps_o[j]
        for name, sub in other.capabilities.items():
            existing = self.capabilities.get(name)
            if existing is None:
                self.capabilities[name] = sub.deepcopy()
            else:
                existing.merge(sub)
        self.validate()
        return self

    def deepcopy(self) -> HopeStats:
        clone = HopeStats(self.n_experts, self.top_k)
        clone.total_rows = self.total_rows
        clone.act_count = list(self.act_count)
        clone.pair_count = [list(row) for row in self.pair_count]
        clone.pair_sum = [list(row) for row in self.pair_sum]
        clone.first_sum = list(self.first_sum)
        clone.gate_sum = list(self.gate_sum)
        clone.norm_sum = list(self.norm_sum)
        for name, sub in self.capabilities.items():
            clone.capabilities[name] = sub.deepcopy()
        return clone

    # -- derived quantities -------------------------------------------------

    def routing_frequency(self) -> list[float]:
        """P(expert active) = act_count / total_rows (REAP's expert_frequency/total)."""
        if self.total_rows == 0:
            return [0.0] * self.n_experts
        rows = self.total_rows
        return [count / rows for count in self.act_count]

    def first_order(self) -> list[float]:
        """REAP first-order score: mean over ACTIVE rows of ``g_k * ||f_k||``.

        This is ``S_REAP = (1/|X_k|) sum g_k ||f_k||`` from §2.2 and matches
        the pinned REAP implementation's per-expert ``reap`` metric semantics
        (mean of ``ean_norm * active_router_weight`` over active tokens).
        """
        return [
            self.first_sum[k] / self.act_count[k] if self.act_count[k] else 0.0
            for k in range(self.n_experts)
        ]

    def conditional_f(self) -> list[list[float]]:
        """Conditional-normalized interaction matrix F (Eq. 3 / A.7).

        ``F_ij = pair_sum_ij / pair_count_ij`` where the pair co-activated,
        else 0.  Entrywise non-negative; symmetric; NOT guaranteed PSD.
        """
        e = self.n_experts
        f = [[0.0] * e for _ in range(e)]
        for i in range(e):
            row_in = self.pair_count[i]
            row_out = f[i]
            row_sum = self.pair_sum[i]
            for j in range(e):
                count = row_in[j]
                if count:
                    row_out[j] = row_sum[j] / count
        return f

    def mean_gates(self) -> list[float]:
        return [
            self.gate_sum[k] / self.act_count[k] if self.act_count[k] else 0.0
            for k in range(self.n_experts)
        ]

    def mean_norms(self) -> list[float]:
        return [
            self.norm_sum[k] / self.act_count[k] if self.act_count[k] else 0.0
            for k in range(self.n_experts)
        ]

    # -- count-safe validation ----------------------------------------------

    def validate(self) -> None:
        """Check every count/sum invariant; raise HopeValidationError on failure."""
        e = self.n_experts
        if type(e) is not int or e < 1:
            raise HopeValidationError(f"n_experts invalid: {e!r}")
        if self.top_k is not None:
            if type(self.top_k) is not int or not 1 <= self.top_k <= e:
                raise HopeValidationError(f"top_k invalid: {self.top_k!r}")
        rows = _require_count(self.total_rows, "total_rows")
        if rows and self.top_k is None:
            raise HopeValidationError("total_rows > 0 but top_k unset")
        if sum(self.act_count) != rows * (self.top_k or 0):
            raise HopeValidationError(
                "count mismatch: sum(act_count) "
                f"{sum(self.act_count)} != total_rows*top_k {rows * (self.top_k or 0)}"
            )
        for k in range(e):
            act = _require_count(self.act_count[k], f"act_count[{k}]")
            if act > rows:
                raise HopeValidationError(f"act_count[{k}]={act} exceeds total_rows={rows}")
            for name, values in (
                ("first_sum", self.first_sum),
                ("gate_sum", self.gate_sum),
                ("norm_sum", self.norm_sum),
            ):
                value = _require_float(values[k], f"{name}[{k}]")
                if value < 0.0:
                    raise HopeValidationError(f"{name}[{k}] must be >= 0, got {value}")
            gate_bound = act * (1.0 + GATE_MAX_SLACK) + 1e-9
            if self.gate_sum[k] > gate_bound:
                raise HopeValidationError(
                    f"gate_sum[{k}]={self.gate_sum[k]} exceeds act_count[{k}]*1 bound"
                )
        for i in range(e):
            if self.pair_count[i][i] != self.act_count[i]:
                raise HopeValidationError(
                    f"pair_count diagonal [{i}][{i}]={self.pair_count[i][i]} "
                    f"!= act_count[{i}]={self.act_count[i]}"
                )
            for j in range(i, e):
                count_ij = _require_count(self.pair_count[i][j], f"pair_count[{i}][{j}]")
                count_ji = _require_count(self.pair_count[j][i], f"pair_count[{j}][{i}]")
                if count_ij != count_ji:
                    raise HopeValidationError(
                        f"pair_count asymmetric at [{i}][{j}]: {count_ij} != {count_ji}"
                    )
                low = max(0, self.act_count[i] + self.act_count[j] - rows)
                if count_ij < low:
                    raise HopeValidationError(
                        f"pair_count[{i}][{j}]={count_ij} violates Frechet bound {low}"
                    )
                if count_ij > min(self.act_count[i], self.act_count[j]):
                    raise HopeValidationError(
                        f"pair_count[{i}][{j}]={count_ij} exceeds "
                        f"min(act_i, act_j)={min(self.act_count[i], self.act_count[j])}"
                    )
                if i == j:
                    continue
                sum_ij = _require_float(self.pair_sum[i][j], f"pair_sum[{i}][{j}]")
                sum_ji = _require_float(self.pair_sum[j][i], f"pair_sum[{j}][{i}]")
                if sum_ij != sum_ji:
                    raise HopeValidationError(
                        f"pair_sum asymmetric at [{i}][{j}]: {sum_ij} != {sum_ji}"
                    )
                if sum_ij < 0.0:
                    raise HopeValidationError(
                        f"pair_sum[{i}][{j}] must be >= 0 (products of nonneg "
                        f"gate*norm terms), got {sum_ij}"
                    )
                if count_ij == 0 and sum_ij != 0.0:
                    raise HopeValidationError(
                        f"pair_sum[{i}][{j}]={sum_ij} with zero co-activation count"
                    )
            for j in range(e):
                _require_float(self.pair_sum[i][j], f"pair_sum[{i}][{j}]")
        cap_rows = 0
        for name, sub in self.capabilities.items():
            if not isinstance(name, str) or not name:
                raise HopeValidationError("capability names must be non-empty strings")
            sub.validate()
            if sub.n_experts != e:
                raise HopeValidationError(
                    f"capability {name!r} n_experts {sub.n_experts} != {e}"
                )
            if sub.total_rows > rows:
                raise HopeValidationError(
                    f"capability {name!r} rows {sub.total_rows} exceed parent {rows}"
                )
            for k in range(e):
                if sub.act_count[k] > self.act_count[k]:
                    raise HopeValidationError(
                        f"capability {name!r} act_count[{k}]={sub.act_count[k]} "
                        f"exceeds parent {self.act_count[k]}"
                    )
            cap_rows += sub.total_rows
        if cap_rows > rows:
            raise HopeValidationError(
                f"capability rows sum {cap_rows} exceed total_rows {rows}"
            )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": STATS_KIND,
            "paper": PAPER_REF,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "total_rows": self.total_rows,
            "act_count": list(self.act_count),
            "pair_count": [list(row) for row in self.pair_count],
            "pair_sum": [list(row) for row in self.pair_sum],
            "first_sum": list(self.first_sum),
            "gate_sum": list(self.gate_sum),
            "norm_sum": list(self.norm_sum),
        }
        if self.capabilities:
            doc["capabilities"] = {
                name: sub.to_dict() for name, sub in sorted(self.capabilities.items())
            }
        return doc

    @classmethod
    def from_dict(cls, doc: object) -> HopeStats:
        if not isinstance(doc, dict):
            raise HopeValidationError("stats document must be a JSON object")
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise HopeValidationError(
                f"schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSION}"
            )
        if doc.get("kind") != STATS_KIND:
            raise HopeValidationError(f"kind {doc.get('kind')!r} != {STATS_KIND!r}")
        n_experts = _require_int(doc.get("n_experts"), "n_experts")
        top_k = doc.get("top_k")
        if top_k is not None:
            top_k = _require_int(top_k, "top_k")
        stats = cls(n_experts, top_k)
        stats.total_rows = _require_count(doc.get("total_rows"), "total_rows")
        act = doc.get("act_count")
        if not isinstance(act, list) or len(act) != n_experts:
            raise HopeValidationError(
                f"act_count must be a list of length {n_experts}"
            )
        stats.act_count = [_require_count(v, f"act_count[{i}]") for i, v in enumerate(act)]
        pair_count = doc.get("pair_count")
        pair_sum = doc.get("pair_sum")
        if not isinstance(pair_count, list) or len(pair_count) != n_experts:
            raise HopeValidationError(f"pair_count must be {n_experts}x{n_experts}")
        if not isinstance(pair_sum, list) or len(pair_sum) != n_experts:
            raise HopeValidationError(f"pair_sum must be {n_experts}x{n_experts}")
        for i in range(n_experts):
            row_c = pair_count[i]
            row_s = pair_sum[i]
            if not isinstance(row_c, list) or len(row_c) != n_experts:
                raise HopeValidationError(f"pair_count[{i}] must have length {n_experts}")
            if not isinstance(row_s, list) or len(row_s) != n_experts:
                raise HopeValidationError(f"pair_sum[{i}] must have length {n_experts}")
            stats.pair_count[i] = [
                _require_count(v, f"pair_count[{i}][{j}]") for j, v in enumerate(row_c)
            ]
            stats.pair_sum[i] = [
                _require_float(v, f"pair_sum[{i}][{j}]") for j, v in enumerate(row_s)
            ]
        for field in ("first_sum", "gate_sum", "norm_sum"):
            values = doc.get(field)
            if not isinstance(values, list) or len(values) != n_experts:
                raise HopeValidationError(f"{field} must be a list of length {n_experts}")
            setattr(stats, field, [
                _require_float(v, f"{field}[{i}]") for i, v in enumerate(values)
            ])
        caps = doc.get("capabilities", {})
        if not isinstance(caps, dict):
            raise HopeValidationError("capabilities must be an object")
        for name, sub_doc in caps.items():
            if isinstance(sub_doc, dict) and sub_doc.get("capabilities"):
                raise HopeValidationError(
                    f"capability {name!r}: nested capability partitions are not allowed"
                )
            stats.capabilities[name] = cls.from_dict(sub_doc)
        stats.validate()
        return stats

    def to_json(self) -> str:
        """Deterministic JSON (sorted keys, no NaN) — byte-stable per content."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_json(cls, text: str) -> HopeStats:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HopeValidationError(f"invalid stats JSON: {exc}") from exc
        return cls.from_dict(doc)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomic write: temp file in the same directory, then os.replace."""
        path = os.fspath(path)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        payload = self.to_json()
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> HopeStats:
        path = os.fspath(path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise HopeValidationError(f"cannot read stats file {path}: {exc}") from exc
        return cls.from_json(text)


# ---------------------------------------------------------------------------
# Objective and selectors
# ---------------------------------------------------------------------------


def validate_f(f: object) -> tuple[int, list[list[float]]]:
    """Validate a square entrywise non-negative finite symmetric-ish F matrix.

    Symmetry is required (stats accumulate both orders); non-negativity is
    required because F entries are products of non-negative gate*norm terms.
    PSD is explicitly NOT required — the paper's QP never assumes it.
    """
    if not isinstance(f, (list, tuple)) or not f:
        raise HopeValidationError("F must be a non-empty square matrix")
    e = len(f)
    matrix: list[list[float]] = []
    for i in range(e):
        row = f[i]
        if not isinstance(row, (list, tuple)) or len(row) != e:
            raise HopeValidationError(f"F row {i} must have length {e}")
        parsed = [
            _require_float(row[j], f"F[{i}][{j}]") for j in range(e)
        ]
        matrix.append(parsed)
    for i in range(e):
        for j in range(e):
            if matrix[i][j] < 0.0:
                raise HopeValidationError(f"F[{i}][{j}] must be >= 0, got {matrix[i][j]}")
            if abs(matrix[i][j] - matrix[j][i]) > 1e-9 * max(1.0, abs(matrix[i][j])):
                raise HopeValidationError(
                    f"F not symmetric at [{i}][{j}]: {matrix[i][j]} vs {matrix[j][i]}"
                )
    return e, matrix


def quadratic_objective(f: Sequence[Sequence[float]], pruned: Sequence[int]) -> float:
    """``p^T F p`` for the prune set (p_k=1 iff k pruned)."""
    total = 0.0
    pruned_list = list(pruned)
    for pos, i in enumerate(pruned_list):
        row = f[i]
        total += row[i]
        for j in pruned_list[pos + 1 :]:
            total += 2.0 * row[j]
    return total


def _check_budget(n_experts: int, budget: int) -> None:
    if type(budget) is not int or isinstance(budget, bool):
        raise HopeSelectionError(f"budget must be an int, got {budget!r}")
    if not 0 <= budget <= n_experts:
        raise HopeSelectionError(
            f"budget {budget} outside [0, {n_experts}] for E={n_experts}"
        )


def exhaustive_select(
    f: object, budget: int, *, max_experts: int = EXACT_MAX_EXPERTS
) -> dict[str, Any]:
    """Complete enumeration of every fixed-size prune set.

    Returns the exact global minimizer of ``p^T F p`` with a certificate.  Ties
    are resolved to the lexicographically smallest expert-id set (iteration is
    in combinations order, replacement only on strict improvement).  Refuses
    ``E > max_experts`` (16) instead of silently approximating.
    """
    e, matrix = validate_f(f)
    _check_budget(e, budget)
    if e > max_experts:
        raise HopeSelectionError(
            f"exhaustive exact selection supports E <= {max_experts} "
            f"(paper toy-oracle range), got E={e}; use method={RELAXATION_METHOD!r} "
            "for larger layers (heuristic, labeled non-exact)"
        )
    best_set: tuple[int, ...] | None = None
    best_obj = math.inf
    ties = 0
    evaluated = 0
    for combo in itertools.combinations(range(e), budget):
        evaluated += 1
        value = quadratic_objective(matrix, combo)
        if value < best_obj:
            best_obj = value
            best_set = combo
            ties = 1
        elif value == best_obj:
            ties += 1
    # Iteration always yields at least one set for 0 <= budget <= E (the
    # budget==0/E cases are single-candidate sets).
    assert best_set is not None
    return {
        "method": EXHAUSTIVE_METHOD,
        "exact": True,
        "global_optimal": True,
        "certificate": "complete enumeration of all C(E,|P|) fixed-size sets",
        "n_experts": e,
        "budget": budget,
        "pruned": list(best_set),
        "objective": best_obj,
        "subsets_evaluated": evaluated,
        "expected_subsets": math.comb(e, budget),
        "ties": ties,
        "tie_policy": "lexicographically-smallest",
        "psd_assumed": False,
    }


def _scipy_optimize_and_numpy():
    """Lazy optional dependency gate — explicit error, never a fallback."""
    import importlib.util

    missing = [
        name
        for name in ("scipy", "numpy")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise HopeDependencyError(
            f"method {RELAXATION_METHOD!r} requires {', '.join(missing)}; "
            + SCIPY_INSTALL_HINT
        )
    import numpy
    import scipy.optimize

    return scipy.optimize, numpy


def _binary_1swap_refine(
    f: Sequence[Sequence[float]],
    pruned: list[int],
    retained: list[int],
    max_passes: int,
) -> tuple[list[int], int]:
    """Deterministic greedy binary 1-swap descent (labeled heuristic).

    Uses O(K) delta evaluation per candidate move.  Accepts only strict
    objective improvements; stops at a local optimum or ``max_passes``.
    """
    current = sorted(pruned)
    current_set = set(current)
    swaps = 0
    for _ in range(max(0, max_passes)):
        improved = False
        for x in list(current):
            out_rest = [j for j in current if j != x]
            base_out = f[x][x] + 2.0 * sum(f[x][j] for j in out_rest)
            best_gain = 0.0
            best_move: tuple[int, int] | None = None
            for y in retained:
                if y in current_set:
                    continue
                in_new = f[y][y] + 2.0 * sum(f[y][j] for j in out_rest)
                gain = base_out - in_new
                if gain > best_gain:
                    best_gain = gain
                    best_move = (x, y)
            if best_move is not None:
                x_out, y_in = best_move
                current.remove(x_out)
                current_set.discard(x_out)
                # insert keeping sorted order
                insert_at = 0
                while insert_at < len(current) and current[insert_at] < y_in:
                    insert_at += 1
                current.insert(insert_at, y_in)
                current_set.add(y_in)
                retained = [r for r in retained if r != y_in] + [x_out]
                retained.sort()
                swaps += 1
                improved = True
        if not improved:
            break
    return current, swaps


def scipy_relax_select(
    f: object,
    budget: int,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    max_iter: int = 300,
    ftol: float = 1e-12,
    local_swap: bool = False,
    local_swap_max_passes: int = 4,
) -> dict[str, Any]:
    """Paper-style relaxation: SLSQP on the capped simplex + top-|P| rounding.

    Mirrors §3.6 ("a standard QP solver" over ``p in [0,1]^E, sum p = |P|``,
    then round to the top-|P| entries); the authors' solver and Appendix D
    settings are unpublished, so solver internals here differ.  Result is
    ALWAYS a heuristic: ``exact=false``, ``global_optimal=false``.  Gradient of
    ``p^T F p`` is ``(F + F^T) p`` — valid for any symmetric F, PSD or not.
    """
    sco, numpy = _scipy_optimize_and_numpy()
    e, matrix = validate_f(f)
    _check_budget(e, budget)
    if budget == 0 or budget == e:
        result = exhaustive_select(matrix, budget)
        result["method"] = RELAXATION_METHOD
        result["exact"] = False
        result["global_optimal"] = False
        result["certificate"] = (
            "trivial budget (0 or E): single feasible set, evaluated directly"
        )
        result["relaxed_objective"] = result["objective"]
        result["rounding_gap"] = 0.0
        result["restarts"] = 0
        result["local_swap_applied"] = False
        return result
    f_arr = numpy.asarray(matrix, dtype=numpy.float64)

    def objective(p):
        return float(p @ f_arr @ p)

    def gradient(p):
        return (f_arr + f_arr.T) @ p

    constraints = (
        {
            "type": "eq",
            "fun": lambda p: float(p.sum()) - budget,
            "jac": lambda p: numpy.ones_like(p),
        },
    )
    bounds = [(0.0, 1.0)] * e
    starts: list[numpy.ndarray] = [
        numpy.full(e, budget / e, dtype=numpy.float64)
    ]
    for seed in seeds:
        if type(seed) is not int:
            raise HopeSelectionError(f"relaxation seeds must be ints, got {seed!r}")
        rng = random.Random(seed)
        starts.append(
            numpy.array([rng.random() for _ in range(e)], dtype=numpy.float64)
        )
    best_p = None
    best_fun = math.inf
    best_seed: int | str = "linspace"
    restarts_run = 0
    for seed_label, p0 in zip(["linspace"] + list(seeds), starts):
        res = sco.minimize(
            objective,
            p0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": ftol},
        )
        restarts_run += 1
        if numpy.isfinite(res.fun) and res.fun < best_fun:
            best_fun = float(res.fun)
            best_p = numpy.clip(res.x, 0.0, 1.0)
            best_seed = seed_label
    if best_p is None:
        raise HopeSelectionError(
            "SLSQP relaxation produced no finite solution; F may be degenerate"
        )
    order = sorted(range(e), key=lambda i: (-best_p[i], i))
    pruned = sorted(order[:budget])
    swaps = 0
    if local_swap:
        retained = [i for i in range(e) if i not in set(pruned)]
        pruned, swaps = _binary_1swap_refine(
            matrix, pruned, retained, local_swap_max_passes
        )
    value = quadratic_objective(matrix, pruned)
    return {
        "method": RELAXATION_METHOD,
        "exact": False,
        "global_optimal": False,
        "certificate": (
            "continuous relaxation (SLSQP) + top-|P| rounding, seeded restarts; "
            "heuristic — no global-optimality or PSD claim"
        ),
        "n_experts": e,
        "budget": budget,
        "pruned": list(pruned),
        "objective": value,
        "relaxed_objective": best_fun,
        "rounding_gap": value - best_fun,
        "restarts": restarts_run,
        "best_restart": best_seed,
        "seeds": list(seeds),
        "max_iter": max_iter,
        "local_swap_applied": bool(local_swap and swaps > 0),
        "local_swap_moves": swaps,
        "psd_assumed": False,
        "solver_note": (
            "paper §3.6 solver identity and Appendix D hyperparameters are "
            "unreleased; this implements the same relaxation+rounding recipe, "
            "not the authors' exact solver"
        ),
    }


def select(
    f: object,
    budget: int,
    *,
    method: str = "auto",
    relaxation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch: ``exhaustive`` (exact, E<=16), ``scipy-...`` (heuristic), or
    ``auto`` = exhaustive when E<=16 else the relaxation (result keeps its
    own exact/global_optimal labels)."""
    relaxation = relaxation or {}
    e = len(f) if isinstance(f, (list, tuple)) else 0
    if method == "auto":
        method = EXHAUSTIVE_METHOD if e <= EXACT_MAX_EXPERTS else RELAXATION_METHOD
    if method == EXHAUSTIVE_METHOD:
        return exhaustive_select(f, budget, max_experts=relaxation.get(
            "max_exact_experts", EXACT_MAX_EXPERTS))
    if method in (RELAXATION_METHOD, "scipy", "relaxation"):
        return scipy_relax_select(
            f,
            budget,
            seeds=tuple(relaxation.get("seeds", (0, 1, 2))),
            max_iter=int(relaxation.get("max_iter", 300)),
            ftol=float(relaxation.get("ftol", 1e-12)),
            local_swap=bool(relaxation.get("local_swap", False)),
            local_swap_max_passes=int(relaxation.get("local_swap_max_passes", 4)),
        )
    raise HopeSelectionError(
        f"unknown selection method {method!r}; expected "
        f"{EXHAUSTIVE_METHOD!r}, {RELAXATION_METHOD!r}, or 'auto'"
    )


def diag_topk_pruned(f: Sequence[Sequence[float]], budget: int) -> list[int]:
    """REAP-equivalent prune set: top-|P| smallest DIAGONAL entries.

    Zeroing HOPE's off-diagonals recovers REAP's prune-set ordering (§3.5,
    Fig. S4) — used by the toys as the first-order contrast.  Ties resolve to
    the smaller expert id (deterministic).
    """
    e = len(f)
    _check_budget(e, budget)
    order = sorted(range(e), key=lambda i: (f[i][i], i))
    return sorted(order[:budget])


# ---------------------------------------------------------------------------
# Synthetic fixtures (paper pathological structures)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureCase:
    """A closed-form F matrix with a hand-verified optimum.

    These are the paper's pathological structures: important pair,
    individually strong redundancy, block interaction, uniform, ties, sparse.
    ``expected_*`` fields are analytic, not solver outputs.
    """

    name: str
    description: str
    n_experts: int
    budget: int
    f: tuple[tuple[float, ...], ...]
    expected_pruned: tuple[int, ...]
    expected_objective: float
    expected_ties: int
    first_order_pruned: tuple[int, ...]
    first_order_objective: float
    note: str


def _matrix(e: int) -> list[list[float]]:
    return [[0.0] * e for _ in range(e)]


def _case_uniform() -> FixtureCase:
    e, k = 12, 4
    f = [[1.0] * e for _ in range(e)]
    return FixtureCase(
        name="uniform",
        description="every pair equally interacting: all C(E,k) sets tie",
        n_experts=e,
        budget=k,
        f=tuple(tuple(row) for row in f),
        expected_pruned=(0, 1, 2, 3),
        expected_objective=16.0,
        expected_ties=math.comb(e, k),
        first_order_pruned=(0, 1, 2, 3),
        first_order_objective=16.0,
        note="diagonal-only ranking cannot beat or lose: HOPE == REAP control",
    )


def _case_important_pair() -> FixtureCase:
    # Diagonals all equal; the single co-pruning of experts {0,1} is punished
    # by 2*1000.  Any set avoiding the joint prune costs 3; REAP's first-order
    # ranking (uniform diagonal -> lex) prunes {0,1,2} and pays the pair.
    e, k = 10, 3
    f = _matrix(e)
    for i in range(e):
        f[i][i] = 1.0
    f[0][1] = f[1][0] = 1000.0
    total = math.comb(e, k)
    joint = math.comb(e - 2, k - 2)
    return FixtureCase(
        name="important_pair",
        description="one decisive interaction pair must never be co-pruned",
        n_experts=e,
        budget=k,
        f=tuple(tuple(row) for row in f),
        expected_pruned=(0, 2, 3),
        expected_objective=3.0,
        expected_ties=total - joint,
        first_order_pruned=(0, 1, 2),
        first_order_objective=3.0 + 2000.0,
        note="first-order (REAP) ranking co-prunes the pair and pays 2003 vs 3",
    )


def _case_redundancy() -> FixtureCase:
    # Experts 0 and 1 are individually cheap to prune (diagonal 5) but fully
    # redundant with each other (pair 40): joint pruning costs 90, pruning one
    # of them plus a normal expert costs 55.
    e, k = 8, 2
    f = _matrix(e)
    diag = [5.0, 5.0] + [50.0] * (e - 2)
    for i in range(e):
        f[i][i] = diag[i]
    f[0][1] = f[1][0] = 40.0
    return FixtureCase(
        name="individually_strong_redundancy",
        description="two individually-strong experts are redundant with each other",
        n_experts=e,
        budget=k,
        f=tuple(tuple(row) for row in f),
        expected_pruned=(0, 2),
        expected_objective=55.0,
        expected_ties=12,
        first_order_pruned=(0, 1),
        first_order_objective=90.0,
        note="first-order prunes both redundant experts (90); HOPE splits them (55)",
    )


def _case_block() -> FixtureCase:
    # Two cliques of 8 experts with strong within-block interaction.  Jointly
    # pruning a 4-clique costs 4 + 2*30*6 = 364; splitting 2+2 costs 124.
    e, k, block, w = 16, 4, 8, 30.0
    f = _matrix(e)
    for i in range(e):
        f[i][i] = 1.0
    for base in (0, block):
        for i in range(base, base + block):
            for j in range(base, base + block):
                if i != j:
                    f[i][j] = w
    split_pairs = math.comb(block, 2)
    return FixtureCase(
        name="block_interaction",
        description="two interaction blocks; pruning must not gut one block",
        n_experts=e,
        budget=k,
        f=tuple(tuple(row) for row in f),
        expected_pruned=(0, 1, 8, 9),
        expected_objective=4.0 + 4.0 * w,
        expected_ties=split_pairs * split_pairs,
        first_order_pruned=(0, 1, 2, 3),
        first_order_objective=4.0 + 2.0 * w * math.comb(4, 2),
        note="first-order empties one block (364); HOPE balances (124)",
    )


def _case_ties() -> FixtureCase:
    # Three linked pairs each punished by 2*10 when co-pruned; the optimum is
    # any 4-set completing none of the links (objective 4).  Inclusion-exclusion:
    # free = C(9,4) - (3*C(7,2) - 3) = 126 - 60 = 66.
    e, k, w = 9, 4, 10.0
    f = _matrix(e)
    for i in range(e):
        f[i][i] = 1.0
    for a, b in ((0, 1), (2, 3), (4, 5)):
        f[a][b] = f[b][a] = w
    # Inclusion-exclusion over the three linked pairs:
    #   singles = 3 * C(7,2) = 63 sets containing one linked pair
    #   doubles = 3 sets containing two full pairs ({0,1,2,3},{0,1,4,5},{2,3,4,5})
    #   triples = 0 (impossible at k=4)
    union = 3 * math.comb(e - 2, k - 2) - 3
    free = math.comb(e, k) - union
    return FixtureCase(
        name="ties",
        description="linked pairs create many exactly-tied optima",
        n_experts=e,
        budget=k,
        f=tuple(tuple(row) for row in f),
        expected_pruned=(0, 2, 4, 6),
        expected_objective=float(k),
        expected_ties=free,
        first_order_pruned=(0, 1, 2, 3),
        first_order_objective=float(k) + 2.0 * w * 2,
        note="exact tie set of 66 optima; lex-smallest selection must be stable",
    )


def _case_sparse() -> FixtureCase:
    # Almost all interaction mass sits on one pair (0,8) with weight 4 while
    # experts 0 and 8 are far cheaper individually (diagonal 1 vs 10): the
    # cheap pair still wins JOINTLY (48 < 51) — a control where a sparse pair
    # does not override first-order cost, and HOPE agrees with REAP.
    e, k, w = 16, 6, 4.0
    f = _matrix(e)
    for i in range(e):
        f[i][i] = 1.0 if i in (0, 8) else 10.0
    f[0][8] = f[8][0] = w
    return FixtureCase(
        name="sparse",
        description="sparse F: one weak pair among near-diagonal costs",
        n_experts=e,
        budget=k,
        f=tuple(tuple(row) for row in f),
        expected_pruned=(0, 1, 2, 3, 4, 8),
        expected_objective=2.0 + 4.0 * 10.0 + 2.0 * w,
        expected_ties=math.comb(e - 2, k - 2),
        first_order_pruned=(0, 1, 2, 3, 4, 8),
        first_order_objective=2.0 + 4.0 * 10.0 + 2.0 * w,
        note="control: HOPE == REAP when the only pair is too weak to matter",
    )


FIXTURE_CASES: dict[str, FixtureCase] = {
    case.name: case
    for case in (
        _case_uniform(),
        _case_important_pair(),
        _case_redundancy(),
        _case_block(),
        _case_ties(),
        _case_sparse(),
    )
}


def _sample_excluding(
    rng: random.Random, population: Iterable[int], k: int, exclude: Sequence[int]
) -> list[int]:
    excluded = set(exclude)
    pool = [x for x in population if x not in excluded]
    return rng.sample(pool, k)


def fixture_rows(
    case_name: str, n_experts: int, top_k: int, rows: int, rng: random.Random
) -> list[list[int]]:
    """Deterministic synthetic routed-id rows shaped like each pathology."""
    out: list[list[int]] = []
    for _ in range(rows):
        pick: list[int] = []
        if case_name == "uniform":
            pick = rng.sample(range(n_experts), top_k)
        elif case_name == "important_pair":
            if rng.random() < 0.6 and top_k >= 2:
                pick = [0, 1] + _sample_excluding(rng, range(2, n_experts), top_k - 2, [])
            else:
                pick = rng.sample(range(n_experts), top_k)
        elif case_name == "individually_strong_redundancy":
            r = rng.random()
            if r < 0.35:
                pick = [0] + _sample_excluding(rng, range(1, n_experts), top_k - 1, [])
            elif r < 0.7:
                pick = [1] + _sample_excluding(rng, range(2, n_experts), top_k - 1, [])
            else:
                pick = rng.sample(range(n_experts), top_k)
        elif case_name == "block_interaction":
            block = n_experts // 2
            base = 0 if rng.random() < 0.5 else block
            pick = rng.sample(range(base, base + block), top_k)
        elif case_name == "ties":
            if rng.random() < 0.5:
                a, b = ((0, 1), (2, 3), (4, 5))[rng.randrange(3)]
                rest = _sample_excluding(
                    rng, range(n_experts), top_k - 2, [a, b]
                )
                pick = [a, b] + rest
            else:
                pick = rng.sample(range(n_experts), top_k)
        elif case_name == "sparse":
            if rng.random() < 0.12 and top_k >= 2:
                pick = [0, 8] + _sample_excluding(
                    rng, range(1, n_experts), top_k - 2, [0, 8]
                )
            else:
                pick = rng.sample(range(n_experts), top_k)
        else:
            raise HopeValidationError(f"unknown fixture case {case_name!r}")
        out.append(sorted(set(pick)) if len(set(pick)) == top_k else pick)
        if len(out[-1]) != top_k or len(set(out[-1])) != top_k:
            # deterministic repair (should not trigger; guards generator bugs)
            fixed = list(dict.fromkeys(out[-1]))
            fixed += _sample_excluding(rng, range(n_experts), top_k - len(fixed), fixed)
            out[-1] = fixed[:top_k]
    return out


def build_fixture_stats(
    case_name: str,
    *,
    rows: int,
    top_k: int,
    seed: int,
    capabilities: Sequence[str] = (),
) -> HopeStats:
    """Synthetic stats for one pathology: real accumulation path, seeded rng.

    Fixture stats exercise the FULL statistics/validation/selection pipeline;
    they carry no closed-form claim — that lives in :data:`FIXTURE_CASES`
    (analytic F matrices) and the toys/tests.
    """
    case = FIXTURE_CASES.get(case_name)
    if case is None:
        raise HopeValidationError(
            f"unknown fixture case {case_name!r}; known: {sorted(FIXTURE_CASES)}"
        )
    rng = random.Random(f"{seed}:{case_name}")
    stats = HopeStats(case.n_experts, top_k)
    selected = fixture_rows(case_name, case.n_experts, top_k, rows, rng)
    for index, row in enumerate(selected):
        raw_gates = [rng.uniform(0.05, 2.0) for _ in range(top_k)]
        total = sum(raw_gates)
        gates = [g / total for g in raw_gates]
        norms = [rng.uniform(0.25, 4.0) for _ in range(top_k)]
        capability = None
        if capabilities:
            capability = capabilities[index % len(capabilities)]
        stats.add_batch([row], [gates], [norms], capability=capability)
    stats.validate()
    return stats


# ---------------------------------------------------------------------------
# Toy suite (closed-form cases + independent oracle)
# ---------------------------------------------------------------------------


def oracle_objective(f: Sequence[Sequence[float]], pruned: Sequence[int]) -> float:
    """Independent objective evaluation: full quadratic form over p vectors.

    Deliberately computed differently from :func:`quadratic_objective`
    (dense double loop over the prune indicator vector) so tests and toys can
    cross-check the fast path.
    """
    e = len(f)
    p = [0] * e
    for k in pruned:
        if type(k) is not int or not 0 <= k < e:
            raise HopeSelectionError(f"oracle: index {k!r} outside F")
        p[k] = 1
    total = 0.0
    for i in range(e):
        if not p[i]:
            continue
        for j in range(e):
            if p[j]:
                total += float(f[i][j])
    return total


def oracle_exhaustive_select(
    f: object, budget: int, *, product_limit: int = 1 << 20
) -> dict[str, Any]:
    """Independent exact selector: iterate ALL binary p vectors.

    Used as the oracle for small E (up to ``product_limit`` vectors).
    """
    e, matrix = validate_f(f)
    _check_budget(e, budget)
    if 1 << e > product_limit:
        raise HopeSelectionError(
            f"product-space oracle refuses E={e}: {1 << e} vectors > limit {product_limit}"
        )
    best: tuple[int, ...] | None = None
    best_obj = math.inf
    ties = 0
    for bits in itertools.product((0, 1), repeat=e):
        if sum(bits) != budget:
            continue
        combo = tuple(i for i, bit in enumerate(bits) if bit)
        value = oracle_objective(matrix, combo)
        if value < best_obj:
            best_obj, best, ties = value, combo, 1
        elif value == best_obj:
            ties += 1
            # product() varies the last coordinate fastest, so iteration order
            # is not subset-lex order; enforce the same lexicographically
            # smallest tie policy as the combinations-based selector.
            if combo < best:
                best = combo
    assert best is not None  # budget validated; product space contains >= 1 set
    return {
        "method": "product-oracle",
        "exact": True,
        "global_optimal": True,
        "n_experts": e,
        "budget": budget,
        "pruned": list(best),
        "objective": best_obj,
        "ties": ties,
    }


def run_toys(config: dict[str, Any]) -> dict[str, Any]:
    """Run every closed-form pathological case through the real selectors.

    Each case is checked three ways: exhaustive optimum vs the analytic
    expectation, objective recomputed by the independent dense oracle, and the
    first-order (REAP diagonal) contrast.  When scipy is available the
    relaxation is run and reported beside the exact optimum — never as a
    replacement for it.
    """
    names = config.get("fixtures", {}).get("cases") or sorted(FIXTURE_CASES)
    relaxation_cfg = config.get("relaxation", {})
    scipy_status = "available"
    try:
        _scipy_optimize_and_numpy()
    except HopeDependencyError:
        scipy_status = "missing (optional; relaxation rows report skip)"
    results = []
    all_ok = True
    for name in names:
        case = FIXTURE_CASES.get(name)
        if case is None:
            raise HopeValidationError(f"config references unknown toy case {name!r}")
        exact = exhaustive_select(case.f, case.budget)
        dense_value = oracle_objective(case.f, exact["pruned"])
        checks = {
            "pruned_matches_analytic": tuple(exact["pruned"]) == case.expected_pruned,
            "objective_matches_analytic": exact["objective"] == case.expected_objective,
            "ties_matches_analytic": exact["ties"] == case.expected_ties,
            "evaluated_all_fixed_sets": exact["subsets_evaluated"]
            == exact["expected_subsets"],
            "dense_oracle_agrees": dense_value == exact["objective"],
            "first_order_contrast": (
                diag_topk_pruned(case.f, case.budget) == list(case.first_order_pruned)
                and oracle_objective(case.f, case.first_order_pruned)
                == case.first_order_objective
            ),
            "hope_not_worse_than_first_order": exact["objective"]
            <= case.first_order_objective,
        }
        relaxation: dict[str, Any]
        if scipy_status == "available":
            relax = scipy_relax_select(
                case.f,
                case.budget,
                seeds=tuple(relaxation_cfg.get("seeds", (0, 1, 2))),
                local_swap=bool(relaxation_cfg.get("local_swap", False)),
                local_swap_max_passes=int(
                    relaxation_cfg.get("local_swap_max_passes", 4)
                ),
            )
            checks["relaxation_label_is_heuristic"] = (
                relax["exact"] is False and relax["global_optimal"] is False
            )
            checks["relaxation_never_beats_exact"] = (
                relax["objective"] >= exact["objective"] - 1e-9
            )
            relaxation = {
                "pruned": relax["pruned"],
                "objective": relax["objective"],
                "relaxed_objective": relax["relaxed_objective"],
                "local_swap_applied": relax["local_swap_applied"],
                "exact": False,
            }
        else:
            try:
                scipy_relax_select(case.f, case.budget)
            except HopeDependencyError as exc:
                relaxation = {"skipped": True, "error": str(exc)}
            else:
                raise HopeValidationError(
                    "scipy reported missing but relaxation ran — dependency gate broken"
                )
        ok = all(checks.values())
        all_ok = all_ok and ok
        results.append(
            {
                "case": name,
                "description": case.description,
                "note": case.note,
                "n_experts": case.n_experts,
                "budget": case.budget,
                "checks": checks,
                "exact": {
                    "pruned": exact["pruned"],
                    "objective": exact["objective"],
                    "ties": exact["ties"],
                    "subsets_evaluated": exact["subsets_evaluated"],
                },
                "first_order": {
                    "pruned": list(case.first_order_pruned),
                    "objective": case.first_order_objective,
                },
                "relaxation": relaxation,
                "pass": ok,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": TOYS_KIND,
        "paper": PAPER_REF,
        "scipy": scipy_status,
        "cases": results,
        "pass": all_ok,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = os.fspath(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HopeValidationError(f"cannot load HOPE config {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("kind") != "hope-config":
        raise HopeValidationError(f"{path}: not a hope-config document")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise HopeValidationError(
            f"{path}: schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSION}"
        )
    exact = doc.get("exact", {})
    if type(exact.get("max_experts")) is not int or exact["max_experts"] < 1:
        raise HopeValidationError(f"{path}: exact.max_experts must be a positive int")
    fixtures = doc.get("fixtures", {})
    cases = fixtures.get("cases")
    if not isinstance(cases, list) or not cases:
        raise HopeValidationError(f"{path}: fixtures.cases must be a non-empty list")
    for name in cases:
        if name not in FIXTURE_CASES:
            raise HopeValidationError(
                f"{path}: unknown fixture case {name!r}; known: {sorted(FIXTURE_CASES)}"
            )
    if type(fixtures.get("rows_per_case")) is not int or fixtures["rows_per_case"] < 1:
        raise HopeValidationError(f"{path}: fixtures.rows_per_case must be positive int")
    if type(fixtures.get("top_k")) is not int or fixtures["top_k"] < 1:
        raise HopeValidationError(f"{path}: fixtures.top_k must be positive int")
    return doc


def _write_json_atomic(path: str, doc: object) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, sort_keys=True, indent=1, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _cmd_observe_fixture(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    fixtures = config["fixtures"]
    seed = int(config.get("seed", 0))
    rows = int(fixtures["rows_per_case"])
    top_k = int(fixtures["top_k"])
    capabilities = fixtures.get("capabilities") or []
    os.makedirs(args.out, exist_ok=True)
    written = []
    # Stream/merge demonstration: cases of equal layer shape (E, top_k) act as
    # shards of one synthetic corpus per layer size; merging across differing E
    # is (correctly) a count-safe error and is rejected by HopeStats.merge.
    groups: dict[tuple[int, int], HopeStats] = {}
    for name in fixtures["cases"]:
        stats = build_fixture_stats(
            name, rows=rows, top_k=top_k, seed=seed, capabilities=capabilities
        )
        path = os.path.join(args.out, f"{name}.json")
        stats.save(path)
        written.append(path)
        key = (stats.n_experts, stats.top_k or 0)
        if key in groups:
            groups[key].merge(stats)
        else:
            groups[key] = stats.deepcopy()
    merged_paths = []
    for (n_experts, _top_k), stats in sorted(groups.items()):
        merged_path = os.path.join(args.out, f"merged-e{n_experts}.json")
        stats.save(merged_path)
        merged_paths.append(merged_path)
        written.append(merged_path)
    print(
        json.dumps(
            {
                "kind": "hope-observe-fixture",
                "paper": PAPER_REF,
                "seed": seed,
                "rows_per_case": rows,
                "top_k": top_k,
                "merged_groups": merged_paths,
                "files": written,
            },
            sort_keys=True,
            indent=1,
        )
    )
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    stats = HopeStats.load(args.stats)
    if stats.total_rows == 0:
        raise HopeValidationError(
            f"{args.stats}: no accumulated rows — refusing to select on empty stats"
        )
    if args.capability is not None:
        sub = stats.capabilities.get(args.capability)
        if sub is None:
            raise HopeValidationError(
                f"capability {args.capability!r} not present; available: "
                f"{sorted(stats.capabilities)}"
            )
        source = sub
    else:
        source = stats
    f_matrix = source.conditional_f()
    budget = args.budget
    relaxation_cfg = dict(config.get("relaxation", {}))
    if args.local_swap:
        relaxation_cfg["local_swap"] = True
    relaxation_cfg["max_exact_experts"] = int(config["exact"]["max_experts"])
    method = args.method
    if method == "auto":
        method = (
            EXHAUSTIVE_METHOD
            if source.n_experts <= relaxation_cfg["max_exact_experts"]
            else RELAXATION_METHOD
        )
    certificate = select(
        f_matrix, budget, method=method, relaxation=relaxation_cfg
    )
    # Independent dense re-check of the delivered set (cheap, always on).
    dense = oracle_objective(f_matrix, certificate["pruned"])
    if dense != certificate["objective"]:
        raise HopeValidationError(
            f"objective mismatch after selection: {certificate['objective']} vs dense {dense}"
        )
    doc = {
        "schema_version": SCHEMA_VERSION,
        "kind": SELECTION_KIND,
        "paper": PAPER_REF,
        "stats_file": os.fspath(args.stats),
        "stats_total_rows": source.total_rows,
        "capability": args.capability,
        "method_requested": args.method,
        "method_used": method,
        "certificate": certificate,
        "pruned": certificate["pruned"],
        "retained": [i for i in range(source.n_experts) if i not in set(certificate["pruned"])],
        "first_order_pruned": diag_topk_pruned(f_matrix, budget),
        "objective": certificate["objective"],
        "dense_oracle_objective": dense,
    }
    if args.out:
        _write_json_atomic(args.out, doc)
    print(json.dumps(doc, sort_keys=True, indent=1, allow_nan=False))
    return 0


def _cmd_toys(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = run_toys(config)
    if args.out:
        _write_json_atomic(args.out, report)
    print(json.dumps(report, sort_keys=True, indent=1, allow_nan=False))
    return 0 if report["pass"] else 1


def _cmd_merge(args: argparse.Namespace) -> int:
    if len(args.stats_files) < 1:
        raise HopeValidationError("merge needs at least one stats file")
    merged = HopeStats.load(args.stats_files[0])
    for path in args.stats_files[1:]:
        merged.merge(HopeStats.load(path))
    merged.validate()
    merged.save(args.out)
    print(
        json.dumps(
            {
                "kind": "hope-merge",
                "out": os.fspath(args.out),
                "inputs": [os.fspath(p) for p in args.stats_files],
                "total_rows": merged.total_rows,
                "n_experts": merged.n_experts,
                "top_k": merged.top_k,
            },
            sort_keys=True,
            indent=1,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.pruning.hope",
        description=(
            "HOPE statistics/exact selector CLI (stdlib core; arXiv:2609.18916v1). "
            "No network access, no telemetry."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    obs = sub.add_parser("observe-fixture", help="accumulate synthetic pathological stats")
    obs.add_argument("--config", required=True, help="hope-config JSON (e.g. configs/hope.json)")
    obs.add_argument("--out", required=True, help="output directory for stats JSON files")
    obs.set_defaults(func=_cmd_observe_fixture)

    sel = sub.add_parser("select", help="solve the prune-set QP on a stats file")
    sel.add_argument("--config", required=True)
    sel.add_argument("--stats", required=True, help="HopeStats JSON file")
    sel.add_argument("--budget", type=int, required=True, help="experts to prune per layer")
    sel.add_argument(
        "--method",
        default="auto",
        choices=["auto", EXHAUSTIVE_METHOD, RELAXATION_METHOD, "scipy", "relaxation"],
    )
    sel.add_argument("--capability", default=None, help="select on a capability partition")
    sel.add_argument("--local-swap", action="store_true", help="add labeled binary 1-swap refinement")
    sel.add_argument("--out", default=None, help="write selection JSON here (also printed)")
    sel.set_defaults(func=_cmd_select)

    toys = sub.add_parser("toys", help="run closed-form pathological cases vs analytic optima")
    toys.add_argument("--config", required=True)
    toys.add_argument("--out", default=None)
    toys.set_defaults(func=_cmd_toys)

    merge = sub.add_parser("merge", help="merge/stream stats documents (resume path)")
    merge.add_argument("--out", required=True)
    merge.add_argument("stats_files", nargs="+")
    merge.set_defaults(func=_cmd_merge)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HopeError as exc:
        print(json.dumps({"kind": "hope-error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
