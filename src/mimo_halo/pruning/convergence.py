"""Calibration-size convergence sweep over HopeStats observation parts.

Given ordered HopeStats observation parts (one ``hope-stats`` JSON document
per observation segment), merge them along the proven resume path
(``HopeStats.merge`` - byte-identical, per ``MergeResumeTests``) up to a
growing token-budget schedule, run HOPE selection at every budget, and emit
the distribution-convergence metrics between consecutive budgets as a JSON
record.

The default schedule is the named constant :data:`TOKEN_BUDGETS` -
50K/100K/250K/500K/1M accumulated routing rows (= tokens) - override with
``--budgets`` for small fixtures or denser sweeps.

Metrics - docs/evaluation-methodology.md section 4, level-3 row::

    | 3. Distribution | Routing and calibration distribution | Routing top8
    overlap, gate correlation, expert frequency, Jaccard/convergence across
    token budgets, heldout perplexity. |

with "levels 1-3 are distribution controls" (section 4) and "token agreement
never becomes task success" (section 5):

- ``jaccard``: ``|P_from INTERSECT P_to| / |P_from UNION P_to|`` over the
  pruned expert-id sets of consecutive budgets - the set-agreement family
  already quoted in ``baselines/compare.py`` ("``retained_jaccard =
  |retained_baseline INTERSECT retained_candidate| / |retained_baseline
  UNION retained_candidate|``"; a set agreement statistic, NOT a quality
  score).  1.0 means the selection did not move, never "better".
- ``rank_stability``: Spearman rank correlation of per-expert first-order
  scores between consecutive budgets:
  ``rho = 1 - 6 * sum_i d_i^2 / (n * (n^2 - 1))`` where ``d_i`` is the rank
  difference of expert ``i`` and ``n = n_experts``.  Ranks order by
  ``first_order`` score descending with ties broken by ascending expert id,
  so ranks are a strict permutation and the no-ties formula is exact.

Every budget's step carries rows actually merged, whether the budget was
reached, the selected prune set, its objective (dense re-checked through
``oracle_objective``), and the first-order scores/ranks behind
``rank_stability`` - the numbers are auditable from the record alone.

CLI::

    PYTHONPATH=src python3 -m mimo_halo.pruning.convergence \
        part-0.json part-1.json [--budgets 4,8,12] --prune-budget K \
        [--config configs/hope.json] [--out report.json]

Parts are consumed left to right (observation order).  Exit codes: 0
success, 2 validation failure (fail closed), mirroring the hope CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..traces.common import file_sha256, sha256_text, stable_json
from .hope import (
    HopeError,
    HopeStats,
    HopeValidationError,
    _write_json_atomic,
    load_config,
    oracle_objective,
    select,
)

SCHEMA_VERSION = "1.0.0"
KIND = "mimo-halo-calibration-convergence"

TOKEN_BUDGETS: tuple[int, ...] = (50_000, 100_000, 250_000, 500_000, 1_000_000)


def _validate_budgets(budgets: Any) -> tuple[int, ...]:
    if isinstance(budgets, (str, bytes)):
        raise HopeValidationError(f"budget schedule must be ints, got {budgets!r}")
    try:
        schedule = tuple(budgets)
    except TypeError as exc:
        raise HopeValidationError(
            f"budget schedule must be an iterable of ints, got {budgets!r}"
        ) from exc
    if not schedule:
        raise HopeValidationError("budget schedule must not be empty")
    previous = 0
    for budget in schedule:
        if type(budget) is not int or budget <= previous:
            raise HopeValidationError(
                f"budgets must be strictly increasing positive ints, got "
                f"{list(schedule)}"
            )
        previous = budget
    return schedule


def _ranks(scores: list[float]) -> list[int]:
    """Strict per-expert ranks: score descending, ties broken by expert id."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for position, expert in enumerate(order):
        ranks[expert] = position + 1
    return ranks


def _rank_stability(prev: list[int], cur: list[int]) -> float:
    """Spearman rho over strict ranks (no ties -> the no-ties formula is exact)."""
    n = len(prev)
    if n <= 1:
        return 1.0
    d2 = sum((a - b) ** 2 for a, b in zip(prev, cur))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _jaccard(prev: list[int], cur: list[int]) -> float:
    a, b = set(prev), set(cur)
    union = a | b
    if not union:  # unreachable: prune-budget >= 1 is validated
        return 1.0
    return len(a & b) / len(union)


def run_convergence(
    part_paths: list,
    *,
    prune_budget: int,
    budgets: tuple[int, ...] = TOKEN_BUDGETS,
    config_path: Any = None,
) -> dict:
    """Sweep token budgets over ordered observation parts; emit the JSON record.

    At each budget the shortest prefix of ordered parts whose cumulative rows
    reach the budget is merged in (the proven resume path), selection runs on
    the merged statistics, and consecutive-budget Jaccard / rank stability are
    recorded.  Budgets beyond the available rows are reported with
    ``reached: false`` instead of being skipped or faked.
    """
    paths = list(part_paths)
    if not paths:
        raise HopeValidationError("convergence needs at least one observation part")
    schedule = _validate_budgets(budgets)
    if (
        type(prune_budget) is not int
        or isinstance(prune_budget, bool)
        or prune_budget < 1
    ):
        raise HopeValidationError(
            f"prune-budget must be an int >= 1, got {prune_budget!r}"
        )
    relaxation: dict[str, Any] = {}
    config_sha: str | None = None
    if config_path is not None:
        config = load_config(config_path)
        relaxation = dict(config.get("relaxation", {}))
        relaxation["max_exact_experts"] = int(config["exact"]["max_experts"])
        config_sha = file_sha256(Path(str(config_path)))

    parts = [HopeStats.load(path) for path in paths]
    part_inputs = [
        {"sha256": file_sha256(Path(str(path))), "rows": part.total_rows}
        for path, part in zip(paths, parts)
    ]

    merged: HopeStats | None = None
    consumed = 0
    steps: list[dict] = []
    consecutive: list[dict] = []
    for budget in schedule:
        # Proven resume path: load first part, merge the rest incrementally.
        while consumed < len(parts) and (
            merged is None or merged.total_rows < budget
        ):
            part = parts[consumed]
            consumed += 1
            merged = part if merged is None else merged.merge(part)
        if merged.total_rows == 0:
            raise HopeValidationError(
                "no accumulated rows - refusing to select on empty stats"
            )
        f_matrix = merged.conditional_f()
        certificate = select(
            f_matrix, prune_budget, method="auto", relaxation=relaxation
        )
        # Independent dense re-check of the delivered set (cheap, always on).
        dense = oracle_objective(f_matrix, certificate["pruned"])
        if dense != certificate["objective"]:
            raise HopeValidationError(
                f"objective mismatch after selection: "
                f"{certificate['objective']} vs dense {dense}"
            )
        scores = merged.first_order()
        ranks = _ranks(scores)
        if steps:
            previous = steps[-1]
            consecutive.append(
                {
                    "from_budget": previous["budget"],
                    "to_budget": budget,
                    "from_rows": previous["rows"],
                    "to_rows": merged.total_rows,
                    "jaccard": _jaccard(previous["pruned"], certificate["pruned"]),
                    "rank_stability": _rank_stability(
                        previous["first_order_ranks"], ranks
                    ),
                }
            )
        steps.append(
            {
                "budget": budget,
                "rows": merged.total_rows,
                "parts_consumed": consumed,
                "reached": merged.total_rows >= budget,
                "pruned": list(certificate["pruned"]),
                "objective": certificate["objective"],
                "method_used": certificate["method"],
                "exact": bool(certificate["exact"]),
                "global_optimal": bool(certificate["global_optimal"]),
                "first_order_scores": list(scores),
                "first_order_ranks": ranks,
            }
        )
    assert merged is not None  # schedule is non-empty; first budget consumes >= 1 part
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "inputs": {
            "digest": sha256_text(
                stable_json(
                    {
                        "parts": part_inputs,
                        "budgets": list(schedule),
                        "prune_budget": prune_budget,
                        "config_sha256": config_sha,
                    }
                )
            ),
            "parts": part_inputs,
            "config_sha256": config_sha,
        },
        "budgets": list(schedule),
        "n_experts": merged.n_experts,
        "top_k": merged.top_k,
        "prune_budget": prune_budget,
        "parts_beyond_schedule": len(parts) - consumed,
        "steps": steps,
        "consecutive": consecutive,
        "definitions": {
            "methodology_row": (
                "docs/evaluation-methodology.md section 4, level-3 row: "
                "| 3. Distribution | Routing and calibration distribution | "
                "Routing top8 overlap, gate correlation, expert frequency, "
                "Jaccard/convergence across token budgets, heldout "
                "perplexity. | (levels 1-3 are distribution controls; "
                "section 5: 'token agreement never becomes task success')"
            ),
            "jaccard": (
                "jaccard = |P_from INTERSECT P_to| / |P_from UNION P_to| over "
                "pruned expert-id sets at consecutive token budgets - set "
                "agreement statistic, same definition family as "
                "src/mimo_halo/baselines/compare.py JACCARD_DEFINITION "
                "('retained_jaccard = |retained_baseline INTERSECT "
                "retained_candidate| / |retained_baseline UNION "
                "retained_candidate|'); 1.0 means identical selections, "
                "never 'better'"
            ),
            "rank_stability": (
                "Spearman rho = 1 - 6 * sum_i d_i^2 / (n * (n^2 - 1)) over "
                "per-expert first-order ranks at consecutive budgets "
                "(d_i = rank difference, n = n_experts; ranks ordered by "
                "first_order score descending, ties by ascending expert id - "
                "strict ranks, so the no-ties formula is exact); 1.0 means "
                "the importance ranking did not move"
            ),
            "budget_semantics": (
                "each budget merges the shortest prefix of the ordered parts "
                "via HopeStats.merge (the proven byte-identical resume path) "
                "whose cumulative rows reach the budget; rows = accumulated "
                "routing rows (tokens); reached=false means the schedule ran "
                "past the available rows"
            ),
            "controls_not_success": (
                "distribution control only - never a task-success metric "
                "(docs/evaluation-methodology.md sections 4-5)"
            ),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m mimo_halo.pruning.convergence",
        description=(
            "sweep growing token budgets over ordered HopeStats observation "
            "parts and emit expert-set Jaccard + rank-stability between "
            "consecutive budgets"
        ),
    )
    parser.add_argument(
        "parts",
        nargs="+",
        help="ordered HopeStats observation part files (observation order)",
    )
    parser.add_argument(
        "--prune-budget",
        type=int,
        required=True,
        help="experts to prune at each checkpoint (>= 1)",
    )
    parser.add_argument(
        "--budgets",
        default=None,
        help=(
            "comma-separated token budgets (default: "
            + ",".join(str(b) for b in TOKEN_BUDGETS)
            + ")"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="optional hope-config (configs/hope.json) for method settings",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the convergence JSON here (also printed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        budgets = TOKEN_BUDGETS
        if args.budgets is not None:
            try:
                budgets = tuple(
                    int(chunk) for chunk in args.budgets.split(",") if chunk.strip()
                )
            except ValueError as exc:
                raise HopeValidationError(
                    f"--budgets must be comma-separated ints, got {args.budgets!r}"
                ) from exc
        doc = run_convergence(
            args.parts,
            prune_budget=args.prune_budget,
            budgets=budgets,
            config_path=args.config,
        )
        if args.out:
            _write_json_atomic(args.out, doc)
        print(json.dumps(doc, sort_keys=True, indent=1, allow_nan=False))
        return 0
    except HopeError as exc:
        print(json.dumps({"kind": "hope-error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
