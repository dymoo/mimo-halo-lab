"""CLI for candidate construction: one command from a REAP selection to a
candidate checkpoint with full artifact sidecars.

    PYTHONPATH=src python -m mimo_halo.build \
        --inventory manifests/models/mimo-v2.6-flash-rl/inventory.json \
        --source-root "$MIMO_LAB/source-models/XiaomiMiMo--MiMo-V2.6-Flash-RL/5711b268169967567844e1e560e8a3966da959b1" \
        --candidate-id reap45-mxfp4 \
        --selection /path/to/reap_selection.json \
        --out "$MIMO_LAB/quantized-models/reap45-mxfp4" \
        --seed 7 --environment /path/to/environment.json \
        --dataset calibration-corpus=SHA256

Selection inputs are turned into a prune map first (mimo_halo.pruning.maps,
same validation the planning CLI uses); a pre-built ``--prune-map`` may be
passed instead. The recipe is the named entry of
``configs/experiments/compression-sweep.json`` ``candidates[]`` (``--config``
may point at any file with the same shape). Exit codes: 0 built and
verified, 1 construction/refusal failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..pruning.maps import PruneMapError, build_prune_map, load_inventory, load_selection_file
from .candidate import build_candidate
from .errors import BuildError


def _preflight_paths(args) -> None:
    """Refuse unsafe output paths BEFORE anything is written.

    ``build_candidate`` re-checks all of this later, but the CLI publishes a
    derived prune map beside ``--out`` first, so the confinement and
    empty-output checks must run here too: an ``--out`` inside the verified
    source tree (or a derived-map path landing in it) must never create or
    modify a file there, and a non-empty ``--out`` is refused up front so
    retries stay clean.
    """
    source_real = Path(os.path.realpath(args.source_root))

    def _inside_source(path_real: Path, label: str) -> None:
        if path_real == source_real or source_real in path_real.parents:
            raise BuildError(
                f"{label} must be outside the verified source root: {path_real}"
            )

    out_real = Path(os.path.realpath(args.out))
    _inside_source(out_real, "output directory")
    if args.prune_map is None:
        _inside_source(
            Path(os.path.realpath(str(args.out) + ".prune_map.json")),
            "derived prune map path",
        )
    out_path = Path(args.out)
    if out_path.exists():
        if not out_path.is_dir():
            raise BuildError(f"output path exists and is not a directory: {args.out!r}")
        if any(os.scandir(out_path)):
            raise BuildError(
                f"output directory must be absent or empty, found existing "
                f"content: {args.out!r}"
            )


def _load_sweep_config(path: str) -> dict:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read sweep config {path}: {exc}") from exc
    if not isinstance(cfg, dict) or not isinstance(cfg.get("candidates"), list):
        raise BuildError(f"{path} has no candidates[] list")
    return cfg


def _recipe_for(cfg: dict, candidate_id: str) -> dict:
    for candidate in cfg["candidates"]:
        if isinstance(candidate, dict) and candidate.get("candidate_id") == candidate_id:
            return candidate
    known = sorted(
        c.get("candidate_id") for c in cfg["candidates"] if isinstance(c, dict)
    )
    raise BuildError(f"candidate-id {candidate_id!r} not in config (known: {known})")


def _digest_basis(cfg: dict, overrides) -> list[str]:
    if overrides:
        return list(overrides)
    basis = (cfg.get("source_weights") or {}).get("digest_basis")
    if not isinstance(basis, list) or not basis:
        raise BuildError(
            "config source_weights.digest_basis is missing; pass --digest-basis"
        )
    return list(basis)


def _publish_map(map_path: str, text: str) -> None:
    """Publish the derived prune map without ever clobbering an existing file.

    An identical file is reused as-is (failed builds stay retryable without
    a rewrite); different existing content, a directory, or any other
    pre-existing entry is refused explicitly.
    """
    encoded = text.encode("utf-8")
    path = Path(map_path)
    if path.exists():
        if not path.is_file():
            raise BuildError(
                f"refusing to publish derived prune map: {map_path!r} exists "
                "and is not a regular file"
            )
        current = path.read_bytes()
        if current == encoded:
            return  # identical prior publication: retry path, never rewritten
        raise BuildError(
            f"refusing to overwrite existing {map_path!r} with different "
            "prune-map content (remove it, or pass --prune-map to use it)"
        )
    try:
        with open(map_path, "xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        # Raced or filesystem-level surprise: same no-clobber rules apply.
        if not path.is_file() or path.read_bytes() != encoded:
            raise BuildError(
                f"refusing to overwrite existing {map_path!r} with different "
                "prune-map content (remove it, or pass --prune-map to use it)"
            )


def _ensure_prune_map(args) -> str:
    """Use --prune-map as given, or build one from --selection (published
    exclusively next to --out so the output directory stays absent-or-empty)."""
    inventory = load_inventory(args.inventory)
    if args.prune_map is not None:
        # Still re-validated against this inventory inside build_candidate.
        return args.prune_map
    selection, provenance = load_selection_file(
        args.selection, inventory["architecture"]
    )
    document = build_prune_map(
        inventory, selection, "external_selection", provenance, args.inventory
    )
    map_path = str(args.out) + ".prune_map.json"
    _publish_map(map_path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return map_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mimo_halo.build",
        description="Construct one candidate checkpoint with artifact sidecars.",
    )
    parser.add_argument("--inventory", required=True, help="pinned inventory JSON")
    parser.add_argument("--source-root", required=True,
                        help="verified source revision directory (pinned SHA)")
    parser.add_argument("--out", required=True,
                        help="output directory (must be absent or empty)")
    parser.add_argument("--config", default="configs/experiments/compression-sweep.json",
                        help="sweep config carrying candidates[]")
    parser.add_argument("--candidate-id", required=True,
                        help="candidate_id entry of the config")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--selection", default=None,
                        help="completed REAP selection JSON (builds the prune map)")
    source.add_argument("--prune-map", default=None,
                        help="pre-built prune map JSON")
    parser.add_argument("--seed", required=True, type=int,
                        help="recorded build/selection seed (artifact stage input)")
    parser.add_argument("--environment", required=True,
                        help="sanitized environment JSON")
    parser.add_argument("--dataset", action="append", required=True,
                        metavar="NAME=SHA256",
                        help="dataset content digest (repeatable)")
    parser.add_argument("--calibration-config", default=None,
                        help="calibration config JSON (recorded as sidecar input)")
    parser.add_argument("--digest-basis", action="append", default=None,
                        metavar="TEXT",
                        help="source digest-basis line (repeatable; default: config)")
    parser.add_argument("--expert-order", default=None,
                        help="JSON {layer: [original ids, lowest salience first]}")
    parser.add_argument("--kind", default="pruned",
                        choices=("pruned", "quantized"),
                        help="artifact stage (quantized requires --parent)")
    parser.add_argument("--parent", action="append", default=[],
                        metavar="ROLE=[KIND:]SHA",
                        help="parent artifact digest (repeatable)")
    parser.add_argument("--command", action="append", default=[],
                        metavar="TEXT",
                        help="extra reproduction command (repeatable)")
    parser.add_argument("--conversion-commit", default=None, metavar="SHA",
                        help="full 40-hex conversion commit (honest null if absent)")
    parser.add_argument("--runtime-commit", default=None, metavar="SHA",
                        help="full 40-hex runtime commit")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _preflight_paths(args)  # before ANY file is written
        cfg = _load_sweep_config(args.config)
        recipe = _recipe_for(cfg, args.candidate_id)
        prune_map_path = _ensure_prune_map(args)
        expert_order = None
        if args.expert_order:
            expert_order = json.loads(Path(args.expert_order).read_text(encoding="utf-8"))
        result = build_candidate(
            inventory_path=args.inventory,
            prune_map_path=prune_map_path,
            recipe=recipe,
            source_root=args.source_root,
            out_dir=args.out,
            seed=args.seed,
            environment_path=args.environment,
            dataset_hashes=list(args.dataset),
            digest_basis=_digest_basis(cfg, args.digest_basis),
            calibration_config_path=args.calibration_config,
            commands=list(args.command) or None,
            parents=list(args.parent),
            expert_order=expert_order,
            kind=args.kind,
            conversion_commit=args.conversion_commit,
            runtime_commit=args.runtime_commit,
        )
    except (BuildError, PruneMapError, OSError, json.JSONDecodeError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    summary = {
        "status": result["verify"]["status"],
        "out_dir": result["out_dir"],
        "resident_weight_bytes": result["build_report"]["resident_weight_bytes"],
        "pure_mxfp4_floor_bytes": result["build_report"]["pure_mxfp4_floor_bytes"],
        "manifest_sha256": result["manifest"]["manifest_sha256"],
        "record": result["record_path"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
