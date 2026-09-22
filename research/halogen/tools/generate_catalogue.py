#!/usr/bin/env python3
"""Generate the Halogen Phase-1 kernel catalogue from one or more inputs.

Inputs may be AMDGPU code objects (relocatable or linked), host ELFs with
embedded device images, clang offload bundles, or raw blobs containing
carved code objects.  Every unrecoverable field becomes null + reason.

Example:
  python3 research/halogen/tools/generate_catalogue.py \\
      $MIMO_LAB/reference/halogen/halogen.so \\
      --out research/halogen/results/kernel-catalogue.json \\
      --toolchain-out research/halogen/results/toolchain.json \\
      --schema research/halogen/schemas/kernel-catalogue.schema.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from halogen_tools import catalogue  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="files to analyse")
    ap.add_argument("--out", required=True, help="catalogue JSON output path")
    ap.add_argument("--toolchain-out", default=None,
                    help="toolchain JSON output path (default: sibling of --out)")
    ap.add_argument("--schema", default=None,
                    help="kernel-catalogue schema; catalogue is validated against it")
    ap.add_argument("--mcpu", default=None,
                    help="force llvm-objdump --mcpu (default: derive from metadata/ELF)")
    ap.add_argument("--no-disasm", action="store_true",
                    help="skip disassembly/classification (metadata-only catalogue)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cat, toolchain, errors = catalogue.build(
        args.inputs,
        schema_path=args.schema,
        mcpu=args.mcpu,
        no_disasm=args.no_disasm,
    )

    if errors["schema"]:
        for e in errors["schema"]:
            print(f"SCHEMA: {e}", file=sys.stderr)
        print("catalogue NOT written: schema validation failed", file=sys.stderr)
        return 2

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _write_verified_json(out_path, cat)
    tc_path = args.toolchain_out or os.path.join(
        os.path.dirname(os.path.abspath(out_path)), "toolchain.json"
    )
    _write_verified_json(tc_path, toolchain)

    nulls = sum(len(r["null_reasons"]) for r in cat["kernels"])
    if not args.quiet:
        print(f"inputs:     {len(cat['inputs'])}")
        print(f"kernels:    {len(cat['kernels'])}")
        print(f"null fields:{nulls:>4} (each carries a reason)")
        print(f"catalogue:  {out_path}")
        print(f"toolchain:  {tc_path}")
        d = toolchain["llvm_objdump"]
        print(f"llvm-objdump: {d.get('path')} ({d.get('version') or d.get('error')})")
        for e in errors["inputs"]:
            print(f"note: {e}")
    return 0


def _write_verified_json(path: str, obj) -> None:
    """Write JSON, then read it back and re-parse (guards file-write hazards)."""
    text = json.dumps(obj, indent=2, sort_keys=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(path, "r", encoding="utf-8") as fh:
        roundtrip = json.load(fh)
    if roundtrip != obj:
        raise RuntimeError(f"read-back verification failed for {path}")


if __name__ == "__main__":
    raise SystemExit(main())
