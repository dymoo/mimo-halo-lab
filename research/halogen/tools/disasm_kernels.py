#!/usr/bin/env python3
"""Disassemble recoverable gfx1151 kernels from HSACO ELFs.

Extracts each kernel's .text.<name> bytes from code_objects.json entries,
disassembles with llvm-objdump (AMDGPU target), and emits:
  - <work>/disasm/<safe_name>.s        full disassembly (local-only)
  - results/disasm_index.json          per-kernel: bytes, ISA instruction count,
                                       and short evidence head/tail excerpts

Usage:
  python disasm_kernels.py --pkg <dir> --cat <code_objects.json> \
      --out results --work <local-only-dir> [--mcpu gfx1151]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

OBJDUMP = "/opt/homebrew/opt/llvm/bin/llvm-objdump"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:180]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True, type=Path,
                    help="code_objects.json produced by inspect_package.py")
    ap.add_argument("--blobs", required=True, type=Path,
                    help="dir of raw HSACO blobs written by extract step, "
                         "named <sha>.co (one per code object)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--mcpu", default="gfx1151")
    args = ap.parse_args()

    code_objects = json.loads(args.cat.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.work / "disasm").mkdir(parents=True, exist_ok=True)

    index = []
    mcpu = args.mcpu
    for co in code_objects:
        if "error" in co or not co.get("kernels") and not co.get("function_symbols"):
            index.append({"source": co.get("source"), "unrecoverable":
                          co.get("error", "no kernels or symbols")})
            continue
        blob_path = args.blobs / (co.get("payload_sha256") or
                                  safe_name(co["source"]) + ".co")
        if not blob_path.exists():
            index.append({"source": co["source"],
                          "unrecoverable": f"blob missing: {blob_path.name}"})
            continue
        # kernel names -> symbol table sizes
        syms = co.get("function_symbols", {})
        kernel_names = [k["name"] for k in co["kernels"] if k.get("name")] or list(syms)
        for kname in kernel_names:
            sym = syms.get(kname, {})
            sym_bytes = sym.get("size", 0)
            try:
                proc = subprocess.run(
                    [OBJDUMP, "-d", f"--mcpu={mcpu}", "--no-show-raw-insn",
                     f"--disassemble-symbols={kname}", str(blob_path)],
                    capture_output=True, text=True, timeout=120)
                text = proc.stdout
            except Exception as e:
                index.append({"kernel": kname, "unrecoverable": f"objdump failed: {e}"})
                continue
            # keep only instruction lines of this symbol
            ins_lines = [ln for ln in text.splitlines()
                         if re.match(r"^\s+[0-9a-f]+:", ln)]
            if not ins_lines and "unknown" in (proc.stderr or "").lower():
                index.append({"kernel": kname, "unrecoverable":
                              "mcpu unsupported by objdump; retry with gfx11 or raw decode",
                              "stderr": (proc.stderr or "")[:300]})
                continue
            outp = args.work / "disasm" / (safe_name(kname) + ".s")
            outp.write_text("\n".join(ins_lines) + "\n")
            index.append({
                "kernel": kname,
                "symbol_size_bytes": sym_bytes,
                "instr_lines": len(ins_lines),
                "disasm": str(outp),
                "excerpts": {
                    "head": ins_lines[:6],
                    "tail": ins_lines[-4:],
                },
            })
    (args.out / "disasm_index.json").write_text(json.dumps(index, indent=1))
    ok = sum(1 for e in index if "instr_lines" in e)
    print(json.dumps({"kernels_indexed": len(index), "disassembled": ok}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
