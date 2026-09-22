#!/usr/bin/env python3
"""Whole-blob disassembly -> per-kernel evidence index (results/disasm_index.json).

One llvm-objdump pass per code object; symbols split from output headers.
Per kernel: ISA byte size, instruction count, motif-flag mnemonics with
kernel-relative offsets, and a short head excerpt (clean-room: bounded
excerpts only, full dumps stay local under --work).

Usage:
  python build_disasm_index.py --blobs <dir> --out results/disasm_index.json \
      [--work <local-only-dir>] [--mcpu gfx1151]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

OBJDUMP = "/opt/homebrew/opt/llvm/bin/llvm-objdump"
NM = "/opt/homebrew/opt/llvm/bin/llvm-nm"
SYM_RE = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
# llvm-objdump 23 default: "\t<mnemonic> <operands>\t// <ADDR>: <BYTES>"
INSN_RE = re.compile(r"^\s+(\S+).*//\s*([0-9A-Fa-f]+):\s*([0-9A-Fa-f ]+)\s*$")
# classic layout fallback: "\t<addr>:\t<bytes>\t<mnemonic> <operands>"
INSN_A = re.compile(
    r"^\s+([0-9a-f]+):\s+((?:[0-9a-f]{2,8} ?)+?)\s+(\S.*)$")

# motif probes: short mnemonic-prefix/regex -> flag
MOTIFS = {
    "dot2_bf16": re.compile(r"^v_dot2_f32_bf16"),
    "dot_acc": re.compile(r"^v_dot[2358].*acc|v_dot_acc"),
    "wmma_f32_f16": re.compile(r"^v_wmma_f32_16x16x16_f16"),
    "wmma_f32_bf16": re.compile(r"^v_wmma_f32_16x16x16_bf16"),
    "wmma_other": re.compile(r"^v_wmma_(?!f32_16x16x16_(f16|bf16))"),
    "mfma": re.compile(r"^v_mfma_"),
    "exp": re.compile(r"^v_exp_f32"),
    "log": re.compile(r"^v_log_f32"),
    "rcp": re.compile(r"^v_rcp_f32|v_div_fixup|v_div_scale|v_div_fmas"),
    "ldexp": re.compile(r"^v_ldexp_f32"),
    "ds_read": re.compile(r"^ds_(read|load)"),
    "ds_write": re.compile(r"^ds_(write|store)"),
    "ds_permute": re.compile(r"^ds_bpermute|ds_swizzle"),
    "global_load_typed": re.compile(r"^global_load_[usi]?(8|16|32|64|128)|^global_load_d16"),
    "global_store_d16": re.compile(r"^global_store_d16"),
    "buffer_load": re.compile(r"^buffer_load"),
    "buffer_store": re.compile(r"^buffer_store"),
    "global_atomic": re.compile(r"^global_atomic"),
    "flat_scan": re.compile(r"^flat_(load|store|atomic)"),
    "bitfield": re.compile(r"^v_bfe_|^v_bfi_|^v_lshlrev|^v_lshrrev|^v_and_b32|^v_or_b32"),
    "cvt_f16": re.compile(r"^v_cvt_f16_|^v_cvt_pkrt|^v_cvt_pk_bf16"),
    "cvt_bf16": re.compile(r"^v_cvt_bf16|bf16"),
    "fma_mix": re.compile(r"^v_fma_mix"),
    "dual_issue": re.compile(r"^v_dual_"),
    "s_barrier": re.compile(r"^s_barrier"),
    "s_clause": re.compile(r"^s_clause"),
    "s_memrealtime": re.compile(r"^s_memrealtime|s_memtime"),
    "cmp": re.compile(r"^v_cmp|^v_cndmask"),
    "permute_lane": re.compile(r"^v_readfirstlane|^v_readlane|^v_writelane|^v_perm"),
    "pack_pk": re.compile(r"^v_pk_"),
    "mad_u64": re.compile(r"^v_mad_u64|^v_mad_i64"),
}


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:180]


def detect_mcpu(data: bytes, default: str) -> str:
    """Read amdhsa.target from the .note msgpack (fast, no LLVM needed)."""
    try:
        import struct
        import msgpack
        from elftools.elf.elffile import ELFFile
        import io
        ef = ELFFile(io.BytesIO(data))
        sec = ef.get_section_by_name(".note") or next(
            (s for s in ef.iter_sections() if s["sh_type"] == "SHT_NOTE"), None)
        if sec is None:
            return default
        blob = sec.data()
        pos = 0
        while pos + 12 <= len(blob):
            namesz, descsz, ntype = struct.unpack_from("<III", blob, pos)
            name = blob[pos + 12:pos + 12 + namesz]
            name_end = (pos + 12 + namesz + 3) & ~3
            desc = blob[name_end:name_end + descsz]
            if name.rstrip(b"\0") == b"AMDGPU" and ntype == 32:
                m = msgpack.unpackb(desc, raw=False, strict_map_key=False)
                tgt = m.get("amdhsa.target", "")
                if "--" in tgt:
                    return tgt.rsplit("--", 1)[1] or default
            pos = (name_end + descsz + 3) & ~3
    except Exception:
        pass
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blobs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--mcpu", default="gfx1151")
    args = ap.parse_args()

    # symbol sizes + kernel metadata correlation via nm
    index = []
    blobs = sorted(args.blobs.glob("*.co"))
    for bp in blobs:
        data = bp.read_bytes()
        if data[:4] != b"\x7fELF":
            continue
        sha = hashlib.sha256(data).hexdigest()
        mcpu = detect_mcpu(data, args.mcpu)
        # symbol values + sizes via pyelftools (st_size is the ISA byte size)
        syms: dict[str, dict] = {}
        try:
            import io
            from elftools.elf.elffile import ELFFile
            from elftools.elf.sections import SymbolTableSection
            ef = ELFFile(io.BytesIO(data))
            for sec in ef.iter_sections():
                if isinstance(sec, SymbolTableSection):
                    for sym in sec.iter_symbols():
                        if sym.name and sym["st_info"]["type"] in (
                                "STT_FUNC", "STT_NOTYPE") and sym["st_value"]:
                            prev = syms.get(sym.name, {})
                            syms[sym.name] = {
                                "size": max(sym["st_size"], prev.get("size") or 0),
                                "value": sym["st_value"]}
        except Exception as e:
            index.append({"blob": sha, "unrecoverable": f"elf symtab failed: {e}"})
            continue
        # whole-blob disasm
        try:
            proc = subprocess.run(
                [OBJDUMP, "-d", f"--mcpu={mcpu}", str(bp)],
                capture_output=True, text=True, timeout=600)
        except Exception as e:
            index.append({"blob": sha, "unrecoverable": f"objdump failed: {e}"})
            continue
        if proc.returncode != 0:
            index.append({"blob": sha, "unrecoverable":
                          (proc.stderr or "objdump rc!=0")[:200]})
            continue
        # split per symbol
        cur = None
        cur_start = 0
        per: dict[str, list] = {}
        order: list[str] = []
        for line in proc.stdout.splitlines():
            m = SYM_RE.match(line)
            if m:
                cur = m.group(2)
                cur_start = int(m.group(1), 16)
                if cur not in per:
                    per[cur] = []
                    order.append(cur)
                continue
            if cur is None:
                continue
            mb = INSN_RE.match(line)
            if mb and not mb.group(1).startswith("0") or (mb and ":" not in mb.group(1)):
                # format B: mnemonic first, address in trailing comment
                if ":" in mb.group(1):
                    mb = None
            if mb:
                addr = int(mb.group(2), 16)
                per[cur].append((addr - cur_start, mb.group(1),
                                 line.split("//", 1)[0].strip()))
                continue
            ma = INSN_A.match(line)
            if ma:
                per[cur].append((int(ma.group(1), 16) - cur_start,
                                 ma.group(3).split()[0],
                                 ma.group(3).strip()))
        # full local dump
        if args.work:
            dd = args.work / "disasm_full"
            dd.mkdir(parents=True, exist_ok=True)
            (dd / (sha[:16] + ".s")).write_text(proc.stdout)
        blob_entry = {"blob": sha, "size": len(data),
                      "mcpu": mcpu, "symbol_count": len(per),
                      "kernels": []}
        # interleave nm values so we know sizes even for zero-insn syms
        for name in sorted(set(list(per)) | set(syms)):
            insns = per.get(name, [])
            if not insns and name not in syms:
                continue
            start = syms.get(name, {}).get("value")
            if start is None and insns:
                start = insns[0][0]
            end = insns[-1][0] + 8 if insns else start
            # instruction-byte span from last insn (approx by next symbol start)
            flags: dict[str, int] = Counter()
            first_hits: dict[str, int] = {}
            for off, mn, _txt in insns:
                for flag, rx in MOTIFS.items():
                    if rx.search(mn):
                        flags[flag] += 1
                        first_hits.setdefault(flag, off)
            head = [f"{o:#x}: {t}" for o, _m, t in insns[:6]]
            k = {
                "name": name,
                "vma": hex(start) if start is not None else None,
                "isa_size": syms.get(name, {}).get("size"),
                "insn_count": len(insns),
                "motif_flags": dict(flags),
                "motif_first_offsets": {k2: hex(v) for k2, v in first_hits.items()},
                "head_excerpt": head,
            }
            blob_entry["kernels"].append(k)
        index.append(blob_entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(index, indent=1)
    args.out.write_text(text)
    # read-back verify (file-write hazard guard)
    if json.loads(args.out.read_text()) != index:
        print("READBACK MISMATCH", file=sys.stderr)
        return 1
    nk = sum(len(b.get("kernels", [])) for b in index)
    print(json.dumps({"blobs": len(index), "kernels_indexed": nk}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
