#!/usr/bin/env python3
"""Static package inspector for Halogen (Phase-1 clean-room archaeology).

Scans a package directory (or single file) for ELF objects, extracts:
  - ELF headers, sections, segments, dynamic deps, imports/exports
  - embedded AMDGPU code objects (HSACO) incl. Clang offload bundles
  - kernel symbols, kernarg layouts, LDS/SGPR/VGPR/wave metadata
  - strings + embedded configuration dumps

Outputs machine-readable JSON under results/. Reads the binary by hash-pinned
path only; never copies binary content into the repo.

Usage: python inspect_package.py --pkg <dir-or-file> --out <results-dir> [--work <local-only-dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import msgpack
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

EM_AMDGPU = 224
AMDGPU_METADATA_NAME = b"AMDGPU"
NT_AMDGPU_METADATA = 32  # 0x20, elf.h: NT_AMDGPU_METADATA
OFFLOAD_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u32(b: bytes, off: int, le: bool = True) -> int:
    return struct.unpack_from("<I" if le else ">I", b, off)[0]


READOBJ = "/opt/homebrew/opt/llvm/bin/llvm-readobj"


def parse_offload_bundles(path: Path, data: bytes) -> list[dict]:
    """List Clang offload-bundle members via llvm-readobj --offloading (ground truth).

    Output lines: '<ident>\\tfile://<path>#offset=<n>&size=<n>'.
    Payload offsets are file offsets; sha256 computed on the extracted bytes.
    """
    import subprocess
    import urllib.parse
    try:
        proc = subprocess.run([READOBJ, "--offloading", str(path)],
                              capture_output=True, text=True, timeout=120)
    except Exception as e:
        return [{"error": f"llvm-readobj failed: {e}"}]
    out = []
    for line in proc.stdout.splitlines():
        if "\t" in line and "file://" in line:
            ident, uri = line.split("\t", 1)
        elif " file://" in line:
            ident, uri = line.rsplit(" file://", 1)
            uri = "file://" + uri
        else:
            continue
        frag = uri.split("#", 1)[-1]
        params = dict(kv.split("=", 1) for kv in frag.split("&") if "=" in kv)
        off = int(params.get("offset", "0"))
        size = int(params.get("size", "0"))
        payload = data[off:off + size]
        out.append({"ident": ident, "offset": off, "size": size,
                    "sha256": hashlib.sha256(payload).hexdigest() if payload else None})
    return out


def find_embedded_elfs(data: bytes, base: int = 0) -> list[tuple[int, int]]:
    """Find ELF magic occurrences; cheap heuristic filter for valid headers."""
    hits = []
    start = 0
    while True:
        i = data.find(b"\x7fELF", start)
        if i < 0:
            break
        if i + 64 <= len(data):
            ei_class = data[i + 4]
            if ei_class in (1, 2):  # 32/64-bit
                hits.append((base + i, i))
        start = i + 1
    return hits


def elf_meta(data: bytes, offset: int) -> dict:
    """Parse ELF header + summary at byte offset using pyelftools on a stream."""
    import io
    bio = io.BytesIO(data[offset:])
    try:
        ef = ELFFile(bio)
    except Exception as e:  # malformed
        return {"error": str(e)}
    meta: dict = {
        "class": {1: "ELF32", 2: "ELF64"}.get(ef.header["e_ident"]["EI_CLASS"], "?"),
        "endian": "little" if ef.header["e_ident"]["EI_DATA"] == "ELFDATA2LSB" else "big",
        "type": ef.header["e_type"],
        "machine": ef.header["e_machine"],
        "entry": hex(ef.header["e_entry"]),
        "osabi": ef.header["e_ident"]["EI_OSABI"],
        "sections": [],
        "segments": [],
    }
    if ef.header["e_machine"] == "EM_AMDGPU" or str(ef.header["e_machine"]) in ("EM_AMDGPU", "224"):
        meta["is_amdgpu"] = True
    SYM_CAP = 5000
    dynsyms: dict[str, dict] = {}
    dynstr = ef.get_section_by_name(".dynstr")

    def _dynstr(off: int) -> str:
        if dynstr is None:
            return f"off={off:#x}"
        try:
            return dynstr.get_string(off)
        except Exception:
            return f"off={off:#x}"

    for s in ef.iter_sections():
        entry = {"name": s.name, "type": str(s["sh_type"]), "size": s["sh_size"],
                 "addr": hex(s["sh_addr"]), "offset": s["sh_offset"]}
        meta["sections"].append(entry)
        if isinstance(s, SymbolTableSection):
            syms = []
            total = 0
            for sym in s.iter_symbols():
                if not sym.name:
                    continue
                total += 1
                if len(syms) < SYM_CAP:
                    syms.append({"name": sym.name, "value": hex(sym["st_value"]),
                                 "size": sym["st_size"], "bind": sym["st_info"]["bind"],
                                 "type": sym["st_info"]["type"],
                                 "shndx": sym["st_shndx"]})
            if syms:
                tbl = {kk: vv for kk, vv in
                       (("symbols", syms), ("total", total)) if kk in ("symbols",)}
                tbl["total_symbols"] = total
                if total > len(syms):
                    tbl["truncated"] = True
                meta.setdefault("symbol_tables", {})[s.name] = tbl
        if s.name in (".dynsym",):
            for sym in s.iter_symbols():
                if sym.name:
                    dynsyms[sym.name] = {"size": sym["st_size"],
                                         "type": sym["st_info"]["type"]}
        if s.name == ".dynamic" and hasattr(s, "iter_tags"):
            dyn = []
            for d in s.iter_tags():
                tag = str(d["d_tag"])
                val = d["d_val"]
                if tag in ("DT_NEEDED", "DT_SONAME", "DT_RPATH", "DT_RUNPATH",
                           "DT_AUXILIARY", "DT_FILTER") and isinstance(val, int):
                    val = _dynstr(val)
                dyn.append({"tag": tag, "val": val})
            meta["dynamic"] = dyn
    for i in range(ef.num_segments()):
        seg = ef.get_segment(i)
        meta["segments"].append({"type": str(seg["p_type"]), "offset": hex(seg["p_offset"]),
                                 "vaddr": hex(seg["p_vaddr"]), "filesz": seg["p_filesz"],
                                 "memsz": seg["p_memsz"], "flags": str(seg["p_flags"])})
    # notes
    meta["notes"] = []
    for s in ef.iter_sections():
        if s["sh_type"] == "SHT_NOTE":
            try:
                for note in s.iter_notes():
                    n = {"name": note["n_name"], "type": str(note["n_type"])}
                    desc = note["n_desc"]
                    if isinstance(desc, str):
                        n["desc"] = desc[:512]
                    elif isinstance(desc, bytes):
                        if desc[:4] == b"AMDG" or note["n_name"] == "AMDGPU":
                            n["desc"] = "<%d bytes amd gpu note>" % len(desc)
                        else:
                            n["desc"] = desc[:512].decode("utf-8", "replace")
                    meta["notes"].append(n)
            except Exception as e:
                meta["notes"].append({"error": str(e), "section": s.name})
    if dynsyms:
        meta["dynsym_count"] = len(dynsyms)
    return meta


def parse_amdgpu_metadata_notes(blob: bytes) -> list[dict]:
    """Walk SHT_NOTE contents manually and msgpack-decode AMDGPU metadata notes."""
    out = []
    pos = 0
    while pos + 12 <= len(blob):
        try:
            namesz, descsz, ntype = struct.unpack_from("<III", blob, pos)
        except struct.error:
            break
        if namesz > 1024 or descsz > 1 << 28:
            break
        name = blob[pos + 12:pos + 12 + namesz]
        name_end = (pos + 12 + namesz + 3) & ~3
        desc = blob[name_end:name_end + descsz]
        next_pos = (name_end + descsz + 3) & ~3
        if name.rstrip(b"\0") == AMDGPU_METADATA_NAME and ntype == NT_AMDGPU_METADATA:
            try:
                out.append(msgpack.unpackb(desc, raw=False, strict_map_key=False))
            except Exception as e:
                out.append({"error": f"msgpack decode failed: {e}", "bytes": descsz})
        pos = next_pos
        if next_pos <= pos - 1 and descsz == 0:
            break
    return out


def normalize_kernel(md: dict) -> dict:
    """Flatten a kernel metadata entry (code object v3/v4 legacy keys or v5 dot-keys)."""
    def g(*keys):
        for kk in keys:
            if kk in md and md[kk] is not None:
                return md[kk]
        return None

    k: dict = {"name": g("Name", ".name"),
               "symbol": g("SymbolName", ".symbol", ".symbol_name")}
    attrs = md.get("Attrs", {}) or {}
    props = md.get("CodeProps", {}) or {}

    def gp(*keys):
        for src in (props, attrs, md):
            for kk in keys:
                if kk in src and src[kk] is not None:
                    return src[kk]
        return None

    k["required_workgroup_size"] = attrs.get("ReqdWorkGroupSize")
    k["workgroup_size_hint"] = attrs.get("WorkgroupSizeHint")
    k["kernarg_segment_size"] = gp("KernargSegmentSize", ".kernarg_segment_size")
    k["kernarg_segment_align"] = gp(".kernarg_segment_align")
    k["group_segment_fixed_size"] = gp("GroupSegmentFixedSize",
                                       ".group_segment_fixed_size")
    k["private_segment_fixed_size"] = gp("PrivateSegmentFixedSize",
                                         ".private_segment_fixed_size")
    k["wavefront_size"] = gp("WavefrontSize", ".wavefront_size")
    k["sgpr_count"] = gp("SGPRCount", ".sgpr_count")
    k["sgpr_spill_count"] = gp(".sgpr_spill_count")
    k["vgpr_count"] = gp("VGPRCount", ".vgpr_count")
    k["vgpr_spill_count"] = gp(".vgpr_spill_count")
    k["max_flat_workgroup_size"] = gp("MaxFlatWorkgroupSize",
                                      ".max_flat_workgroup_size")
    k["max_wave_count"] = gp("MaxWaveCount")
    k["workgroup_processor_mode"] = gp(".workgroup_processor_mode")
    args = props.get("Args") or md.get("Args") or md.get(".args") or []
    kernargs = []
    for a in args:
        kernargs.append({kk.lstrip("."): vv for kk, vv in a.items()})
    k["kernargs"] = kernargs
    # unknown top-level keys kept verbatim (recorded, not guessed)
    known_legacy = {"Name", "SymbolName", "Attrs", "CodeProps", "Args",
                    "Language", "LanguageVersion", "LanguageStd"}
    k["extra"] = {kk: vv for kk, vv in md.items() if kk not in known_legacy}
    return k


def inspect_amdgpu_elf(data: bytes, offset: int, source: str) -> dict:
    import io
    bio = io.BytesIO(data[offset:])
    try:
        ef = ELFFile(bio)
    except Exception as e:
        return {"source": source, "error": str(e)}
    info: dict = {
        "source": source,
        "class": {1: "ELF32", 2: "ELF64"}.get(ef.header["e_ident"]["EI_CLASS"], "?"),
        "type": ef.header["e_type"],
        "machine": str(ef.header["e_machine"]),
        "entry": hex(ef.header["e_entry"]),
        "sections": [{"name": s.name, "size": s["sh_size"], "addr": hex(s["sh_addr"])}
                     for s in ef.iter_sections()],
        "kernels": [],
        "metadata_sources": [],
    }
    if ef.header["e_machine"] not in ("EM_AMDGPU", 224, "224"):
        info["note"] = "not an AMDGPU machine"
    # symbol table -> kernel candidates: functions in .text with nonempty names
    funcs = []
    text_syms = {}
    for s in ef.iter_sections():
        if isinstance(s, SymbolTableSection):
            for sym in s.iter_symbols():
                if not sym.name:
                    continue
                if sym["st_info"]["type"] in ("STT_FUNC", "STT_NOTYPE") and sym["st_size"] > 0:
                    text_syms[sym.name] = {"size": sym["st_size"],
                                           "value": hex(sym["st_value"]),
                                           "section": sym["st_shndx"]}
    info["function_symbols"] = text_syms
    # metadata: .note (v3/v4) or .metadata (v5) section
    for s in ef.iter_sections():
        if s.name in (".note", ".note.amdgpu", ".metadata") or s["sh_type"] == "SHT_NOTE":
            if s.name == ".metadata" or s["sh_type"] == "SHT_NOTE":
                blob = s.data()
                metas = parse_amdgpu_metadata_notes(blob) if s["sh_type"] == "SHT_NOTE" else \
                    [msgpack.unpackb(blob, raw=False, strict_map_key=False)]
                for m in metas:
                    if isinstance(m, dict) and "error" not in m:
                        info["metadata_sources"].append(s.name)
                        kernels = (m.get("amdhsa.kernels") or m.get("Kernels")
                                   or m.get("kernels") or [])
                        for kd in kernels:
                            info["kernels"].append(normalize_kernel(kd))
                        # top-level: amdhsa.target / amdhsa.version etc.
                        info["metadata_header"] = {kk: vv for kk, vv in m.items()
                                                   if kk not in (
                                                       "amdhsa.kernels",
                                                       "Kernels", "kernels")}
                    elif isinstance(m, dict):
                        info.setdefault("metadata_errors", []).append(m)
    # correlate: symbols without metadata and vice versa
    return info


def extract_strings(data: bytes, min_len: int = 6, limit: int = 400000) -> list[dict]:
    pat = re.compile(rb"[\x20-\x7e]{%d,}" % min_len)
    out = []
    for m in pat.finditer(data):
        if len(out) >= limit:
            break
        out.append({"off": m.start(), "s": m.group().decode("ascii", "replace")})
    return out


CONFIG_PAT = re.compile(
    r"(gfx11|gfx9|rocm|hip|hsaco|quant|q[48]|fp8|fp4|moe|topk|top_k|kv_?cache|"
    r"specul|draft|target|weight|gemm|tune|tuning|layout|repack|gather|scatter|"
    r"shard|expert|attn|attention|decode|prefill|batch|chunked|page|block_size|"
    r"flash|rope|rotary|tensor|mma|wmma|ds_write|ds_read|buffer|swizzle)",
    re.IGNORECASE)

# Halogen's own payload (per pin.json inner_files): full strings retained.
PAYLOAD_PREFIXES = (
    "usr/local/bin/flash_serve", "usr/local/bin/halogen-tools",
    "usr/local/bin/entrypoint.sh", "usr/local/bin/halogen-healthcheck",
    "halogen/tools/", "opt/halogen/",
)


def is_payload(rel: str) -> bool:
    return rel.startswith(PAYLOAD_PREFIXES)


def classify_strings(strings: list[dict]) -> dict:
    cfg, keyval, other_interesting = [], [], []
    kv = re.compile(r"^[A-Za-z0-9_.\-/]+[=:].{0,200}$")
    for e in strings:
        s = e["s"]
        if CONFIG_PAT.search(s):
            other_interesting.append(e)
            if kv.match(s):
                keyval.append(e)
                if len(keyval) > 20000:
                    break
        if s.lstrip().startswith(("{", "[")) and len(s) > 8:
            cfg.append(e)
    return {"config_like": cfg[:5000], "key_value": keyval[:20000],
            "keyword_hits_count": len(other_interesting),
            "keyword_hits_sample": other_interesting[:5000]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=None,
                    help="local-only dir for full dumps (never committed)")
    ap.add_argument("--strings", action="store_true", help="also dump full strings")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.work:
        args.work.mkdir(parents=True, exist_ok=True)

    files = [args.pkg] if args.pkg.is_file() else \
        sorted(p for p in args.pkg.rglob("*") if p.is_file())
    inventory = []
    elf_reports = []
    codeobj_reports = []
    bundle_reports = []
    strings_agg: dict[str, list] = {}

    for f in files:
        st = f.stat()
        rel = str(f.relative_to(args.pkg)) if args.pkg.is_dir() else f.name
        entry = {"path": rel, "size": st.st_size, "sha256": sha256_file(f)}
        inventory.append(entry)
        data = f.read_bytes()
        # Clang offload bundles first (they host HSACO payloads)
        if OFFLOAD_MAGIC in data:
            bundles = parse_offload_bundles(f, data)
            bundle_reports.append({"file": rel, "bundles": bundles})
            for b in bundles:
                if "error" in b:
                    continue
                if "amdgcn" in b["ident"] or "gfx" in b["ident"]:
                    payload = data[b["offset"]:b["offset"] + b["size"]]
                    if payload[:4] == b"\x7fELF":
                        if args.work:  # local-only blob dump for disassembly
                            blob_dir = args.work / "blobs"
                            blob_dir.mkdir(parents=True, exist_ok=True)
                            (blob_dir / (b["sha256"] + ".co")).write_bytes(payload)
                        r = inspect_amdgpu_elf(payload, 0, f"{rel}@bundle:{b['ident']}")
                        r["bundle_ident"] = b["ident"]
                        r["payload_sha256"] = b["sha256"]
                        codeobj_reports.append(r)
                    else:
                        codeobj_reports.append({
                            "source": f"{rel}@bundle:{b['ident']}",
                            "unrecoverable": "bundle payload is not an ELF code object",
                            "payload_sha256": b["sha256"], "size": b["size"]})
        # standalone / embedded ELFs
        if data[:4] == b"\x7fELF":
            meta = elf_meta(data, 0)
            meta["file"] = rel
            elf_reports.append(meta)
            if meta.get("is_amdgpu") or meta.get("machine") == "EM_AMDGPU":
                codeobj_reports.append(inspect_amdgpu_elf(data, 0, rel))
        elif data[:4] in (b"PK\x03\x04",):
            entry["kind"] = "zip/jar"
        if args.strings:
            strs = extract_strings(data)
            if not is_payload(rel):
                # third-party: keep only keyword/config matches, bounded
                strs = [e for e in strs if CONFIG_PAT.search(e["s"])][:3000]
                entry["strings_filtered"] = True
            strings_agg[rel] = strs
        # embedded ELF scan for non-ELF containers (so/tar members are pre-extracted)
        if data[:4] != b"\x7fELF":
            emb = find_embedded_elfs(data)
            if emb:
                entry["embedded_elf_offsets"] = [o for o, _ in emb[:50]]
                for off, roff in emb[:50]:
                    sub = data[roff:roff + 64 * 1024 * 1024]
                    if len(sub) > 64:
                        try:
                            m = elf_meta(sub, 0)
                        except Exception:
                            continue
                        if m.get("is_amdgpu"):
                            codeobj_reports.append(inspect_amdgpu_elf(sub, 0, f"{rel}+{off:#x}"))

    (args.out / "inventory.json").write_text(json.dumps(inventory, indent=1))
    (args.out / "elf_structure.json").write_text(json.dumps(elf_reports, indent=1, default=str))
    (args.out / "offload_bundles.json").write_text(json.dumps(bundle_reports, indent=1))
    (args.out / "code_objects.json").write_text(json.dumps(codeobj_reports, indent=1, default=str))
    if args.strings:
        (args.out / "strings.json").write_text(json.dumps(
            {k: {"count": len(v), "entries": v} for k, v in strings_agg.items()}, indent=1))
        all_s = [e for v in strings_agg.values() for e in v]
        (args.out / "config_strings.json").write_text(
            json.dumps(classify_strings(all_s), indent=1))
    print(json.dumps({"files": len(inventory), "elfs": len(elf_reports),
                      "code_objects": len(codeobj_reports),
                      "bundles": len(bundle_reports)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
