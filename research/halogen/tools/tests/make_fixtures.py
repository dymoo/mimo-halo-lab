#!/usr/bin/env python3
"""Build SYNTHETIC fixtures + ground truth for the catalogue generator.

Nothing here touches the real Halogen binary.  Fixtures are compiler-made
code objects (brew LLVM: llc -> .s patch -> llvm-mc), hand-built classic
v3 metadata (the toolchain no longer emits code object v3), offload
bundles (clang-offload-bundler), and a garbage input.

After building, the script RUNS the generator against every fixture and
asserts the emitted rows match ground_truth.json (self-check); a mismatch
fails the build.  Ground truth records fixture design values, not parser
output, except where an independent assembler path (-show-encoding) sizes
the bodies.

Usage:
  python3 research/halogen/tools/tests/make_fixtures.py [--llvm-bin DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # tools/
sys.path.insert(0, HERE)                           # tests/ (elfbuild)

import elfbuild  # noqa: E402
from halogen_tools import catalogue, msgpack_lite, validate as V  # noqa: E402

FIXTURE_DIR = os.path.join(HERE, "fixtures")
LLVM_CANDIDATES = ("/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin",
                   "/opt/rocm/llvm/bin")

# ---------------------------------------------------------------------------
# motif bodies (synthetic instruction sequences exercising classifier rules)
# ---------------------------------------------------------------------------
BODY_QGEMM = """\
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	buffer_load_ubyte v10, v0, s[4:7], s0 offen
	buffer_load_ubyte v11, v1, s[4:7], s0 offen
	buffer_load_u16 v12, v2, s[4:7], s0 offen
	buffer_load_dword v13, v3, s[4:7], s0 offen
	v_and_b32 v14, 0xff, v10
	v_lshlrev_b32 v15, 4, v14
	v_or_b32 v16, v15, v11
	v_cvt_f32_ubyte0 v17, v10
	v_cvt_f32_ubyte1 v18, v11
	v_pk_mul_f16 v19, v12, v12
	v_pk_fma_f16 v20, v19, v12, v20
	v_pk_lshrrev_b16 v21, 1, v20
	v_pk_add_f16 v22, v21, v19
	v_dot2_f32_bf16 v23, v13, v14, v23
	v_fma_f32 v24, v23, v17, v18
	buffer_store_dword v24, v15, s[4:7], s0 offen
	s_endpgm"""

BODY_REDUCE = """\
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v5, v0
	ds_read_b32 v6, v1
	ds_write_b32 v0, v5
	v_add_f32 v7, v5, v6
	v_add_f32 v8, v7, v5
	v_add_f32 v9, v8, v6
	v_add_u32 v10, v9, v8
	v_exp_f32 v11, v7
	v_cmp_gt_f32 vcc_lo, v7, v6
	s_and_b32 s5, vcc_lo, s5
	buffer_store_dword v11, v0, s[4:7], s0 offen
	s_endpgm"""

BODY_TINY = "\ts_endpgm"

BODY_CLASSIC = """\
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	buffer_load_dword v5, v0, s[4:7], s0 offen
	s_endpgm"""

BASE_IR = """\
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @{name}(i32 %a) {{
entry:
  ret void
}}
"""


class Toolchain:
    def __init__(self, llvm_bin: str | None):
        self.bin = llvm_bin
        self.versions = {}
        self._cache = {}

    def path(self, tool: str) -> str:
        if self.bin:
            p = os.path.join(self.bin, tool)
            if os.access(p, os.X_OK):
                return p
        raise SystemExit(
            f"fixture build requires {tool}; pass --llvm-bin (brew llvm ships it)"
        )

    def run(self, tool: str, *args: str, input_text: str | None = None,
            check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.path(tool), *args]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              input=input_text, timeout=300)
        if check and proc.returncode != 0:
            raise SystemExit(
                f"{' '.join(cmd)} failed rc={proc.returncode}:\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        return proc

    def version(self, tool: str) -> str:
        if tool not in self.versions:
            proc = self.run(tool, "--version", check=False)
            lines = (proc.stdout or proc.stderr).strip().splitlines()
            self.versions[tool] = lines[0] if lines else "unknown"
        return self.versions[tool]


def write_verified(path: str, data) -> None:
    mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
    with open(path, mode) as fh:
        fh.write(data)
    # read-back verification (file-write hazard guard)
    with open(path, "rb") as fh:
        back = fh.read()
    if isinstance(data, (bytes, bytearray)):
        assert back == bytes(data), f"read-back mismatch: {path}"
    else:
        assert back.decode() == data, f"read-back mismatch: {path}"


# ---------------------------------------------------------------------------
# asm patching helpers
# ---------------------------------------------------------------------------
def make_asm(tc: Toolchain, name: str, workdir: str) -> str:
    ll = os.path.join(workdir, f"{name}.ll")
    s = os.path.join(workdir, f"{name}.s")
    write_verified(ll, BASE_IR.format(name=name))
    tc.run("llc", "-filetype=asm", "-mtriple=amdgcn-amd-amdhsa",
           "-mcpu=gfx1151", ll, "-o", s)
    with open(s, "r", encoding="utf-8") as fh:
        text = fh.read()
    return text.replace("fx_probe", name)


def patch_directive(text: str, directive: str, value) -> str:
    pat = re.compile(rf"(\.amdhsa_{re.escape(directive)}\s+)\S+")
    new, n = pat.subn(lambda m: f"{m.group(1)}{value}", text)
    if n != 1:
        raise SystemExit(f"directive .amdhsa_{directive}: {n} matches (want 1)")
    return new


def patch_meta(text: str, key: str, value) -> str:
    pat = re.compile(rf"(\.{re.escape(key)}:\s*)\d+")
    new, n = pat.subn(lambda m: f"{m.group(1)}{value}", text)
    if n != 1:
        raise SystemExit(f"metadata key .{key}: {n} matches (want 1)")
    return new


def drop_metadata_region(text: str) -> str:
    pat = re.compile(r"^\t\.amdgpu_metadata\n.*?^\t\.end_amdgpu_metadata\n",
                     re.S | re.M)
    new, n = pat.subn("", text)
    if n != 1:
        raise SystemExit(f"metadata region: {n} matches (want 1)")
    return new


def insert_body(text: str, body: str) -> str:
    idx = text.find("\ts_endpgm")
    if idx < 0:
        raise SystemExit("no s_endpgm anchor in asm")
    return text[:idx] + body + text[idx + len("\ts_endpgm"):]


def assemble(tc: Toolchain, asm_path: str, out_path: str) -> None:
    tc.run("llvm-mc", "-filetype=obj", "-triple=amdgcn-amd-amdhsa",
           "-mcpu=gfx1151", asm_path, "-o", out_path)


def body_size(tc: Toolchain, body: str, workdir: str) -> int:
    """Independent ground-truth sizing via llvm-mc -show-encoding."""
    path = os.path.join(workdir, "size_probe.s")
    write_verified(path, "\t.text\nprobe:\n" + body + "\n")
    proc = tc.run("llvm-mc", "-triple=amdgcn", "-mcpu=gfx1151",
                  "-show-encoding", path)
    total = 0
    for m in re.finditer(r"encoding:\s*\[([^\]]*)\]", proc.stdout):
        total += len([x for x in m.group(1).split(",") if x.strip()])
    if total == 0:
        raise SystemExit("no encodings parsed for size probe")
    return total


# ---------------------------------------------------------------------------
# fixture construction
# ---------------------------------------------------------------------------
def surgery(path_in: str, path_out: str, *, drop_keys=(), drop_symbol_suffixes=()) -> None:
    """Rebuild a compiler object with kernel-metadata keys removed and/or
    symbols dropped (mc refuses to assemble metadata missing required keys,
    so the missing-field fixture is produced by msgpack surgery here).

    Relocation sections are not carried over: the catalogue reads code,
    symbols, notes, and the descriptor - never relocations.
    """
    from halogen_tools import elfr

    data = open(path_in, "rb").read()
    elf = elfr.parse(data, path_in)

    # transform metadata note
    kept_sections = []
    note_replaced = False
    for sec in elf.sections:
        if sec.index == 0:
            continue
        if sec.name in (".shstrtab", ".strtab", ".symtab") or sec.sh_type == elfr.SHT_RELA:
            continue
        blob = elf.section_data(sec)
        if sec.sh_type == elfr.SHT_NOTE and blob:
            note = _parse_single_note(blob)
            if note is not None and note[2] == 32:
                desc = note[3]
                root = msgpack_lite.decode(desc)
                changed = False
                for entry in root.get("amdhsa.kernels", []) if isinstance(root, dict) else []:
                    for key in drop_keys:
                        if key in entry:
                            del entry[key]
                            changed = True
                if changed:
                    desc = msgpack_lite.encode(root)
                    namesz = len(note[1].encode()) + 1
                    owner = note[1].encode() + b"\x00"
                    new_note = struct.pack("<III", namesz, len(desc), 32) + owner
                    new_note += b"\x00" * ((-len(new_note)) % 4) + desc
                    new_note += b"\x00" * ((-len(new_note)) % 4)
                    blob = new_note
                    note_replaced = True
        kept_sections.append({
            "name": sec.name, "type": sec.sh_type, "flags": sec.flags,
            "data": blob, "align": sec.align or 1,
            "info": sec.info, "link": 0, "entsize": sec.entsize,
            "addr": 0,
        })
    if drop_keys and not note_replaced:
        raise SystemExit(f"surgery: metadata note not rewritten for {path_in}")

    # section index remap: user sections become ELF indices 1..n
    old_to_new = {}
    new_idx = 1
    for sec in elf.sections:
        if sec.index == 0:
            continue
        if sec.name in (".shstrtab", ".strtab", ".symtab") or sec.sh_type == elfr.SHT_RELA:
            continue
        old_to_new[sec.index] = new_idx
        new_idx += 1

    syms = []
    for sym in elf.symbols:
        if drop_symbol_suffixes and any(
            sym.name.endswith(sfx) for sfx in drop_symbol_suffixes
        ):
            continue
        shndx = sym.shndx
        if shndx in old_to_new:
            shndx = old_to_new[shndx]
        elif shndx >= len(elf.sections) or shndx == 0:
            pass  # SHN_UNDEF / SHN_ABS passthrough
        else:
            continue  # symbol pointed at a dropped section
        syms.append({
            "name": sym.name, "value": sym.value, "size": sym.size,
            "shndx": shndx, "bind": sym.bind, "type": sym.sym_type,
            "other": sym.other,
        })
    if not syms:
        syms = [{"name": "", "value": 0, "size": 0, "shndx": 0,
                 "bind": 0, "type": 0, "other": 0}]

    out = elfbuild.build_relocatable(
        kept_sections, syms,
        e_flags=elf.e_flags, osabi=elf.osabi, abiversion=elf.abiversion,
    )
    write_verified(path_out, out)


def _parse_single_note(blob: bytes):
    """-> (namesz, owner, type, desc) for the first note in a section."""
    if len(blob) < 12:
        return None
    namesz, descsz, ntype = struct.unpack_from("<III", blob, 0)
    if 12 + namesz + descsz > len(blob):
        return None
    owner = blob[12 : 12 + namesz].split(b"\x00", 1)[0].decode("utf-8", "replace")
    desc_off = 12 + ((namesz + 3) // 4) * 4
    return namesz, owner, ntype, blob[desc_off : desc_off + descsz]


def build(llvm_bin: str | None) -> dict:
    tc = Toolchain(llvm_bin)
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="halogen_fixtures_")
    gt: dict = {
        "_toolchain": {},
        "fixtures": {},
    }
    for tool in ("llc", "llvm-mc", "llvm-objdump", "clang-offload-bundler",
                 "llvm-objcopy"):
        try:
            gt["_toolchain"][tool] = tc.version(tool)
        except SystemExit:
            gt["_toolchain"][tool] = "unavailable"
    gt["_toolchain"]["notes"] = (
        "synthetic fixtures only; built with brew LLVM on the dev host; "
        "no real Halogen binary involved"
    )

    # ---- fixture A: wave32 quantized-GEMM motif ------------------------
    name_a = "fx_qgemm_w32"
    s_a = make_asm(tc, name_a, workdir)
    for d, v in (("group_segment_fixed_size", 32768),
                 ("kernarg_size", 128),
                 ("next_free_vgpr", 48),
                 ("next_free_sgpr", 40)):
        s_a = patch_directive(s_a, d, v)
    for k, v in (("group_segment_fixed_size", 32768),
                 ("kernarg_segment_size", 128),
                 ("vgpr_count", 48),
                 ("sgpr_count", 40),
                 ("max_flat_workgroup_size", 256)):
        s_a = patch_meta(s_a, k, v)
    s_a = insert_body(s_a, BODY_QGEMM)
    asm_a = os.path.join(workdir, f"{name_a}.s")
    write_verified(asm_a, s_a)
    fix_a = os.path.join(FIXTURE_DIR, f"{name_a}.o")
    assemble(tc, asm_a, fix_a)
    gt["fixtures"][f"{name_a}.o"] = {
        "kernels": [{
            "name": name_a,
            "vgpr": 48, "sgpr": 40, "lds": 32768, "wave_size": 32,
            "kernarg_segment_size": 128, "max_flat_workgroup_size": 256,
            "private_segment_fixed_size": 0,
            "likely_operation": "fused_dequant_gemm", "min_confidence": 0.5,
            "isa_size": body_size(tc, BODY_QGEMM, workdir),
            "kd_preload": {"length": 0, "offset": 0},
            "sources": {"vgpr": "metadata", "lds": "metadata"},
        }],
        "input_error": False,
    }

    # ---- fixture B: wave64 reduction motif -----------------------------
    name_b = "fx_reduce_w64"
    s_b = make_asm(tc, name_b, workdir)
    for d, v in (("group_segment_fixed_size", 8192),
                 ("kernarg_size", 64),
                 ("next_free_vgpr", 32),
                 ("next_free_sgpr", 24),
                 ("wavefront_size32", 0)):
        s_b = patch_directive(s_b, d, v)
    for k, v in (("group_segment_fixed_size", 8192),
                 ("kernarg_segment_size", 64),
                 ("vgpr_count", 32),
                 ("sgpr_count", 24),
                 ("wavefront_size", 64)):
        s_b = patch_meta(s_b, k, v)
    s_b = insert_body(s_b, BODY_REDUCE)
    asm_b = os.path.join(workdir, f"{name_b}.s")
    write_verified(asm_b, s_b)
    fix_b = os.path.join(FIXTURE_DIR, f"{name_b}.o")
    assemble(tc, asm_b, fix_b)
    gt["fixtures"][f"{name_b}.o"] = {
        "kernels": [{
            "name": name_b,
            "vgpr": 32, "sgpr": 24, "lds": 8192, "wave_size": 64,
            "kernarg_segment_size": 64, "max_flat_workgroup_size": 1024,
            "private_segment_fixed_size": 0,
            "likely_operation": "reduction", "min_confidence": 0.6,
            "isa_size": body_size(tc, BODY_REDUCE, workdir),
            "kd_preload": {"length": 0, "offset": 0},
            "sources": {"wave_size": "metadata"},
        }],
        "input_error": False,
    }

    # ---- fixture C: metadata missing vgpr + wavefront keys -------------
    name_c = "fx_missing_fields"
    s_c = make_asm(tc, name_c, workdir)
    for d, v in (("group_segment_fixed_size", 4096),
                 ("kernarg_size", 96),
                 ("next_free_sgpr", 16)):
        s_c = patch_directive(s_c, d, v)
    # mc validates that required metadata keys are present at assembly time,
    # so the missing keys are dropped later by msgpack surgery below
    for k, v in (("group_segment_fixed_size", 4096),
                 ("kernarg_segment_size", 96),
                 ("sgpr_count", 16),
                 ("max_flat_workgroup_size", 512)):
        s_c = patch_meta(s_c, k, v)
    asm_c = os.path.join(workdir, f"{name_c}.s")
    write_verified(asm_c, s_c)
    fix_c_raw = os.path.join(workdir, f"{name_c}_raw.o")
    assemble(tc, asm_c, fix_c_raw)
    fix_c = os.path.join(FIXTURE_DIR, f"{name_c}.o")
    # msgpack surgery: drop the two keys AND the descriptor symbol so the
    # kd fallback cannot mask the missing-metadata nulls
    surgery(
        fix_c_raw, fix_c,
        drop_keys=(".vgpr_count", ".wavefront_size"),
        drop_symbol_suffixes=(".kd",),
    )
    gt["fixtures"][f"{name_c}.o"] = {
        "kernels": [{
            "name": name_c,
            "vgpr": None, "sgpr": 16, "lds": 4096, "wave_size": None,
            "kernarg_segment_size": 96, "max_flat_workgroup_size": 512,
            "private_segment_fixed_size": 0,
            "likely_operation": "unclassified", "min_confidence": 0.0,
            "isa_size": body_size(tc, BODY_TINY, workdir),
            "kd_preload": None,
            "nulls": {
                "vgpr": "absent or null in kernel metadata",
                "wave_size": "absent or null in kernel metadata",
                "kd_preload": "descriptor not found",
            },
        }],
        "input_error": False,
    }

    # ---- fixture D: no metadata note at all (kd-only fallback) ---------
    name_d = "fx_nometa"
    s_d = make_asm(tc, name_d, workdir)
    for d, v in (("group_segment_fixed_size", 4096),
                 ("kernarg_size", 128),
                 ("next_free_vgpr", 16),
                 ("wavefront_size32", 1)):
        s_d = patch_directive(s_d, d, v)
    s_d = drop_metadata_region(s_d)
    asm_d = os.path.join(workdir, f"{name_d}.s")
    write_verified(asm_d, s_d)
    fix_d = os.path.join(FIXTURE_DIR, f"{name_d}.o")
    assemble(tc, asm_d, fix_d)
    gt["fixtures"][f"{name_d}.o"] = {
        "kernels": [{
            "name": name_d,
            "vgpr": 16, "sgpr": None, "lds": 4096, "wave_size": 32,
            "kernarg_segment_size": 128, "max_flat_workgroup_size": None,
            "private_segment_fixed_size": 0,
            "likely_operation": "unclassified", "min_confidence": 0.0,
            "isa_size": body_size(tc, BODY_TINY, workdir),
            "kd_preload": {"length": 0, "offset": 0},
            "sources": {"vgpr": "kd", "lds": "kd", "wave_size": "kd"},
            "nulls": {
                "sgpr": "granule encodes 0",
                "max_flat_workgroup_size": "not derivable from the kernel descriptor",
            },
        }],
        "input_error": False,
    }

    # ---- fixture E: classic code object v3 metadata (hand-built ELF) ---
    name_e = "fx_v3_classic"
    body_e_path = os.path.join(workdir, f"{name_e}_body.s")
    write_verified(body_e_path, "\t.text\n" + BODY_CLASSIC + "\n")
    proc = tc.run("llvm-mc", "-triple=amdgcn", "-mcpu=gfx1151", "-show-encoding",
                  body_e_path)
    text_bytes = bytearray()
    for m in re.finditer(r"encoding:\s*\[([^\]]*)\]", proc.stdout):
        for hx in m.group(1).split(","):
            hx = hx.strip()
            if hx:
                text_bytes.append(int(hx, 16))
    if not text_bytes:
        raise SystemExit("classic body produced no encodings")
    classic_meta = {
        "Version": [1, 0],
        "Printf": [],
        "Kernels": [{
            "Name": name_e,
            "SymbolName": name_e,
            "Language": "OpenCL C",
            "LanguageVersion": [2, 0],
            "KernargSegmentSize": 64,
            "GroupSegmentFixedSize": 2048,
            "PrivateSegmentFixedSize": 0,
            "WavefrontSize": 64,
            "SGPRCount": 24,
            "VGPRCount": 32,
            "MaxFlatWorkgroupSize": 256,
        }],
    }
    desc = msgpack_lite.encode(classic_meta)
    note = struct.pack("<III", 4, len(desc), 32) + b"AMD\x00" + desc
    note += b"\x00" * ((-len(note)) % 4)
    elf_bytes = elfbuild.build_relocatable(
        [
            {"name": ".text", "type": 1, "flags": 6,
             "data": bytes(text_bytes), "align": 8},
            {"name": ".note", "type": 7, "flags": 0, "data": note, "align": 4},
        ],
        [
            {"name": name_e, "value": 0, "size": len(text_bytes),
             "shndx": 1, "bind": 1, "type": 2, "other": 3},
        ],
        e_flags=0x4A, osabi=64, abiversion=1,
    )
    fix_e = os.path.join(FIXTURE_DIR, f"{name_e}.o")
    write_verified(fix_e, elf_bytes)
    gt["fixtures"][f"{name_e}.o"] = {
        "kernels": [{
            "name": name_e,
            "vgpr": 32, "sgpr": 24, "lds": 2048, "wave_size": 64,
            "kernarg_segment_size": 64, "max_flat_workgroup_size": 256,
            "private_segment_fixed_size": 0,
            "likely_operation": "unclassified", "min_confidence": 0.0,
            "isa_size": len(text_bytes),
            "kd_preload": None,
            "schema": "classic",
            "nulls": {"kd_preload": "descriptor not found"},
        }],
        "input_error": False,
    }

    # ---- fixture F: clang offload bundle (host + hip gfx1151) ----------
    host_payload = os.path.join(workdir, "host_dummy.o")
    write_verified(host_payload, "host_dummy_payload_not_an_elf")
    fix_f = os.path.join(FIXTURE_DIR, "fx_fat_bundle.bin")
    tc.run("clang-offload-bundler",
           "--targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151",
           "--type=o", f"--input={host_payload}", f"--input={fix_a}",
           f"--output={fix_f}")
    gt["fixtures"]["fx_fat_bundle.bin"] = {
        "kernels": [dict(gt["fixtures"][f"{name_a}.o"]["kernels"][0])],
        "bundles": [{
            "format": "clang-offload-bundle",
            "entry_ids": [
                "host-x86_64-unknown-linux-gnu-",
                "hipv4-amdgcn-amd-amdhsa--gfx1151",
            ],
        }],
        "input_error": False,
    }

    # ---- fixture G: host ELF embedding the bundle container ------------
    with open(fix_f, "rb") as fh:
        bundle_blob = fh.read()
    fix_g = os.path.join(FIXTURE_DIR, "fx_hostelf_bundle.o")
    write_verified(fix_g, elfbuild.build_relocatable(
        [{"name": ".offload_fatbin", "type": 1, "flags": 2,
          "data": bundle_blob, "align": 8}],
        [],
        e_flags=0, osabi=0, abiversion=0, machine=62,  # EM_X86_64 host object
    ))
    gt["fixtures"]["fx_hostelf_bundle.o"] = {
        "kernels": [dict(gt["fixtures"][f"{name_a}.o"]["kernels"][0])],
        "bundles": [{"format": "clang-offload-bundle", "entry_ids": None}],
        "kind": "host-elf",
        "input_error": False,
    }

    # ---- fixture H: garbage ---------------------------------------------
    garbage = bytes((i * 17 + 41) % 256 for i in range(8192))
    assert b"\x7fELF" not in garbage and b"__CLANG_OFFLOAD" not in garbage
    fix_h = os.path.join(FIXTURE_DIR, "fx_garbage.bin")
    write_verified(fix_h, garbage)
    gt["fixtures"]["fx_garbage.bin"] = {
        "kernels": [],
        "input_error_contains": "no AMDGPU code object found",
    }

    # ---- self-check: run the generator, compare against ground truth ----
    gt_path = os.path.join(FIXTURE_DIR, "ground_truth.json")
    write_verified(gt_path, json.dumps(gt, indent=2) + "\n")
    verify(gt_path)
    with open(gt_path, "r", encoding="utf-8") as fh:
        json.load(fh)
    print(f"fixtures built + self-check OK in {FIXTURE_DIR}")
    for name in sorted(os.listdir(FIXTURE_DIR)):
        print(f"  {name}")
    return gt


def verify(gt_path: str) -> None:
    with open(gt_path, "r", encoding="utf-8") as fh:
        gt = json.load(fh)
    schema_path = os.path.join(
        HERE, "..", "..", "schemas", "kernel-catalogue.schema.json"
    )
    paths = [os.path.join(FIXTURE_DIR, name) for name in gt["fixtures"]]
    cat, _tc, errors = catalogue.build(paths, schema_path=schema_path)
    problems: list[str] = []
    problems += [f"schema: {e}" for e in errors["schema"]]

    rows = {(os.path.basename(r["input"]), r["name"]): r for r in cat["kernels"]}
    inputs = {os.path.basename(i["path"]): i for i in cat["inputs"]}

    for fname, spec in gt["fixtures"].items():
        if "input_error_contains" in spec:
            iinfo = inputs.get(fname)
            if iinfo is None:
                problems.append(f"{fname}: input missing from catalogue")
                continue
            joined = " ".join(iinfo["errors"])
            if spec["input_error_contains"] not in joined:
                problems.append(
                    f"{fname}: expected error containing "
                    f"{spec['input_error_contains']!r}, got {iinfo['errors']!r}"
                )
            continue
        for kspec in spec["kernels"]:
            row = rows.get((fname, kspec["name"]))
            if row is None:
                problems.append(f"{fname}: kernel row {kspec['name']!r} missing")
                continue
            for field_name in (
                "vgpr", "sgpr", "lds", "wave_size", "kernarg_segment_size",
                "max_flat_workgroup_size", "private_segment_fixed_size",
                "likely_operation", "isa_size", "kd_preload",
            ):
                if field_name not in kspec:
                    continue
                want = kspec[field_name]
                got = row.get(field_name)
                if want != got:
                    problems.append(
                        f"{fname}:{kspec['name']}.{field_name}: "
                        f"want {want!r} got {got!r}"
                    )
            if "min_confidence" in kspec:
                got = row.get("confidence")
                if got is None or got < kspec["min_confidence"]:
                    problems.append(
                        f"{fname}:{kspec['name']}.confidence: want >= "
                        f"{kspec['min_confidence']} got {got!r}"
                    )
            if "schema" in kspec:
                got_schema = (row.get("metadata") or {}).get("schema")
                if got_schema != kspec["schema"]:
                    problems.append(
                        f"{fname}:{kspec['name']}.metadata.schema: "
                        f"want {kspec['schema']!r} got {got_schema!r}"
                    )
            for field_name, needle in kspec.get("nulls", {}).items():
                if row.get(field_name) is not None:
                    problems.append(
                        f"{fname}:{kspec['name']}.{field_name}: expected null"
                    )
                    continue
                reason = row["null_reasons"].get(field_name, "")
                if needle not in reason:
                    problems.append(
                        f"{fname}:{kspec['name']}.{field_name} reason: "
                        f"want substring {needle!r} got {reason!r}"
                    )
            if "kd_preload" in kspec and kspec["kd_preload"] is not None:
                if row.get("kd_preload") != kspec["kd_preload"]:
                    problems.append(
                        f"{fname}:{kspec['name']}.kd_preload: want "
                        f"{kspec['kd_preload']!r} got {row.get('kd_preload')!r}"
                    )
            for field_name, want_src in kspec.get("sources", {}).items():
                got_src = row.get("field_sources", {}).get(field_name)
                if got_src != want_src:
                    problems.append(
                        f"{fname}:{kspec['name']}.field_sources.{field_name}: "
                        f"want {want_src!r} got {got_src!r}"
                    )
        for bspec in spec.get("bundles", []):
            matching = [
                b for b in inputs[fname]["bundles"]
                if b["format"] == bspec["format"]
            ]
            if not matching:
                problems.append(
                    f"{fname}: bundle {bspec['format']!r} not recorded"
                )
                continue
            if bspec.get("entry_ids") is not None:
                got_ids = [e["id"] for e in matching[0]["entries"]]
                if got_ids != bspec["entry_ids"]:
                    problems.append(
                        f"{fname}: bundle ids want {bspec['entry_ids']!r} "
                        f"got {got_ids!r}"
                    )
        if "kind" in spec:
            got_kind = inputs[fname]["kind"]
            if got_kind != spec["kind"]:
                problems.append(
                    f"{fname}: kind want {spec['kind']!r} got {got_kind!r}"
                )

    if problems:
        raise SystemExit("ground-truth self-check FAILED:\n  " +
                         "\n  ".join(problems))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-bin", default=None,
                    help="directory containing llc/llvm-mc/objcopy/bundler")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-run self-check against committed fixtures")
    args = ap.parse_args()
    llvm_bin = args.llvm_bin
    if llvm_bin is None:
        for cand in LLVM_CANDIDATES:
            if os.access(os.path.join(cand, "llvm-mc"), os.X_OK):
                llvm_bin = cand
                break
    if args.verify_only:
        verify(os.path.join(FIXTURE_DIR, "ground_truth.json"))
        print("ground-truth self-check OK (fixtures unchanged)")
        return 0
    build(llvm_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
