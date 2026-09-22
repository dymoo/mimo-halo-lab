"""AMDGPU code-object discovery, metadata parsing, kernel descriptor decode.

Discovery order for one input file:
  1. the file itself if it is an AMDGPU ELF;
  2. classic clang offload bundles (magic ``__CLANG_OFFLOAD_BUNDLE__``);
  3. llvm OffloadBinary wrappers (magic ``10 FF 10 AD``);
  4. CCOB-compressed bundle payloads (zlib/gzip tried; others -> reason);
  5. raw embedded AMDGPU ELF carving anywhere in the byte stream.

Every kernel field is resolved with provenance and, when unrecoverable,
kept as ``None`` plus a reason string (honest degradation is load-bearing:
tests assert on it).
"""

from __future__ import annotations

import gzip
import struct
import zlib
from dataclasses import dataclass, field

from . import elfr, msgpack_lite

NT_AMDGPU_METADATA = 32
METADATA_OWNERS = {"AMD", "AMDGPU"}
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
OFFLOAD_BINARY_MAGIC = b"\x10\xff\x10\xad"
CCOB_MAGIC = b"CCOB"
KD_SIZE = 64

# AMDHSA kernel descriptor offsets (llvm/Support/AMDHSAKernelDescriptor.h,
# LLVM 23.1.1): keep in sync with elfr mach table for gfx10+ wave bit use.
KD_OFF_GROUP = 0
KD_OFF_PRIVATE = 4
KD_OFF_KERNARG = 8
KD_OFF_ENTRY = 16
KD_OFF_RSRC3 = 44
KD_OFF_RSRC1 = 48
KD_OFF_RSRC2 = 52
KD_OFF_PROPS = 56
KD_OFF_PRELOAD = 58

# metadata key -> canonical field, modern amdhsa.* schema (dotted keys)
MODERN_KEYS = {
    ".name": "name",
    ".symbol": "kd_symbol",
    ".kernarg_segment_size": "kernarg_segment_size",
    ".group_segment_fixed_size": "lds",
    ".private_segment_fixed_size": "private_segment_fixed_size",
    ".sgpr_count": "sgpr",
    ".vgpr_count": "vgpr",
    ".wavefront_size": "wave_size",
    ".max_flat_workgroup_size": "max_flat_workgroup_size",
}
# classic (code object v3-era) schema: candidate keys per canonical field
CLASSIC_KEYS = {
    "name": ("Name", "name"),
    "kd_symbol": ("SymbolName", "symbol_name"),
    "kernarg_segment_size": ("KernargSegmentSize", "kernarg_size"),
    "lds": ("GroupSegmentFixedSize", "group_segment_fixed_size"),
    "private_segment_fixed_size": ("PrivateSegmentFixedSize", "private_segment_fixed_size"),
    "sgpr": ("SGPRCount", "sgpr_count"),
    "vgpr": ("VGPRCount", "vgpr_count"),
    "wave_size": ("WavefrontSize", "wavefront_size"),
    "max_flat_workgroup_size": ("MaxFlatWorkgroupSize", "max_flat_workgroup_size"),
}
CANON_FIELDS = (
    "kernarg_segment_size", "lds", "private_segment_fixed_size",
    "sgpr", "vgpr", "wave_size", "max_flat_workgroup_size",
)


@dataclass
class Field:
    """One resolved kernel field with provenance or an honest reason."""

    value: object | None
    source: str = "none"          # metadata | kd | symbol | none
    detail: str = ""              # evidence detail (key path, struct offset...)
    reason: str | None = None     # set when value is None


@dataclass
class KernelRec:
    name: str
    origin: str                   # input label + code-object provenance
    code_object_index: int
    fields: dict = field(default_factory=dict)   # name -> Field
    code_symbol: elfr.Symbol | None = None
    kd_symbol: elfr.Symbol | None = None
    code: bytes | None = None
    code_reason: str | None = None
    metadata: dict | None = None  # schema/note/amdhsa.version/target
    kd: dict | None = None        # parsed descriptor incl. preload

    def get(self, name: str) -> Field:
        return self.fields.get(name) or Field(None, reason="field never resolved")


@dataclass
class CodeObject:
    origin: str                   # e.g. "file ELF", "bundle entry #1 (...)", "carve @0x1234"
    data: bytes
    offset: int | None            # absolute file offset when applicable
    parse_error: str | None = None
    elf: elfr.Elf | None = None
    target: dict | None = None
    kernels: list = field(default_factory=list)
    notes_meta: list = field(default_factory=list)


@dataclass
class InputAnalysis:
    label: str
    size: int
    sha256: str
    kind: str = "unknown"
    bundles: list = field(default_factory=list)
    code_objects: list = field(default_factory=list)
    errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def analyze(label: str, data: bytes) -> InputAnalysis:
    import hashlib

    res = InputAnalysis(label=label, size=len(data), sha256="sha256:" + hashlib.sha256(data).hexdigest())
    seen_offsets: set[int] = set()
    seen_synthetic: set[str] = set()

    def add_blob(origin: str, blob: bytes, offset: int | None, synthetic_key: str | None = None):
        if offset is not None:
            if offset in seen_offsets:
                return
            seen_offsets.add(offset)
        if synthetic_key is not None:
            if synthetic_key in seen_synthetic:
                return
            seen_synthetic.add(synthetic_key)
        co = CodeObject(origin=origin, data=blob, offset=offset)
        try:
            co.elf = elfr.parse(blob, f"{label}#{origin}")
        except elfr.ElfError as e:
            co.parse_error = str(e)
        res.code_objects.append(co)

    # 0) top-level file
    if data[:4] == b"\x7fELF":
        try:
            top = elfr.parse(data, label)
            if top.is_amdgpu:
                add_blob("file ELF", data, 0)
                res.kind = "amdgpu-elf"
            else:
                res.kind = "host-elf"
        except elfr.ElfError as e:
            res.errors.append(f"top-level ELF parse failed: {e}")
            res.kind = "elf"
    else:
        res.kind = "raw"

    # 1) classic clang offload bundles
    start = 0
    while True:
        i = data.find(BUNDLE_MAGIC, start)
        if i < 0:
            break
        start = i + 1
        entries, err = _parse_classic_bundle(data, i)
        if err:
            res.errors.append(f"bundle @0x{i:x}: {err}")
            continue
        res.bundles.append(
            {
                "format": "clang-offload-bundle",
                "offset": i,
                "entries": [
                    {"index": n, "id": eid, "offset": i + off, "size": size}
                    for n, (off, size, eid) in enumerate(entries)
                ],
            }
        )
        for n, (off, size, eid) in enumerate(entries):
            # entry offsets are relative to the container start (magic position)
            payload = data[i + off : i + off + size]
            origin = f"bundle @0x{i:x} entry #{n} id={eid!r}"
            if elfr.looks_like_amdgpu_elf(payload):
                add_blob(origin, payload, i + off)
            elif payload[:4] == OFFLOAD_BINARY_MAGIC:
                _expand_offload_binary(res, origin, payload, i + off, add_blob)

    # 2) OffloadBinary wrappers not already covered (scan raw)
    start = 0
    while True:
        i = data.find(OFFLOAD_BINARY_MAGIC, start)
        if i < 0:
            break
        start = i + 1
        ok = _expand_offload_binary(
            res, f"offload-binary @0x{i:x}", data[i:], i, add_blob
        )
        if not ok:
            res.errors.append(f"offload-binary @0x{i:x}: header/entries not parseable")

    # 3) CCOB compressed payloads
    start = 0
    while True:
        i = data.find(CCOB_MAGIC, start)
        if i < 0:
            break
        start = i + 1
        raw, err = _decompress_ccob(data[i:])
        key = f"ccob@{i}"
        if err:
            res.errors.append(f"ccob @0x{i:x}: {err}")
            continue
        res.bundles.append({"format": "ccob-compressed", "offset": i, "decompressed_size": len(raw)})
        # nested classic bundle?
        j = raw.find(BUNDLE_MAGIC)
        if j >= 0:
            entries, berr = _parse_classic_bundle(raw, j)
            if not berr:
                res.bundles.append(
                    {
                        "format": "clang-offload-bundle(inside-ccob)",
                        "offset": i,
                        "entries": [
                            {"index": n, "id": eid, "offset": None, "size": size}
                            for n, (_o, size, eid) in enumerate(entries)
                        ],
                    }
                )
                for n, (off, size, eid) in enumerate(entries):
                    payload = raw[off : off + size]
                    if elfr.looks_like_amdgpu_elf(payload):
                        add_blob(f"ccob @0x{i:x} entry #{n} id={eid!r}", payload, None, f"{key}:{n}")
        elif elfr.looks_like_amdgpu_elf(raw):
            add_blob(f"ccob @0x{i:x} payload", raw, None, key)

    # 4) raw embedded AMDGPU ELF carve (covers anything above that missed)
    start = 0
    while True:
        i = data.find(b"\x7fELF", start)
        if i < 0:
            break
        start = i + 4
        if i in seen_offsets:
            continue
        if not elfr.looks_like_amdgpu_elf(data, i):
            continue
        # structural sanity: section header table in bounds
        try:
            shoff = struct.unpack_from("<Q", data, i + 40)[0]
            shnum = struct.unpack_from("<H", data, i + 60)[0]
        except struct.error:
            res.errors.append(f"carve @0x{i:x}: truncated ELF header")
            continue
        if shoff == 0 or shnum == 0 or i + shoff + shnum * 64 > len(data):
            res.errors.append(f"carve @0x{i:x}: section table out of bounds (skipping)")
            continue
        add_blob(f"carve @0x{i:x}", data[i:], i)

    if not res.code_objects and not res.errors:
        res.errors.append("no AMDGPU code object found in input (no ELF, bundle, or carve hit)")
    return res


def _parse_classic_bundle(data: bytes, magic_off: int):
    """-> ([(offset, size, id)], error|None); offsets are absolute file offsets."""
    p = magic_off + len(BUNDLE_MAGIC)
    try:
        (count,) = struct.unpack_from("<Q", data, p)
        p += 8
        if count > 1_000_000:
            return [], f"implausible entry count {count}"
        out = []
        for _ in range(count):
            off, size, idlen = struct.unpack_from("<QQQ", data, p)
            p += 24
            if idlen > 1 << 20 or p + idlen > len(data):
                return [], f"entry id out of bounds (idlen={idlen})"
            eid = data[p : p + idlen].decode("utf-8", "replace")
            p += idlen
            if off + size > len(data):
                return [], f"entry payload out of bounds (off={off} size={size})"
            out.append((off, size, eid))
        return out, None
    except struct.error as e:
        return [], f"truncated header: {e}"


def _expand_offload_binary(res: InputAnalysis, origin: str, blob: bytes,
                           base: int, add_blob) -> bool:
    """Parse llvm OffloadBinary (magic 10FF10AD) at blob[0]; register images.

    Offsets inside the wrapper are relative to the wrapper start (the
    buffer the runtime hands to OffloadBinary::extractHeader); a fallback
    attempt at absolute-file offsets keeps odd embeddings working.
    """
    if blob[:4] != OFFLOAD_BINARY_MAGIC or len(blob) < 32:
        return False
    try:
        version, total_size, entries_off, entries_count = struct.unpack_from("<IQQQ", blob, 4)
    except struct.error:
        return False
    if version > 4 or entries_count > 100_000 or entries_off + entries_count * 40 > len(blob):
        # Absolute-in-outer-file embeddings still get picked up by the raw
        # AMDGPU ELF carve pass in analyze(); nothing to register here.
        return False
    res.bundles.append(
        {
            "format": "llvm-offload-binary",
            "offset": base,
            "version": version,
            "entries": entries_count,
        }
    )
    for n in range(entries_count):
        eo = entries_off + n * 40
        img_kind, off_kind, flags, str_off, num_str, img_off, img_size = struct.unpack_from(
            "<HHIQQQQ", blob, eo
        )
        if img_off + img_size > len(blob):
            res.errors.append(f"{origin}: entry #{n} image out of bounds")
            continue
        strings = _offload_strings(blob, str_off, num_str, version, 0, len(blob))
        image = blob[img_off : img_off + img_size]
        eid = f"offkind={off_kind} imgkind={img_kind} triple={strings.get('triple')!r} arch={strings.get('arch')!r}"
        if elfr.looks_like_amdgpu_elf(image):
            add_blob(f"{origin} entry #{n} ({eid})", image, base + img_off)
    return True


def _offload_strings(blob: bytes, str_off: int, num_str: int,
                     version: int, lo: int, hi: int) -> dict:
    out = {}
    entry_size = 16 if version < 2 else 24
    try:
        for k in range(num_str):
            p = str_off + k * entry_size
            key_off, val_off = struct.unpack_from("<QQ", blob, p)
            key = _cstr(blob, key_off)
            if key is None:
                continue
            if version >= 2:
                (val_size,) = struct.unpack_from("<Q", blob, p + 16)
                val = (
                    blob[val_off : val_off + val_size]
                    if val_off + val_size <= len(blob)
                    else b""
                )
                out[key] = val.split(b"\x00", 1)[0].decode("utf-8", "replace")
            else:
                v = _cstr(blob, val_off)
                out[key] = v if v is not None else ""
    except (struct.error, IndexError):
        return out
    return out


def _cstr(buf: bytes, off: int):
    if not (0 <= off < len(buf)):
        return None
    end = buf.find(b"\x00", off)
    if end < 0:
        return None
    return buf[off:end].decode("utf-8", "replace")


def _decompress_ccob(blob: bytes):
    """CCOB: magic4, ver u16, method u16, sizes (u32 pre-v3 / u64 v3+), md5 8B."""
    if len(blob) < 20:
        return None, "truncated CCOB header"
    version, method = struct.unpack_from("<HH", blob, 4)
    try:
        if version >= 3:
            total, uncomp = struct.unpack_from("<QQ", blob, 8)
            data_off = 8 + 16 + 8
        else:
            total, uncomp = struct.unpack_from("<II", blob, 8)
            data_off = 8 + 8 + 8
    except struct.error:
        return None, "truncated CCOB sizes"
    # zlib tolerates trailing bytes, so decompress from payload start to EOF;
    # `total`/`uncomp` sizes are recorded by callers from the header only.
    comp = blob[data_off:]
    for name, fn in (
        ("zlib", lambda b: zlib.decompress(b)),
        ("gzip", lambda b: gzip.decompress(b)),
        ("raw-deflate", lambda b: zlib.decompress(b, -15)),
    ):
        try:
            out = fn(comp)
            return out, None
        except Exception:
            continue
    return None, f"decompression failed for method={method} version={version} (no zlib/gzip match; zstd unsupported)"


# ---------------------------------------------------------------------------
# per-code-object parsing
# ---------------------------------------------------------------------------

def parse_code_object(co: CodeObject, origin_index: int) -> None:
    if co.parse_error or co.elf is None:
        return
    elf = co.elf
    co.target = {
        "e_machine": elf.e_machine,
        "is_amdgpu": elf.is_amdgpu,
        "mach": elf.mach,
        "mach_name": elf.mach_name,
        "osabi": elf.osabi,
        "abiversion": elf.abiversion,
        "e_type": elf.e_type,
        "elf_errors": list(elf.errors),
    }

    meta_root, meta_info = _load_metadata(elf)
    co.notes_meta = meta_info
    entries = _kernel_entries(meta_root)

    syms = elf.symbol_map()
    if entries is None:
        if meta_root is None:
            reason = "no AMDGPU metadata note found"
        else:
            reason = _schema_reason(meta_root)
        # symbol-only degradation: surface STT_FUNC symbols as kernels
        func_syms = [
            s for s in elf.symbols
            if s.sym_type == elfr.STT_FUNC and s.bind == 1 and s.name and s.size
        ]
        seen = set()
        for s in func_syms:
            if s.name in seen:
                continue
            seen.add(s.name)
            k = KernelRec(name=s.name, origin=co.origin, code_object_index=origin_index)
            _fill_from_symbol(k, s, elf)
            for f in CANON_FIELDS:
                k.fields[f] = Field(
                    None,
                    reason=(
                        f"{reason}; {f} not derivable from the kernel "
                        "descriptor either"
                    ),
                )
            # descriptor still helps when present (stripped-metadata objects)
            kd_sym = syms.get(f"{s.name}.kd")
            if kd_sym is not None:
                k.kd_symbol = kd_sym
                blob = elf.symbol_code(kd_sym)
                if blob is not None and len(blob) >= KD_SIZE:
                    k.kd = _parse_kd(blob, elf.mach)
                    k.kd["symbol"] = kd_sym.name
                    k.kd["file_offset"] = elf.symbol_file_offset(kd_sym)
                    _apply_kd_fallbacks(k, elf)
            k.metadata = {"schema": None, "error": reason}
            co.kernels.append(k)
        if func_syms:
            co.notes_meta.append({"error": reason})
        return

    schema, entries_list = entries
    for idx, entry in enumerate(entries_list):
        canon = _canonize(entry, schema)
        name = canon.get("name")
        if not isinstance(name, str) or not name:
            co.notes_meta.append({"error": f"kernel entry #{idx} has no usable name; skipped"})
            continue
        k = KernelRec(name=name, origin=co.origin, code_object_index=origin_index)
        k.metadata = {
            "schema": schema,
            "note_section": meta_info[0]["section"] if meta_info else None,
            "note_offset": meta_info[0]["offset"] if meta_info else None,
            "amdhsa_version": meta_info[0].get("amdhsa_version") if meta_info else None,
            "amdhsa_target": meta_info[0].get("amdhsa_target") if meta_info else None,
            "entry_index": idx,
        }

        # code symbol
        code_name = name
        cs = syms.get(code_name)
        if cs is None and schema == "classic":
            cs = syms.get(str(canon.get("kd_symbol") or ""))
        if cs is None:
            # some producers prefix; try exact metadata name first, else fail
            k.code_reason = f"code symbol {code_name!r} not found in ELF symtab"
        else:
            _fill_from_symbol(k, cs, elf)

        # kd symbol: only treat a metadata symbol as the descriptor when it
        # is descriptor-shaped ('*.kd'); classic SymbolName is the code symbol
        kd_sym = None
        meta_kd = canon.get("kd_symbol")
        if (
            schema == "amdhsa"
            and isinstance(meta_kd, str)
            and meta_kd.endswith(".kd")
            and meta_kd in syms
        ):
            kd_sym = syms[meta_kd]
        elif f"{code_name}.kd" in syms:
            kd_sym = syms[f"{code_name}.kd"]
        else:
            sec = elf.section(".amdhsa.kd")
            if sec is not None:
                cands = [
                    s for s in elf.symbols
                    if s.shndx == sec.index and s.name.startswith(code_name)
                ]
                if len(cands) == 1:
                    kd_sym = cands[0]
        k.kd_symbol = kd_sym

        # metadata fields (numeric coercion; unusable values fall through to
        # the absent/null reason loop below)
        for f in CANON_FIELDS:
            if f in canon and canon[f] is not None:
                v = canon[f]
                if f != "wave_size":
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        continue
                k.fields[f] = Field(
                    v, source="metadata",
                    detail=f"schema={schema} key={canon.get('_keys', {}).get(f, '?')}",
                )

        # kd fallback / augmentation
        if kd_sym is not None:
            blob = elf.symbol_code(kd_sym)
            if blob is None or len(blob) < KD_SIZE:
                k.kd = {"error": f"descriptor symbol {kd_sym.name!r} unreadable"}
            else:
                k.kd = _parse_kd(blob, elf.mach)
                k.kd["symbol"] = kd_sym.name
                k.kd["file_offset"] = elf.symbol_file_offset(kd_sym)
                _apply_kd_fallbacks(k, elf)

        for f in CANON_FIELDS:
            cur = k.fields.get(f)
            if cur is None:
                k.fields[f] = Field(
                    None,
                    reason=f"key absent or null in kernel metadata (schema={schema})",
                )
            elif cur.value is None and cur.reason is None:
                cur.reason = f"key absent or null in kernel metadata (schema={schema})"
        co.kernels.append(k)


def _fill_from_symbol(k: KernelRec, sym: elfr.Symbol, elf: elfr.Elf) -> None:
    k.code_symbol = sym
    size = sym.size
    reason = None
    if size == 0:
        size = _successor_size(elf, sym)
        if size is None:
            reason = (
                f"code symbol {sym.name!r} has zero size and no successor symbol "
                "to delimit it"
            )
            size = None
    if reason:
        k.code = None
        k.code_reason = reason
    else:
        code = elf.symbol_code(sym)
        if code is None:
            k.code = None
            k.code_reason = f"code bytes for {sym.name!r} not readable (NOBITS/absent)"
        elif len(code) != size:
            # reader clamps to section; keep what we got, flag mismatch
            k.code = code
            k.code_reason = (
                f"symbol size {size} exceeds section bounds; read {len(code)} bytes"
            )
        else:
            k.code = code
            k.code_reason = None


def _successor_size(elf: elfr.Elf, sym: elfr.Symbol) -> int | None:
    best = None
    for s in elf.symbols:
        if s.shndx != sym.shndx or s.value <= sym.value or not s.name:
            continue
        if best is None or s.value < best.value:
            best = s
    if best is None:
        return None
    return best.value - sym.value


def _parse_kd(blob: bytes, mach: int) -> dict:
    group, priv, kernarg = struct.unpack_from("<III", blob, KD_OFF_GROUP)
    (entry_off,) = struct.unpack_from("<q", blob, KD_OFF_ENTRY)
    r3, r1, r2 = struct.unpack_from("<III", blob, KD_OFF_RSRC3)
    (props,) = struct.unpack_from("<H", blob, KD_OFF_PROPS)
    (preload,) = struct.unpack_from("<H", blob, KD_OFF_PRELOAD)
    return {
        "group_segment_fixed_size": group,
        "private_segment_fixed_size": priv,
        "kernarg_size": kernarg,
        "kernel_code_entry_byte_offset": entry_off,
        "compute_pgm_rsrc3": r3,
        "compute_pgm_rsrc1": r1,
        "compute_pgm_rsrc2": r2,
        "kernel_code_properties": props,
        "kernarg_preload_raw": preload,
        "kernarg_preload": {
            "length": preload & 0x7F,
            "offset": (preload >> 7) & 0x1FF,
        },
        "vgpr_granule": r1 & 0x3F,
        "sgpr_granule": (r1 >> 6) & 0xF,
        "wavefront_size32_bit": (props >> 10) & 1,
        "mach": mach,
    }


GFX10_PLUS_MACHS = {
    0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3d, 0x3e, 0x41, 0x42,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4E,
    0x50, 0x55, 0x57, 0x58, 0x5A, 0x5C, 0x5D, 0x5E,
}


def _apply_kd_fallbacks(k: KernelRec, elf: elfr.Elf) -> None:
    kd = k.kd
    if not kd or "error" in kd:
        return

    def need(name: str) -> bool:
        cur = k.fields.get(name)
        return cur is None or cur.value is None

    if need("lds"):
        k.fields["lds"] = Field(
            kd["group_segment_fixed_size"], source="kd",
            detail="descriptor offset 0 group_segment_fixed_size (exact u32)",
        )
    if need("kernarg_segment_size"):
        k.fields["kernarg_segment_size"] = Field(
            kd["kernarg_size"], source="kd",
            detail="descriptor offset 8 kernarg_size (exact u32)",
        )
    if need("private_segment_fixed_size"):
        k.fields["private_segment_fixed_size"] = Field(
            kd["private_segment_fixed_size"], source="kd",
            detail="descriptor offset 4 private_segment_fixed_size (exact u32)",
        )
    if need("wave_size") and kd["mach"] in GFX10_PLUS_MACHS:
        k.fields["wave_size"] = Field(
            32 if kd["wavefront_size32_bit"] else 64, source="kd",
            detail="kernel_code_properties bit 10 (ENABLE_WAVEFRONT_SIZE32)",
        )
    if need("vgpr"):
        g = kd["vgpr_granule"]
        k.fields["vgpr"] = Field(
            (g + 1) * 8, source="kd",
            detail=(
                f"compute_pgm_rsrc1 granule {g}, decode (gran+1)*8 "
                "(encoder convention calibrated against LLVM 23.1.1: "
                "next_free_vgpr 48 -> granule 5)"
            ),
        )
    if need("sgpr"):
        g = kd["sgpr_granule"]
        if g > 0:
            k.fields["sgpr"] = Field(
                (g + 1) * 8, source="kd",
                detail=f"compute_pgm_rsrc1 granule {g}, decode (gran+1)*8",
            )
        else:
            k.fields["sgpr"] = Field(
                None, source="kd",
                reason=(
                    "descriptor sgpr granule encodes 0 (ambiguous minimal/"
                    "producer-degenerate encoding); exact count needs metadata"
                ),
            )


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def _load_metadata(elf: elfr.Elf):
    """-> (root dict|None, [info...]); first decodable metadata note wins."""
    info = []
    for note in elf.notes:
        if note.note_type != NT_AMDGPU_METADATA or note.owner not in METADATA_OWNERS:
            continue
        entry = {
            "section": note.section,
            "owner": note.owner,
            "offset": note.desc_offset,
            "desc_size": len(note.desc),
        }
        head = note.desc[:32]
        if note.desc.startswith(b"---") or note.desc.startswith(b"%"):
            info.append({**entry, "error": "YAML metadata (code object v2 era) unsupported: expected msgpack"})
            continue
        try:
            root = msgpack_lite.decode(note.desc)
        except msgpack_lite.MsgpackError as e:
            info.append({**entry, "error": f"msgpack decode failed: {e}; head={head.hex()}"})
            continue
        if not isinstance(root, dict):
            info.append({**entry, "error": f"metadata root is {type(root).__name__}, expected map"})
            continue
        entry["top_keys"] = sorted(str(x) for x in root.keys())
        if "amdhsa.version" in root:
            entry["amdhsa_version"] = root.get("amdhsa.version")
        if "amdhsa.target" in root:
            entry["amdhsa_target"] = root.get("amdhsa.target")
        if "Version" in root:
            entry["classic_version"] = root.get("Version")
        info.append(entry)
        return root, info
    return None, info


def _schema_reason(root: dict) -> str:
    keys = sorted(str(k) for k in root.keys())
    return (
        "unrecognized metadata schema: top-level keys "
        f"{keys} (expected 'amdhsa.kernels' or classic 'Kernels')"
    )


def _kernel_entries(root: dict):
    """-> (schema, [entry]) | None when schema not recognized."""
    if not isinstance(root, dict):
        return None
    if isinstance(root.get("amdhsa.kernels"), list):
        return "amdhsa", root["amdhsa.kernels"]
    if isinstance(root.get("Kernels"), list):
        return "classic", root["Kernels"]
    return None


def _canonize(entry: dict, schema: str) -> dict:
    out: dict = {"_keys": {}}
    if not isinstance(entry, dict):
        return out
    if schema == "amdhsa":
        for key, canon_name in MODERN_KEYS.items():
            if key in entry:
                out[canon_name] = entry[key]
                out["_keys"][canon_name] = key
    else:
        for canon_name, candidates in CLASSIC_KEYS.items():
            for key in candidates:
                if key in entry:
                    out[canon_name] = entry[key]
                    out["_keys"][canon_name] = key
                    break
    # normalize wave size
    if "wave_size" in out and out["wave_size"] not in (32, 64):
        try:
            out["wave_size"] = int(out["wave_size"])
        except (TypeError, ValueError):
            out["wave_size"] = None
    return out
