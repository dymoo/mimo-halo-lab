"""Minimal ELF64 reader for AMDGPU code objects (stdlib only).

Scope: ELF64 little-endian (what every AMDGPU code object uses).  Anything
else raises ElfError with a reason string so callers can record honest
null+reason values instead of guessing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

EM_AMDGPU = 224
ELFOSABI_AMDGPU_HSA = 64
ELFOSABI_AMDGPU_PAL = 65
ELFOSABI_AMDGPU_MESA3D = 66
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_NOTE = 7
SHT_NOBITS = 8
STT_OBJECT = 1
STT_FUNC = 2

# EF_AMDGPU_MACH_* -> target name, extracted from LLVM 23.1.1
# llvm/BinaryFormat/ELF.h (X-macro table).
EF_AMDGPU_MACH = {
    0x20: "gfx600", 0x21: "gfx601", 0x22: "gfx700", 0x23: "gfx701",
    0x24: "gfx702", 0x25: "gfx703", 0x26: "gfx704", 0x28: "gfx801",
    0x29: "gfx802", 0x2a: "gfx803", 0x2b: "gfx810", 0x2c: "gfx900",
    0x2d: "gfx902", 0x2e: "gfx904", 0x2f: "gfx906", 0x30: "gfx908",
    0x31: "gfx909", 0x32: "gfx90c", 0x33: "gfx1010", 0x34: "gfx1011",
    0x35: "gfx1012", 0x36: "gfx1030", 0x37: "gfx1031", 0x38: "gfx1032",
    0x39: "gfx1033", 0x3a: "gfx602", 0x3b: "gfx705", 0x3c: "gfx805",
    0x3d: "gfx1035", 0x3e: "gfx1034", 0x3f: "gfx90a", 0x41: "gfx1100",
    0x42: "gfx1013", 0x43: "gfx1150", 0x44: "gfx1103", 0x45: "gfx1036",
    0x46: "gfx1101", 0x47: "gfx1102", 0x48: "gfx1200", 0x49: "gfx1250",
    0x4A: "gfx1151", 0x4C: "gfx942", 0x4E: "gfx1201", 0x4F: "gfx950",
    0x50: "gfx1310", 0x55: "gfx1152", 0x57: "gfx1154", 0x58: "gfx1153",
    0x5A: "gfx1251", 0x5C: "gfx1172", 0x5D: "gfx1170", 0x5E: "gfx1171",
}
EF_AMDGPU_MACH_MASK = 0x00FF


class ElfError(ValueError):
    """Unrecoverable ELF parsing condition (reason string attached)."""


@dataclass
class Section:
    index: int
    name: str
    sh_type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    align: int
    entsize: int


@dataclass
class Symbol:
    name: str
    value: int
    size: int
    shndx: int
    bind: int
    sym_type: int
    other: int
    table_section: int  # index of section holding this symtab


@dataclass
class Note:
    section: str
    owner: str
    note_type: int
    desc: bytes
    desc_offset: int  # absolute file offset of the descriptor bytes


@dataclass
class Elf:
    data: bytes
    path_label: str
    osabi: int
    abiversion: int
    e_type: int
    e_machine: int
    e_flags: int
    sections: list
    symbols: list
    notes: list
    errors: list = field(default_factory=list)

    # ---- helpers -------------------------------------------------------
    @property
    def mach(self) -> int:
        return self.e_flags & EF_AMDGPU_MACH_MASK

    @property
    def mach_name(self) -> str | None:
        return EF_AMDGPU_MACH.get(self.mach)

    @property
    def is_amdgpu(self) -> bool:
        return self.e_machine == EM_AMDGPU

    def section(self, name: str) -> Section | None:
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def section_data(self, s: Section) -> bytes:
        if s.sh_type == SHT_NOBITS:
            return b""
        end = s.offset + s.size
        if end > len(self.data):
            raise ElfError(
                f"section {s.name!r} data out of bounds "
                f"(offset {s.offset}+size {s.size} > {len(self.data)})"
            )
        return self.data[s.offset:end]

    def symbol_map(self) -> dict:
        """First symbol per name (last-defined wins, like a linker view)."""
        out: dict[str, Symbol] = {}
        for sym in self.symbols:
            if sym.name:
                out[sym.name] = sym
        return out

    def symbol_code(self, sym: Symbol) -> bytes | None:
        """Raw instruction bytes for a symbol, or None (NOBITS/absent)."""
        if sym.shndx <= 0 or sym.shndx >= len(self.sections):
            return None
        sec = self.sections[sym.shndx]
        if sec.sh_type == SHT_NOBITS:
            return None
        start = sec.offset + sym.value
        # Relocatable objects: sym.value is section-relative.  Linked
        # objects: also section-relative for section-backed symbols.
        if sec.addr and not self._relatable():
            # values are VMAs; convert via section address
            start = sec.offset + (sym.value - sec.addr)
        end = start + (sym.size or 0)
        if start < 0 or end > len(self.data) or end < start:
            return None
        if sym.size == 0:
            return b""
        return self.data[start:end]

    def symbol_file_offset(self, sym: Symbol) -> int | None:
        if sym.shndx <= 0 or sym.shndx >= len(self.sections):
            return None
        sec = self.sections[sym.shndx]
        if sec.sh_type == SHT_NOBITS:
            return None
        rel = sym.value if self._relatable() else (sym.value - sec.addr)
        off = sec.offset + rel
        return off if 0 <= off < len(self.data) else None

    def _relatable(self) -> bool:
        return self.e_type == 1  # ET_REL: values are section-relative


def parse(data: bytes, path_label: str = "<memory>") -> Elf:
    if len(data) < 64:
        raise ElfError(f"{len(data)} bytes too short for an ELF64 header")
    if data[:4] != b"\x7fELF":
        raise ElfError("missing ELF magic \\x7fELF")
    if data[4] != 2:
        raise ElfError(f"unsupported ELF class {data[4]} (only ELF64/class 2)")
    if data[5] != 1:
        raise ElfError(f"unsupported ELF endianness {data[5]} (only LE/1)")
    (e_type, e_machine, _ver, _entry, _phoff, e_shoff) = struct.unpack_from(
        "<HHIQQQ", data, 16
    )
    e_flags = struct.unpack_from("<I", data, 48)[0]
    (_ehsize, _phentsize, _phnum, e_shentsize, e_shnum, e_shstrndx) = (
        struct.unpack_from("<HHHHHH", data, 52)
    )
    if e_shentsize != 64:
        raise ElfError(f"unexpected e_shentsize {e_shentsize} (want 64)")
    if e_shoff == 0 or e_shnum == 0:
        raise ElfError("no section header table (stripped/flat images unsupported)")
    if e_shoff + e_shnum * 64 > len(data):
        raise ElfError("section header table out of bounds")
    if e_shstrndx >= e_shnum:
        raise ElfError(f"bad e_shstrndx {e_shstrndx}")

    raw = [
        struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * 64)
        for i in range(e_shnum)
    ]
    strtab_off = raw[e_shstrndx][4]
    strtab_size = raw[e_shstrndx][5]
    if strtab_off + strtab_size > len(data):
        raise ElfError("shstrtab out of bounds")
    strtab = data[strtab_off : strtab_off + strtab_size]

    def name_at(o: int) -> str:
        if o >= len(strtab):
            return ""
        end = strtab.find(b"\x00", o)
        return strtab[o:end].decode("utf-8", "replace") if end >= 0 else ""

    sections = []
    for i, (noff, sht, flags, addr, off, size, link, info, align, entsz) in enumerate(raw):
        sections.append(
            Section(i, name_at(noff), sht, flags, addr, off, size, link, info, align, entsz)
        )

    errors: list[str] = []
    symbols: list[Symbol] = []
    notes: list[Note] = []

    # symbols
    for sec in sections:
        if sec.sh_type != SHT_SYMTAB:
            continue
        if sec.entsize not in (0, 24):
            errors.append(f"symbol table {sec.name}: entsize {sec.entsize} unsupported")
            continue
        if sec.link >= len(sections):
            errors.append(f"symbol table {sec.name}: bad string table link")
            continue
        strd = sections[sec.link]
        try:
            stab = _bounded(data, strd.offset, strd.size, f"strtab {strd.name}")
            body = _bounded(data, sec.offset, sec.size, f"symtab {sec.name}")
        except ElfError as e:
            errors.append(str(e))
            continue
        for i in range(len(body) // 24):
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
                "<IBBHQQ", body, i * 24
            )
            end = stab.find(b"\x00", st_name)
            nm = stab[st_name:end].decode("utf-8", "replace") if 0 <= st_name < len(stab) and end >= 0 else ""
            symbols.append(
                Symbol(
                    nm, st_value, st_size, st_shndx,
                    st_info >> 4, st_info & 0xF, st_other, sec.index,
                )
            )

    # notes
    for sec in sections:
        if sec.sh_type != SHT_NOTE:
            continue
        try:
            blob = _bounded(data, sec.offset, sec.size, f"note section {sec.name}")
        except ElfError as e:
            errors.append(str(e))
            continue
        p = 0
        while p + 12 <= len(blob):
            namesz, descsz, ntype = struct.unpack_from("<III", blob, p)
            if namesz > len(blob) or descsz > len(blob):
                errors.append(f"note section {sec.name}: note sizes out of range at +{p}")
                break
            owner_raw = blob[p + 12 : p + 12 + namesz]
            owner = owner_raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
            desc_rel = p + 12 + ((namesz + 3) // 4) * 4
            if desc_rel + descsz > len(blob):
                errors.append(f"note section {sec.name}: descriptor out of range at +{p}")
                break
            notes.append(
                Note(
                    sec.name,
                    owner,
                    ntype,
                    blob[desc_rel : desc_rel + descsz],
                    sec.offset + desc_rel,
                )
            )
            p = desc_rel + ((descsz + 3) // 4) * 4
            if descsz == 0 and namesz == 0:
                break

    return Elf(
        data=data,
        path_label=path_label,
        osabi=data[7],
        abiversion=data[8],
        e_type=e_type,
        e_machine=e_machine,
        e_flags=e_flags,
        sections=sections,
        symbols=symbols,
        notes=notes,
        errors=errors,
    )


def _bounded(data: bytes, off: int, size: int, what: str) -> bytes:
    if off + size > len(data):
        raise ElfError(f"{what} out of bounds (offset {off}+size {size})")
    return data[off : off + size]


def looks_like_amdgpu_elf(data: bytes, offset: int = 0) -> bool:
    """Cheap structural check: ELF64 LE with e_machine == EM_AMDGPU."""
    if data[offset : offset + 4] != b"\x7fELF":
        return False
    if len(data) < offset + 20 or data[offset + 4] != 2 or data[offset + 5] != 1:
        return False
    return struct.unpack_from("<H", data, offset + 18)[0] == EM_AMDGPU
