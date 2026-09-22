"""Tiny ELF64 writer used ONLY to build synthetic fixtures.

Deliberately minimal: ET_REL, ELF64 LE, sections + symtab.  Used for
fixtures the real toolchain cannot emit (code object v3 classic metadata)
and for host ELFs that embed bundle containers.  Never used in production
parsing paths.
"""

from __future__ import annotations

import struct

EM_AMDGPU = 224


def _strtab(strings: list[str]) -> tuple[bytes, dict]:
    data = bytearray(b"\x00")
    index = {"": 0}
    for s in strings:
        if s in index:
            continue
        index[s] = len(data)
        data += s.encode("utf-8") + b"\x00"
    return bytes(data), index


def build_relocatable(
    sections: list[dict],
    symbols: list[dict],
    *,
    e_flags: int = 0,
    osabi: int = 64,
    abiversion: int = 0,
    machine: int = EM_AMDGPU,
) -> bytes:
    """sections: [{name, type, flags, data, align, info, link, entsize}]
    symbols:  [{name, value, size, shndx, bind, type, other}]  (sym0 = null)
    """
    shstr_names = [s["name"] for s in sections] + [".shstrtab", ".strtab", ".symtab"]
    # symtab implies .strtab
    needs_strtab = any(s["name"] != ".shstrtab" for s in symbols)
    if needs_strtab and ".strtab" not in shstr_names:
        shstr_names.append(".strtab")

    shstr_data, shstr_index = _strtab(shstr_names)
    if needs_strtab:
        strtab_data, str_index = _strtab([s["name"] for s in symbols])
    else:
        strtab_data, str_index = b"\x00", {"": 0}

    # layout: header | section data... | shstrtab | strtab | symtab? | shdrs
    off = 64
    laid = []
    for s in sections:
        align = s.get("align", 1) or 1
        off = (off + align - 1) // align * align
        data = s.get("data", b"")
        laid.append((off, data))
        off += len(data)

    shstr_off = off
    off += len(shstr_data)
    strtab_off = off
    if needs_strtab:
        off += len(strtab_data)

    # symbol table needs its own index among sections
    n_sections = len(sections) + 1 + (1 if needs_strtab else 0) + 1  # +shstr +symtab
    # order: [user sections..., .shstrtab, .strtab?, .symtab?, ...shdrs]
    symtab_index = None
    if needs_strtab:
        # symbol index 0 must be the null entry; keep a caller-supplied one,
        # otherwise prepend it (callers must not lose their first symbol)
        if not (
            symbols
            and symbols[0].get("name", "") == ""
            and symbols[0].get("value", 0) == 0
            and symbols[0].get("shndx", 0) == 0
        ):
            symbols = [
                {"name": "", "value": 0, "size": 0, "shndx": 0,
                 "bind": 0, "type": 0, "other": 0}
            ] + list(symbols)
        symtab_index = len(sections) + 2
        align = 8
        off = (off + align - 1) // align * align
        symtab_off = off
        sym_blob = bytearray()
        first_nonlocal = 1
        for i, sym in enumerate(symbols):
            st_info = (sym.get("bind", 1) << 4) | sym.get("type", 2)
            sym_blob += struct.pack(
                "<IBBHQQ",
                str_index.get(sym["name"], 0),
                st_info,
                sym.get("other", 0),
                sym.get("shndx", 0),
                sym.get("value", 0),
                sym.get("size", 0),
            )
            if sym.get("bind", 1) != 0 and first_nonlocal == i:
                first_nonlocal = i
        off += len(sym_blob)
    else:
        symtab_off = 0
        sym_blob = b""
        first_nonlocal = 1

    shoff = (off + 7) // 8 * 8

    # section header table
    def shdr(name, shtype, flags, addr, soff, ssize, link, info, align, entsize):
        return struct.pack(
            "<IIQQQQIIQQ",
            shstr_index.get(name, 0), shtype, flags, addr, soff, ssize,
            link, info, align, entsize,
        )

    hdrs = bytearray()
    hdrs += shdr("", 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for i, s in enumerate(sections):
        soff, data = laid[i]
        hdrs += shdr(
            s["name"], s["type"], s.get("flags", 0), s.get("addr", 0),
            soff, len(data), s.get("link", 0), s.get("info", 0),
            s.get("align", 1), s.get("entsize", 0),
        )
    hdrs += shdr(".shstrtab", 3, 0, 0, shstr_off, len(shstr_data), 0, 0, 1, 0)
    strndx_sh = None
    if needs_strtab:
        strndx_sh = len(sections) + 2
        hdrs += shdr(".strtab", 3, 0, 0, strtab_off, len(strtab_data), 0, 0, 1, 0)
        symtab_link = strndx_sh
        hdrs += shdr(
            ".symtab", 2, 0, 0, symtab_off, len(sym_blob),
            symtab_link, first_nonlocal, 8, 24,
        )

    shstrtab_idx = len(sections) + 1
    # null hdr + user sections + .shstrtab [+ .strtab + .symtab]
    e_shnum = len(sections) + 1 + (2 if needs_strtab else 0) + 1 - 1 + 1 - 1
    e_shnum = 1 + len(sections) + 1 + (2 if needs_strtab else 0)

    eh = bytearray(64)
    eh[0:4] = b"\x7fELF"
    eh[4] = 2  # ELFCLASS64
    eh[5] = 1  # ELFDATA2LSB
    eh[6] = 1  # EV_CURRENT
    eh[7] = osabi
    eh[8] = abiversion
    struct.pack_into(
        "<HHIQQQIHHHHHH", eh, 16,
        1, machine, 1, 0, 0, shoff, e_flags,
        64, 0, 0, 64, e_shnum, shstrtab_idx,
    )

    out = bytearray(eh)
    # section data in order
    for i, s in enumerate(sections):
        soff, data = laid[i]
        out += b"\x00" * (soff - len(out))
        out += data
    out += b"\x00" * (shstr_off - len(out))
    out += shstr_data
    if needs_strtab:
        out += b"\x00" * (strtab_off - len(out))
        out += strtab_data
        out += b"\x00" * (symtab_off - len(out))
        out += sym_blob
    out += b"\x00" * (shoff - len(out))
    out += hdrs
    return bytes(out)
