"""Minimal self-contained MessagePack codec (no third-party deps).

Covers exactly what AMDGPU metadata notes and offload fixtures contain:
maps, arrays, strs, bins, ints, floats, bool, nil, plus a permissive ext
placeholder.  decode() raises MsgpackError with a precise reason on
malformed input so callers can surface honest null+reason strings.
"""

from __future__ import annotations

import struct


class MsgpackError(ValueError):
    """Malformed or unsupported MessagePack payload."""


# ext placeholder: (type_code, payload_bytes)
Ext = tuple


def encode(obj) -> bytes:
    """Encode a limited Python subset to MessagePack."""
    out = bytearray()

    def put(n: int) -> None:
        out.extend(n.to_bytes(1, "big"))

    def pack(o) -> None:
        if o is None:
            put(0xC0)
        elif o is False:
            put(0xC2)
        elif o is True:
            put(0xC3)
        elif isinstance(o, int):
            if 0 <= o <= 127:
                put(o)
            elif -32 <= o < 0:
                put(0x100 + o)
            elif 0 <= o <= 0xFFFFFFFF:
                out.append(0xCE)
                out.extend(o.to_bytes(4, "big"))
            elif -0x80000000 <= o < 0:
                out.append(0xD2)
                out.extend((o + (1 << 32)).to_bytes(4, "big"))
            elif 0 <= o <= 0xFFFFFFFFFFFFFFFF:
                out.append(0xCF)
                out.extend(o.to_bytes(8, "big"))
            else:
                out.append(0xD3)
                out.extend((o + (1 << 64)).to_bytes(8, "big"))
        elif isinstance(o, float):
            out.append(0xCB)
            out.extend(struct.pack(">d", o))
        elif isinstance(o, str):
            b = o.encode("utf-8")
            n = len(b)
            if n <= 31:
                put(0xA0 | n)
            elif n <= 0xFF:
                put(0xD9)
                put(n)
            elif n <= 0xFFFF:
                put(0xDA)
                out.extend(n.to_bytes(2, "big"))
            else:
                put(0xDB)
                out.extend(n.to_bytes(4, "big"))
            out.extend(b)
        elif isinstance(o, (bytes, bytearray)):
            n = len(o)
            if n <= 0xFF:
                put(0xC4)
                put(n)
            elif n <= 0xFFFF:
                put(0xC5)
                out.extend(n.to_bytes(2, "big"))
            else:
                put(0xC6)
                out.extend(n.to_bytes(4, "big"))
            out.extend(o)
        elif isinstance(o, (list, tuple)):
            n = len(o)
            if n <= 15:
                put(0x90 | n)
            elif n <= 0xFFFF:
                put(0xDC)
                out.extend(n.to_bytes(2, "big"))
            else:
                put(0xDD)
                out.extend(n.to_bytes(4, "big"))
            for x in o:
                pack(x)
        elif isinstance(o, dict):
            n = len(o)
            if n <= 15:
                put(0x80 | n)
            elif n <= 0xFFFF:
                put(0xDE)
                out.extend(n.to_bytes(2, "big"))
            else:
                put(0xDF)
                out.extend(n.to_bytes(4, "big"))
            for k, v in o.items():
                pack(k)
                pack(v)
        else:
            raise MsgpackError(f"cannot encode type {type(o).__name__}")

    pack(obj)
    return bytes(out)


def decode(data: bytes):
    """Decode one MessagePack object; trailing NUL padding tolerated."""
    pos = 0
    value, pos = _decode_at(data, pos)
    trailing = data[pos:]
    if trailing.strip(b"\x00"):
        raise MsgpackError(
            f"{len(trailing)} unexpected trailing bytes after msgpack object"
        )
    return value


def _decode_at(data: bytes, pos: int):
    if pos >= len(data):
        raise MsgpackError(f"truncated msgpack at offset {pos}")
    b = data[pos]
    pos += 1

    # positive fixint
    if b <= 0x7F:
        return b, pos
    # fixmap
    if 0x80 <= b <= 0x8F:
        return _map(data, pos, b & 0x0F)
    # fixarray
    if 0x90 <= b <= 0x9F:
        return _array(data, pos, b & 0x0F)
    # fixstr
    if 0xA0 <= b <= 0xBF:
        return _str(data, pos, b & 0x1F)
    # negative fixint
    if b >= 0xE0:
        return b - 0x100, pos

    if b == 0xC0:
        return None, pos
    if b == 0xC2:
        return False, pos
    if b == 0xC3:
        return True, pos
    if b in (0xC4, 0xC5, 0xC6):  # bin8/16/32
        width = 1 << (b - 0xC4)
        n = int.from_bytes(_take(data, pos, width), "big")
        pos += width
        return bytes(_take(data, pos, n)), pos + n
    if b in (0xC7, 0xC8, 0xC9):  # ext8/16/32
        width = 1 << (b - 0xC7)
        n = int.from_bytes(_take(data, pos, width), "big")
        pos += width
        t = _take(data, pos, 1)[0]
        pos += 1
        return Ext((t, bytes(_take(data, pos, n)))), pos + n
    if b == 0xCA:
        return struct.unpack(">f", _take(data, pos, 4))[0], pos + 4
    if b == 0xCB:
        return struct.unpack(">d", _take(data, pos, 8))[0], pos + 8
    if b in (0xCC, 0xCD, 0xCE, 0xCF):  # uint8/16/32/64
        width = 1 << (b - 0xCC)
        v = int.from_bytes(_take(data, pos, width), "big")
        return v, pos + width
    if b in (0xD0, 0xD1, 0xD2, 0xD3):  # int8/16/32/64
        width = 1 << (b - 0xD0)
        v = int.from_bytes(_take(data, pos, width), "big", signed=True)
        return v, pos + width
    if b in (0xD4, 0xD5, 0xD6, 0xD7, 0xD8):  # fixext1..16
        width = 1 << (b - 0xD4)
        t = _take(data, pos, 1)[0]
        pos += 1
        return Ext((t, bytes(_take(data, pos, width)))), pos + width
    if b in (0xD9, 0xDA, 0xDB):  # str8/16/32
        width = 1 << (b - 0xD9)
        n = int.from_bytes(_take(data, pos, width), "big")
        pos += width
        return _str(data, pos, n)
    if b in (0xDC, 0xDD):  # array16/32
        width = 2 if b == 0xDC else 4
        n = int.from_bytes(_take(data, pos, width), "big")
        pos += width
        return _array(data, pos, n)
    if b in (0xDE, 0xDF):  # map16/32
        width = 2 if b == 0xDE else 4
        n = int.from_bytes(_take(data, pos, width), "big")
        pos += width
        return _map(data, pos, n)
    raise MsgpackError(f"unsupported msgpack marker 0x{b:02x} at offset {pos - 1}")


def _take(data: bytes, pos: int, n: int) -> bytes:
    if pos + n > len(data):
        raise MsgpackError(f"truncated msgpack: want {n} bytes at {pos}")
    return data[pos : pos + n]


def _str(data: bytes, pos: int, n: int):
    raw = _take(data, pos, n)
    try:
        return raw.decode("utf-8"), pos + n
    except UnicodeDecodeError as e:
        raise MsgpackError(f"invalid utf-8 str at {pos}: {e}") from None


def _array(data: bytes, pos: int, n: int):
    out = []
    for _ in range(n):
        v, pos = _decode_at(data, pos)
        out.append(v)
    return out, pos


def _map(data: bytes, pos: int, n: int):
    out = {}
    for _ in range(n):
        k, pos = _decode_at(data, pos)
        v, pos = _decode_at(data, pos)
        out[k] = v
    return out, pos
