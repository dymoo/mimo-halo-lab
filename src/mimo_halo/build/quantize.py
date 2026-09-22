"""Second-generation quantization kernels for candidate construction.

Vocabulary (configs/experiments/compression-sweep.json ``quant_formats``)
maps the sweep's format enum to the bits / group_size / mode vocabulary of
the project (mode in {affine, mxfp4}); native formats carry no quant
parameters and are dispatched as literal copies or an exact dense row slice.

Affine byte rule (pinned by the sweep config, exact integers only)::

    bytes = logical_parameters * bits // 8
            + (logical_parameters // group_size) * 2

The two bytes per group are ONE int8 scale slot and ONE int8 zero slot.
Numeric convention (documented here, in the quant assignment sidecar, and
in the build report): the scale slot stores an int8 power-of-two EXPONENT
``e`` (group scale ``2**e`` - the same power-of-two scale convention as
Xiaomi's E8M0 MXFP4 scale bytes) and the zero slot stores the int8
zero-point code ``z``. Dequantization is ``value = (code - z) * 2**e`` with
``code`` in ``[0, 2**bits - 1]``. Both slots are literally int8, the pinned
byte rule holds exactly, and the encoding stays affine in the code.

Source formats:

* MXFP4 packed weights: ``dtype U8``, two E2M1 nibbles per stored byte (low
  nibble = even column), sibling ``weight_scale`` ``dtype U8`` with one E8M0
  exponent byte per block of 32 logical columns (value ``2**(byte - 127)``,
  byte 255 = NaN -> fail closed). Evidence: models/inventory.py
  ``quantization_record``/``logical_view`` and the exact 4.25-bpw figure.
* BF16 stored payloads: decoded to float32; the decoded value equals the
  BF16-represented value bit-for-bit.

Decoded intermediates are COMPUTE-ONLY: they never round-trip back to a
stored source format and never claim to restore pre-QAT/full-precision
weights. Re-encoding from them is second-generation quantization and is
gated by the caller (candidate.py) on the recipe's recorded ledger.

numpy is imported lazily: a bit-exact passthrough candidate builds with the
standard library alone; the second-gen kernels fail closed with a named
error when numpy is absent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import AccountingError, BuildError

# Format enum -> vocabulary, transcribed from
# configs/experiments/compression-sweep.json quant_formats.formats.
FORMATS: dict[str, dict] = {
    "mxfp4_native": {
        "bits": 4,
        "group_size": 32,
        "mode": "mxfp4",
        "second_gen": False,
        "source_dtypes": ("U8",),
    },
    "q3_affine_g128": {
        "bits": 3,
        "group_size": 128,
        "mode": "affine",
        "second_gen": True,
        "source_dtypes": ("U8", "BF16"),
    },
    "q8_affine_g64": {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
        "second_gen": True,
        "source_dtypes": ("BF16",),
    },
    "q6_affine_g128": {
        "bits": 6,
        "group_size": 128,
        "mode": "affine",
        "second_gen": True,
        "source_dtypes": ("BF16",),
    },
    "native_fp8_e4m3": {
        "mode": "passthrough",
        "second_gen": False,
        "source_dtypes": ("F8_E4M3",),
    },
    "native_f32": {
        "mode": "passthrough",
        "second_gen": False,
        "source_dtypes": ("F32",),
    },
    "native_bf16": {
        "mode": "passthrough",
        "second_gen": False,
        "source_dtypes": ("BF16",),
    },
    "native_dense_row_sliced": {
        "mode": "row_slice",
        "second_gen": False,
        "source_dtypes": ("BF16", "F32"),
    },
}

#: Affine bit widths this encoder can pack (the sweep vocabulary: 3/6/8).
AFFINE_BITS = (3, 6, 8)

#: Chunk size for streamed byte copies (bounded memory).
COPY_CHUNK_BYTES = 1 << 20
#: Upper bound on source bytes held at once while affine-encoding.
AFFINE_CHUNK_BYTES = 1 << 23

#: E2M1 (MXFP4 element) positive values, indexed by the nibble's low three
#: bits: (exp, mant) -> value. Sign is nibble bit 3.
_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class PackingShapeError(BuildError):
    """Stored byte counts contradict the declared layout (fail closed)."""


def _numpy():
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - only without numpy
        raise BuildError(
            "numpy is required for second-generation quantization kernels "
            "(bit-exact passthrough construction is stdlib-only; install numpy "
            "or choose an all-passthrough recipe)"
        ) from exc
    return numpy


def require_numpy():
    """Fail closed up-front when a build plan needs the second-gen kernels."""
    return _numpy()


def affine_expected_bytes(logical_parameters: int, bits: int, group_size: int) -> int:
    """Exact stored bytes of one affine-encoded tensor under the pinned rule.

    Fails closed when the byte rule does not land on exact integers: the
    bit stream must pack to whole bytes (``params * bits`` divisible by 8)
    and every group must be full (``params`` divisible by ``group_size``).
    """
    if not isinstance(logical_parameters, int) or isinstance(logical_parameters, bool):
        raise AccountingError(f"logical_parameters must be an int: {logical_parameters!r}")
    if (
        not isinstance(bits, int)
        or isinstance(bits, bool)
        or not isinstance(group_size, int)
        or isinstance(group_size, bool)
    ):
        raise AccountingError("bits and group_size must be ints")
    if logical_parameters <= 0:
        raise AccountingError(f"logical_parameters must be positive: {logical_parameters}")
    if group_size <= 0:
        raise AccountingError(f"group_size must be positive: {group_size}")
    if bits <= 0:
        raise AccountingError(f"bits must be positive: {bits}")
    if (logical_parameters * bits) % 8 != 0:
        raise AccountingError(
            f"byte accounting does not close: {logical_parameters} params x {bits} bits "
            "is not a whole number of bytes"
        )
    if logical_parameters % group_size != 0:
        raise AccountingError(
            f"byte accounting does not close: {logical_parameters} params not divisible "
            f"by group_size {group_size}"
        )
    return (logical_parameters * bits) // 8 + (logical_parameters // group_size) * 2


def decode_mxfp4(weight: bytes, scale: bytes, rows: int, logical_cols: int):
    """Decode packed MXFP4 (weight + E8M0 scale sibling) to float32 values.

    Returns a ``(rows, logical_cols)`` float32 array of the represented
    values (compute-only). Structural mismatches and NaN scale bytes fail
    closed.
    """
    np = _numpy()
    if logical_cols % 32 != 0:
        raise PackingShapeError(
            f"mxfp4 decode requires a logical width divisible by 32, got {logical_cols}"
        )
    stored_cols = logical_cols // 2
    scale_cols = logical_cols // 32
    packed = np.frombuffer(weight, dtype=np.uint8)
    if packed.size != rows * stored_cols:
        raise PackingShapeError(
            f"mxfp4 weight byte count {packed.size} != {rows} x {stored_cols}"
        )
    packed = packed.reshape(rows, stored_cols)
    scale = np.frombuffer(scale, dtype=np.uint8)
    if scale.size != rows * scale_cols:
        raise PackingShapeError(
            f"mxfp4 scale byte count {scale.size} != {rows} x {scale_cols}"
        )
    if bool((scale == 255).any()):
        raise BuildError("mxfp4 scale sibling contains an E8M0 NaN byte (255); fail closed")
    scale = scale.reshape(rows, scale_cols)

    lut = np.empty(16, dtype=np.float32)
    for nibble in range(16):
        magnitude = _E2M1_POSITIVE[nibble & 0x07]
        lut[nibble] = -magnitude if nibble & 0x08 else magnitude
    low = packed & np.uint8(0x0F)
    high = packed >> np.uint8(4)
    values = np.empty((rows, logical_cols), dtype=np.float32)
    values[:, 0::2] = lut[low]
    values[:, 1::2] = lut[high]
    block_scale = np.exp2(scale.astype(np.float32) - np.float32(127.0))
    values *= np.repeat(block_scale, 32, axis=1)
    return values


def decode_bf16(payload: bytes, rows: int, cols: int):
    """Decode a stored BF16 payload to float32 (exact; compute-only view)."""
    np = _numpy()
    raw = np.frombuffer(payload, dtype="<u2")
    if raw.size != rows * cols:
        raise PackingShapeError(
            f"bf16 payload has {raw.size} values, expected {rows} x {cols}"
        )
    widened = raw.astype(np.uint32).reshape(rows, cols) << np.uint32(16)
    return widened.view(np.float32).reshape(rows, cols)


@dataclass
class AffineTensor:
    """One affine-encoded tensor: packed codes + interleaved scale/zero."""

    codes: bytes
    scale_zero: bytes  # int8 pairs, row-major: [rows, groups, 2] -> (e, z)
    rows: int
    cols: int
    bits: int
    group_size: int
    stats: dict

    @property
    def nbytes(self) -> int:
        return len(self.codes) + len(self.scale_zero)


def encode_affine(values, bits: int, group_size: int) -> AffineTensor:
    """Group-wise affine quantization of a ``(rows, cols)`` float array.

    Layout (row-major, deterministic):

    * codes: ``bits``-bit codes packed LSB-first; each row is independently
      byte-aligned (``cols * bits % 8 == 0``).
    * scale_zero: one ``int8`` pair ``(e, z)`` per group of ``group_size``
      consecutive row-major values (``cols % group_size == 0``).

    Per group: ``e = ceil(log2(range / (2**bits - 1)))`` clamped to int8,
    ``z = round(-min / 2**e)`` with ``e`` raised until ``z`` fits int8,
    ``code = clip(round(value / 2**e) + z, 0, 2**bits - 1)``. Error
    statistics compare the dequantized ``(code - z) * 2**e`` against the
    input values (the BF16-represented source values, never a pre-QAT
    claim). Constant groups (range 0) reduce naturally: all-equal values
    encode with ``code == z`` and dequantize exactly when the value is 0,
    within one step otherwise.
    """
    np = _numpy()
    if bits not in AFFINE_BITS:
        raise BuildError(f"affine encoder supports bits {AFFINE_BITS}, got {bits}")
    if values.ndim != 2:
        raise BuildError(f"affine encoder expects a 2-D array, got ndim={values.ndim}")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if cols <= 0 or rows <= 0:
        raise AccountingError(f"empty tensor shape ({rows}, {cols})")
    if cols % group_size != 0:
        raise AccountingError(f"group_size {group_size} does not divide row width {cols}")
    if (cols * bits) % 8 != 0:
        raise AccountingError(f"row of {cols} values x {bits} bits is not byte-aligned")

    v = np.asarray(values, dtype=np.float64).reshape(rows, cols)
    groups_per_row = cols // group_size
    blocks = v.reshape(rows, groups_per_row, group_size)
    vmin = blocks.min(axis=2)
    rng = blocks.max(axis=2) - vmin
    levels = float((1 << bits) - 1)

    # range == 0 groups get a stand-in range so the log path stays valid;
    # the e/z math below still yields code == z and an exact/near-exact
    # reconstruction for constants.
    safe_rng = np.where(rng == 0.0, np.maximum(np.abs(vmin) * 2.0, 1e-300), rng)
    target = safe_rng / levels
    e = np.ceil(np.log2(target)).astype(np.int64)
    # IEEE log2 can land a hair under an exact power of two; nudge until
    # 2**e >= target so the step never under-covers the group range.
    while bool((np.exp2(np.minimum(e, 1023).astype(np.float64)) < target).any()):
        e = np.where(np.exp2(np.minimum(e, 1023).astype(np.float64)) < target, e + 1, e)
    e = np.clip(e, -128, 127)

    def _zero_point(exponent):
        return np.rint(-vmin / np.exp2(exponent.astype(np.float64)))

    z = _zero_point(e)
    bumps = 0
    while bool((np.abs(z) > 127).any()):
        at_limit = e >= 127
        if bool(at_limit.all()):  # unreachable for float32-sourced data
            raise BuildError("zero point overflows int8 even at exponent 127")
        e = np.where(at_limit, e, e + 1)
        z = _zero_point(e)
        bumps += 1
        if bumps > 512:  # pragma: no cover - defensive guard
            raise BuildError("zero-point exponent search did not converge")

    step = np.exp2(e.astype(np.float64)).reshape(rows, groups_per_row, 1)
    z_b = z.reshape(rows, groups_per_row, 1)
    raw = np.rint(blocks / step) + z_b
    codes = np.clip(raw, 0, levels).astype(np.uint8).reshape(rows, cols)

    # Dequantized error statistics against the compute-only reference.
    deq = ((codes.astype(np.float64).reshape(rows, groups_per_row, group_size) - z_b)
           * step).reshape(rows, cols)
    err = np.abs(deq - v)
    stats = {
        "elements": int(rows * cols),
        "max_abs_err": float(err.max()) if err.size else 0.0,
        "mean_abs_err": float(err.mean()) if err.size else 0.0,
        "max_group_scale_exp": int(e.max()) if e.size else 0,
        "min_group_scale_exp": int(e.min()) if e.size else 0,
        "exponent_bumps": bumps,
    }

    return AffineTensor(
        codes=_pack_codes(codes, bits),
        scale_zero=np.ascontiguousarray(
            np.stack([e.astype(np.int8), z.astype(np.int8)], axis=-1)
        ).tobytes(),
        rows=rows,
        cols=cols,
        bits=bits,
        group_size=group_size,
        stats=stats,
    )


def _pack_codes(codes, bits: int) -> bytes:
    """Pack ``bits``-bit codes LSB-first, each row byte-aligned."""
    np = _numpy()
    rows, cols = int(codes.shape[0]), int(codes.shape[1])
    if bits == 8:
        return np.ascontiguousarray(codes, dtype=np.uint8).tobytes()
    if bits in (3, 6):
        per = 8 if bits == 3 else 4  # 24 bits = 3 bytes per group of codes
        grouped = codes.reshape(rows, cols // per, per).astype(np.uint32)
        if bits == 6:
            packed = (
                grouped[..., 0]
                | (grouped[..., 1] << np.uint32(6))
                | (grouped[..., 2] << np.uint32(12))
                | (grouped[..., 3] << np.uint32(18))
            )
        else:
            packed = np.zeros(grouped.shape[:2], dtype=np.uint32)
            for i in range(per):
                packed |= grouped[..., i] << np.uint32(3 * i)
        out = np.stack(
            [
                packed & np.uint32(0xFF),
                (packed >> np.uint32(8)) & np.uint32(0xFF),
                (packed >> np.uint32(16)) & np.uint32(0xFF),
            ],
            axis=-1,
        ).astype(np.uint8)
        return np.ascontiguousarray(out).tobytes()
    raise BuildError(f"affine encoder supports bits {AFFINE_BITS}, got {bits}")


def unpack_codes(payload: bytes, rows: int, cols: int, bits: int):
    """Inverse of :func:`_pack_codes`; returns a ``(rows, cols)`` uint8 array."""
    np = _numpy()
    if (cols * bits) % 8 != 0:
        raise AccountingError(f"row of {cols} values x {bits} bits is not byte-aligned")
    expected = rows * cols * bits // 8
    if len(payload) != expected:
        raise PackingShapeError(f"packed code byte count {len(payload)} != {expected}")
    raw = np.frombuffer(payload, dtype=np.uint8)
    if bits == 8:
        return raw.reshape(rows, cols).copy()
    per_codes = 8 if bits == 3 else 4
    triples = raw.reshape(rows, cols // per_codes, 3).astype(np.uint32)
    packed = (
        triples[..., 0]
        | (triples[..., 1] << np.uint32(8))
        | (triples[..., 2] << np.uint32(16))
    )
    out = np.empty((rows, cols // per_codes, per_codes), dtype=np.uint8)
    mask = np.uint32((1 << bits) - 1)
    for i in range(per_codes):
        out[..., i] = ((packed >> np.uint32(bits * i)) & mask).astype(np.uint8)
    return out.reshape(rows, cols)


def dequant_affine(
    payload: bytes, scale_zero: bytes, rows: int, cols: int, bits: int, group_size: int
):
    """Dequantize an affine-encoded tensor back to float32 (for verification)."""
    np = _numpy()
    if cols % group_size != 0:
        raise AccountingError(f"group_size {group_size} does not divide row width {cols}")
    groups = rows * (cols // group_size)
    if len(scale_zero) != groups * 2:
        raise PackingShapeError(f"scale/zero byte count {len(scale_zero)} != {groups} x 2")
    pair = np.frombuffer(scale_zero, dtype=np.int8).reshape(rows, cols // group_size, 2)
    e = pair[..., 0].astype(np.float64)
    z = pair[..., 1].astype(np.float64)
    codes = (
        unpack_codes(payload, rows, cols, bits)
        .astype(np.float64)
        .reshape(rows, cols // group_size, group_size)
    )
    deq = (codes - z[:, :, np.newaxis]) * np.exp2(e)[:, :, np.newaxis]
    return deq.reshape(rows, cols).astype(np.float32)
