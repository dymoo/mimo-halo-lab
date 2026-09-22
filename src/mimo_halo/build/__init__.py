"""Candidate construction for the compression sweep.

Turns the ORIGINAL pinned Xiaomi MiMo-V2.6-Flash weights + an external
prune selection (prune map) + a quant recipe (the sweep allocation
vocabulary: bits / group_size / mode incl. mxfp4) into a candidate
checkpoint: streamed, bounded-memory sharded safetensors with exact byte
accounting, per-tensor quant assignment, build report, the schema-shaped
candidate record, and full ``mimo_halo.artifacts`` provenance sidecars.

Refusals fail closed (:mod:`mimo_halo.build.errors`): input paths outside
the verified source root, unrecorded second-generation quantisation,
second-gen quant on a recipe class marked passthrough, byte accounting
that does not close exactly against the recipe, and missing packing-
evidence siblings.
"""

from .candidate import (
    CANDIDATE_RECORD_NAME,
    RECIPE_KEYS,
    SCHEMA_CLASSES,
    build_candidate,
    candidate_record,
    read_safetensors_header,
)
from .errors import (
    AccountingError,
    BuildError,
    PackingEvidenceError,
    PassthroughQuantError,
    RequantError,
    SourcePathError,
)
from .quantize import (
    FORMATS,
    AffineTensor,
    affine_expected_bytes,
    decode_bf16,
    decode_mxfp4,
    dequant_affine,
    encode_affine,
    unpack_codes,
)

__all__ = [
    "AccountingError",
    "AffineTensor",
    "BuildError",
    "CANDIDATE_RECORD_NAME",
    "FORMATS",
    "PackingEvidenceError",
    "PassthroughQuantError",
    "RECIPE_KEYS",
    "RequantError",
    "SCHEMA_CLASSES",
    "SourcePathError",
    "affine_expected_bytes",
    "build_candidate",
    "candidate_record",
    "decode_bf16",
    "decode_mxfp4",
    "dequant_affine",
    "encode_affine",
    "read_safetensors_header",
    "unpack_codes",
]
