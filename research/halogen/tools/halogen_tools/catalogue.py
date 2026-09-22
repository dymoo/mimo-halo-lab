"""Assemble per-kernel catalogue rows from inputs.

Row contract (schemas/kernel-catalogue.schema.json):
  name, hash, isa_size, vgpr, sgpr, lds, wave_size, likely_operation,
  confidence, evidence, null_reasons (+ richer optional context).

Invariant enforced here (and asserted by tests): every contract field
that is null MUST appear in null_reasons with a reason string.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os

from . import TOOL_NAME, TOOL_VERSION
from . import classify as classify_mod
from . import codeobj, disasm as disasm_mod
from . import validate as validate_mod

NULLABLE_CONTRACT_FIELDS = (
    "hash",
    "isa_size",
    "vgpr",
    "sgpr",
    "lds",
    "wave_size",
    "kernarg_segment_size",
    "private_segment_fixed_size",
    "max_flat_workgroup_size",
    "likely_operation",
    "confidence",
    "kd_preload",
)


def _input_label(path: str, used: set) -> str:
    """Privacy-safe catalogue label for an input path.

    Absolute paths never reach the output: paths under $MIMO_LAB become
    '$MIMO_LAB/...' placeholders, any other absolute path collapses to its
    basename (deduplicated with '#N' suffixes).  Relative paths pass through.
    """
    mimo = os.environ.get("MIMO_LAB")
    if mimo and os.path.isabs(path):
        root = mimo.rstrip("/\\")
        norm = os.path.normpath(path)
        if norm == root:
            base = "$MIMO_LAB"
        elif norm.startswith(root + os.sep):
            base = "$MIMO_LAB/" + os.path.relpath(norm, root).replace(os.sep, "/")
        else:
            base = os.path.basename(norm)
    elif os.path.isabs(path):
        base = os.path.basename(os.path.normpath(path))
    else:
        base = path
    label = base
    n = 2
    while label in used:
        label = f"{base}#{n}"
        n += 1
    used.add(label)
    return label


def build(paths, *, schema_path: str | None = None, mcpu: str | None = None,
          no_disasm: bool = False) -> tuple[dict, dict, dict]:
    """-> (catalogue, toolchain, errors) where errors =
    {"inputs": [...], "schema": [...]}.  Never raises on bad inputs."""
    errors: list[str] = []
    schema_errors: list[str] = []
    tool, tool_err = (None, "disassembly disabled via --no-disasm") if no_disasm \
        else disasm_mod.discover()
    toolchain = disasm_mod.toolchain_info(tool, tool_err)
    toolchain["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    toolchain["tool"] = {"name": TOOL_NAME, "version": TOOL_VERSION}
    commands: list[str] = []

    inputs_out = []
    rows = []
    used_labels: set = set()

    for path in paths:
        label = _input_label(path, used_labels)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as e:
            inputs_out.append({
                "path": label, "sha256": None, "size": None, "kind": "unreadable",
                "bundles": [], "code_objects": [], "errors": [f"read failed: {e}"],
            })
            errors.append(f"{label}: read failed: {e}")
            continue

        res = codeobj.analyze(label, data)
        co_summaries = []
        for idx, co in enumerate(res.code_objects):
            codeobj.parse_code_object(co, idx)
            if co.parse_error:
                co_summaries.append({
                    "origin": co.origin, "offset": co.offset,
                    "parse_error": co.parse_error, "kernel_count": 0,
                })
                errors.append(f"{label} [{co.origin}]: {co.parse_error}")
                continue
            # disassemble once per code object
            dis = None
            dis_err = None
            if tool:
                want_mcpu = mcpu or disasm_mod.mcpu_for(
                    {
                        "amdhsa_target": (co.notes_meta or [{}])[0].get("amdhsa_target"),
                        "mach_name": co.target.get("mach_name") if co.target else None,
                    }
                )
                dis = disasm_mod.disassemble(tool, co.data, want_mcpu, f"{label}#{co.origin}")
                if dis.error:
                    dis_err = dis.error
                    errors.append(f"{label} [{co.origin}]: {dis.error}")
                else:
                    commands.extend(dis.commands)
            else:
                dis_err = tool_err or "disassembly unavailable"

            co_summaries.append({
                "origin": co.origin,
                "offset": co.offset,
                "parse_error": None,
                "target": co.target,
                "metadata": co.notes_meta,
                "kernel_count": len(co.kernels),
            })
            for k in co.kernels:
                rows.append(_row(k, co, dis, dis_err, label, want_mcpu if tool else None))

        inputs_out.append({
            "path": label,
            "sha256": res.sha256,
            "size": res.size,
            "kind": res.kind,
            "bundles": res.bundles,
            "code_objects": co_summaries,
            "errors": res.errors,
        })
        errors.extend(f"{label}: {e}" for e in res.errors)

    catalogue = {
        "catalogue_version": 1,
        "generated_by": f"{TOOL_NAME} {TOOL_VERSION}",
        "generated_at": toolchain["generated_at"],
        "toolchain": toolchain,
        "disassembly_commands": sorted(set(commands)),
        "inputs": inputs_out,
        "kernels": rows,
    }

    if schema_path:
        try:
            with open(schema_path, "r", encoding="utf-8") as fh:
                schema = validate_mod.load_schema(json.load(fh))
        except (OSError, ValueError, validate_mod.SchemaError) as e:
            schema_errors.append(f"schema {schema_path}: {e}")
        else:
            verrs = validate_mod.validate(catalogue, schema)
            schema_errors.extend(f"schema violation: {v}" for v in verrs)

    toolchain_out = dict(toolchain)
    toolchain_out["disassembly_commands"] = sorted(set(commands))
    return catalogue, toolchain_out, {"inputs": errors, "schema": schema_errors}


def _row(k: codeobj.KernelRec, co: codeobj.CodeObject, dis, dis_err: str | None,
          label: str, mcpu_used: str | None) -> dict:
    null_reasons: dict[str, str] = {}
    evidence: list[dict] = []

    # ---- hash + ISA size --------------------------------------------
    if k.code:
        code_hash = "sha256:" + hashlib.sha256(k.code).hexdigest()
        isa_size = len(k.code)
    else:
        code_hash = None
        isa_size = None
        null_reasons["hash"] = k.code_reason or "no code bytes recovered"
        null_reasons["isa_size"] = k.code_reason or "no code bytes recovered"

    # ---- metadata / kd fields ----------------------------------------
    row = {
        "name": k.name,
        "input": label,
        "code_object_origin": co.origin,
        "hash": code_hash,
        "isa_size": isa_size,
    }
    field_sources = {}
    for f in ("vgpr", "sgpr", "lds", "wave_size", "kernarg_segment_size",
              "private_segment_fixed_size", "max_flat_workgroup_size"):
        fl = k.get(f)
        row[f] = fl.value
        field_sources[f] = fl.source
        if fl.value is None:
            null_reasons[f] = fl.reason or "unresolved without reason (bug guard)"

    # ---- kd preload ---------------------------------------------------
    if k.kd and "error" not in k.kd:
        row["kd_preload"] = k.kd["kernarg_preload"]
        evidence.append({
            "source": "kernel-descriptor",
            "offset": k.kd.get("file_offset"),
            "mnemonic": None,
            "detail": (
                f"symbol={k.kd.get('symbol')} group@0={k.kd['group_segment_fixed_size']} "
                f"priv@4={k.kd['private_segment_fixed_size']} "
                f"kernarg@8={k.kd['kernarg_size']} "
                f"rsrc1@48=0x{k.kd['compute_pgm_rsrc1']:08x} "
                f"rsrc2@52=0x{k.kd['compute_pgm_rsrc2']:08x} "
                f"props@56=0x{k.kd['kernel_code_properties']:04x} "
                f"preload@58=0x{k.kd['kernarg_preload_raw']:04x}"
            ),
        })
    else:
        row["kd_preload"] = None
        if k.kd_symbol is None:
            null_reasons["kd_preload"] = (
                f"kernel descriptor not found (no {k.name + '.kd'!r} symbol, "
                "no usable .amdhsa.kd entry)"
            )
        else:
            null_reasons["kd_preload"] = k.kd.get("error", "descriptor unreadable")

    # ---- evidence: symbol + metadata + target -------------------------
    if k.code_symbol is not None:
        sym = k.code_symbol
        evidence.append({
            "source": "symbol",
            "offset": co.elf.symbol_file_offset(sym) if co.elf else None,
            "mnemonic": None,
            "detail": (
                f"name={sym.name} section_index={sym.shndx} "
                f"value=0x{sym.value:x} size={sym.size}"
            ),
        })
    if k.code_symbol is None and k.code_reason:
        null_reasons.setdefault("isa_size", k.code_reason)

    if k.metadata:
        if k.metadata.get("note_offset") is not None:
            evidence.append({
                "source": "metadata",
                "offset": k.metadata["note_offset"],
                "mnemonic": None,
                "detail": (
                    f"NT_AMDGPU_METADATA note in {k.metadata.get('note_section')!r} "
                    f"schema={k.metadata.get('schema')} "
                    f"amdhsa.version={k.metadata.get('amdhsa_version')} "
                    f"target={k.metadata.get('amdhsa_target')}"
                ),
            })
        for f, fl in ((x, k.get(x)) for x in field_sources):
            if fl.value is not None and fl.source == "metadata":
                evidence.append({
                    "source": "metadata-field",
                    "offset": k.metadata.get("note_offset"),
                    "mnemonic": None,
                    "detail": f"{fl.detail} -> {fl.value}",
                })
                break  # one concrete key-path sample is enough per row
    if co.target:
        evidence.append({
            "source": "elf-header",
            "offset": 0,
            "mnemonic": None,
            "detail": (
                f"e_machine={co.target['e_machine']} "
                f"EF_AMDGPU_MACH=0x{co.target['mach']:02x} "
                f"({co.target['mach_name']}) osabi={co.target['osabi']} "
                f"abiversion={co.target['abiversion']} "
                f"e_type={co.target['e_type']}"
            ),
        })

    # ---- classification ------------------------------------------------
    insns = None
    sym_name = k.code_symbol.name if k.code_symbol else k.name
    if dis_err:
        cls = classify_mod.classify(None)
        null_reasons["likely_operation"] = dis_err
        null_reasons["confidence"] = dis_err
        disasm_info = {"available": False, "error": dis_err, "mcpu": mcpu_used,
                       "symbol": sym_name, "insn_count": None}
    elif dis is not None and sym_name in dis.per_symbol:
        raw_insns = dis.per_symbol[sym_name]
        # objdump disassembles the whole section under a symbol header;
        # keep only what belongs to this kernel's code range.
        insns = [
            i for i in raw_insns
            if isa_size is None or i.offset < isa_size
        ]
        cls = classify_mod.classify(insns)
        disasm_info = {"available": True, "error": None, "mcpu": mcpu_used,
                       "symbol": sym_name, "insn_count": len(insns)}
        for ev in cls.evidence:
            evidence.append({
                "source": "disasm",
                "offset": ev.offset,
                "mnemonic": ev.mnemonic,
                "detail": ev.detail,
            })
        if cls.error and cls.likely_operation is None:
            null_reasons["likely_operation"] = cls.error
            null_reasons["confidence"] = cls.error
    else:
        cls = classify_mod.classify(None)
        reason = (
            f"symbol {sym_name!r} not present in disassembly output "
            "(stripped/aliased symbol)" if dis is not None else "no disassembly"
        )
        null_reasons["likely_operation"] = reason
        null_reasons["confidence"] = reason
        disasm_info = {"available": dis is not None, "error": reason,
                       "mcpu": mcpu_used, "symbol": sym_name, "insn_count": None}

    row["likely_operation"] = cls.likely_operation
    row["confidence"] = cls.confidence
    if cls.likely_operation is None and "likely_operation" not in null_reasons:
        null_reasons["likely_operation"] = cls.error or "classifier returned no label"
    if cls.confidence is None and "confidence" not in null_reasons:
        null_reasons["confidence"] = cls.error or "classifier returned no confidence"

    row["disassembly"] = disasm_info
    row["classification_rules"] = cls.fired_rules
    row["field_sources"] = field_sources
    row["metadata"] = k.metadata
    row["evidence"] = evidence
    row["null_reasons"] = null_reasons

    # contract invariant: null -> reason
    for f in NULLABLE_CONTRACT_FIELDS:
        if row.get(f) is None and f not in null_reasons:
            raise AssertionError(
                f"kernel {k.name!r}: field {f} is null without a reason (bug guard)"
            )
    return row
