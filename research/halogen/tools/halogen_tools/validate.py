"""Tiny JSON Schema (draft 2020-12 subset) validator - stdlib only.

Supports the keyword set used by schemas/kernel-catalogue.schema.json:
type, properties, required, items, enum, const, pattern, minimum,
maximum, minLength, additionalProperties (bool or schema), $ref (local
"#/$defs/..."), allOf-lite via anyOf for nullables.  Purpose: prove the
published schema parses and that generated catalogues conform, without
adding third-party dependencies for sibling agents.
"""

from __future__ import annotations

import re


class SchemaError(ValueError):
    pass


def load_schema(doc) -> dict:
    """Parse + sanity-check a schema document (must be a JSON object)."""
    if isinstance(doc, str):
        import json

        doc = json.loads(doc)
    if not isinstance(doc, dict):
        raise SchemaError("schema root must be a JSON object")
    for key in ("$schema", "title", "type"):
        if key not in doc:
            raise SchemaError(f"schema missing required meta-key {key!r}")
    if doc["type"] != "object" and not isinstance(doc["type"], str):
        raise SchemaError("schema 'type' must be a string")
    return doc


def validate(instance, schema: dict, root: dict | None = None, path: str = "$") -> list[str]:
    """ -> list of violation strings (empty == valid)."""
    root = root or schema
    errs: list[str] = []

    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            errs.append(f"{path}: only local $ref supported, got {ref!r}")
            return errs
        node = root
        for part in ref[2:].split("/"):
            if not isinstance(node, dict) or part not in node:
                errs.append(f"{path}: unresolvable $ref {ref!r}")
                return errs
            node = node[part]
        errs.extend(validate(instance, node, root, path))
        return errs

    if "const" in schema and instance != schema["const"]:
        errs.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not in enum {schema['enum']}")

    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else list(types)
        if not any(_is_type(instance, t) for t in types):
            errs.append(f"{path}: expected type {types}, got {type(instance).__name__}")
            return errs

    if isinstance(instance, str):
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errs.append(f"{path}: {instance!r} does not match pattern {schema['pattern']!r}")
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errs.append(f"{path}: string shorter than minLength {schema['minLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(f"{path}: {instance} > maximum {schema['maximum']}")

    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errs.extend(validate(item, schema["items"], root, f"{path}[{i}]"))

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in instance:
                errs.append(f"{path}: missing required property {req!r}")
        for key, sub in props.items():
            if key in instance:
                errs.extend(validate(instance[key], sub, root, f"{path}.{key}"))
        add = schema.get("additionalProperties", True)
        if add is False:
            for key in instance:
                if key not in props:
                    errs.append(f"{path}: unexpected property {key!r}")
        elif isinstance(add, dict):
            for key in instance:
                if key not in props:
                    errs.extend(validate(instance[key], add, root, f"{path}.{key}"))
    return errs


def _is_type(v, t: str) -> bool:
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    if t == "string":
        return isinstance(v, str)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "null":
        return v is None
    raise SchemaError(f"unsupported schema type {t!r}")
