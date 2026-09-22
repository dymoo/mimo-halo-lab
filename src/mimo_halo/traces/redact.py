"""Credential-pattern redaction for normalized private outputs.

This removes well-known secret shapes. It is redaction, NOT a guarantee of
complete anonymization: free-form credentials without recognizable prefixes,
paths, hostnames and proprietary content may remain and must never leave the
workstation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|[A-Za-z0-9])[A-Za-z0-9_\-]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret", re.compile(r"(?i)aws.{0,20}?['\"][0-9a-zA-Z/+]{40}['\"]")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("github_fine_grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[aebp]-[A-Za-z0-9\-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openrouter_key", re.compile(r"\bsk-or-[A-Za-z0-9_\-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("bearer_header", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{20,}\b")),
    ("basic_auth_url", re.compile(r"\bhttps?://[^\s/:@]+:[^\s/@]+@[^\s]+")),
    (
        "secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|apikey|secret|token|passwd|password|passphrase|client[_-]?secret|private[_-]?key)\b"
            r"['\"]?\s*[:=]\s*(?:['\"][^'\"\s]{12,}['\"]|[^\s'\"]{12,})"
        ),
    ),
    ("long_hex_secret", re.compile(r"\b[0-9a-f]{64,}\b")),
)


@dataclass
class RedactionResult:
    text: str
    hits: int
    kinds: list[str]


def redact(text: str) -> RedactionResult:
    """Redact known credential shapes; returns new text plus hit statistics."""
    hits = 0
    kinds: list[str] = []
    out = text
    for kind, pattern in _KINDS:
        out, n = pattern.subn(f"[REDACTED:{kind}]", out)
        if n:
            hits += n
            kinds.append(kind)
    return RedactionResult(text=out, hits=hits, kinds=kinds)
