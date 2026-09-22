"""llvm-objdump discovery, invocation, and output parsing.

One disassembly pass per code object; results are grouped per symbol so
the classifier can consume a kernel's instructions directly.  When no
toolchain is available every kernel degrades to classification=null with
a reason (honest degradation; asserted by tests).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

TOOL_CANDIDATES = (
    "/opt/homebrew/opt/llvm/bin/llvm-objdump",
    "/usr/local/opt/llvm/bin/llvm-objdump",
    "/opt/rocm/llvm/bin/llvm-objdump",
    "/opt/rocm/llvm/bin/llvm-objdump-19",
)


@dataclass
class Insn:
    offset: int          # byte offset within the owning symbol (kernel-relative)
    vma: int             # address as printed by objdump
    mnemonic: str
    operands: str
    raw_bytes: str       # hex bytes as printed by objdump


@dataclass
class DisasmResult:
    tool_path: str | None
    tool_version: str | None
    commands: list = field(default_factory=list)     # exact argv strings used
    per_symbol: dict = field(default_factory=dict)   # symbol -> [Insn]
    headers: list = field(default_factory=list)      # (vma, symbol) in order
    error: str | None = None
    raw: str = ""


_SYM_RE = re.compile(r"^([0-9A-Fa-f]+)\s+<([^>]+)>:$")
_INSN_RE = re.compile(
    r"^\s*(\S+)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*"
    r"([0-9A-Fa-f]+(?:\s+[0-9A-Fa-f]+)*)\s*$"
)
# insurance: instruction-looking line without the '// ADDR: BYTES' comment
# (format drift across llvm versions must never silently drop code)
_INSN_LOOSE_RE = re.compile(r"^\t([A-Za-z][A-Za-z0-9_.]*)\t?(\S.*?)\s*$")


def discover(prefer: str | None = None) -> tuple[str | None, str | None]:
    """-> (path, None) or (None, reason).  Env HALOGEN_LLVM_OBJDUMP wins."""
    env = os.environ.get("HALOGEN_LLVM_OBJDUMP") or os.environ.get("LLVM_OBJDUMP")
    tried = []
    for cand in ([prefer] if prefer else []) + ([env] if env else []):
        if cand is None:
            continue
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand, None
        tried.append(cand)
        return None, f"configured llvm-objdump {cand!r} not found or not executable"
    w = shutil.which("llvm-objdump")
    if w:
        return w, None
    for p in TOOL_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p, None
    return None, (
        "llvm-objdump not found (checked HALOGEN_LLVM_OBJDUMP, PATH, "
        "brew llvm, /opt/rocm); disassembly and ISA classification unavailable"
    )


def toolchain_info(tool: str | None, tool_err: str | None) -> dict:
    info = {
        "llvm_objdump": {"path": tool, "version": None, "error": tool_err},
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if tool:
        try:
            out = subprocess.run(
                [tool, "--version"], capture_output=True, text=True, timeout=30
            )
            lines = (out.stdout or out.stderr).strip().splitlines()
            info["llvm_objdump"]["version"] = " | ".join(lines[:3])
        except (OSError, subprocess.SubprocessError) as e:
            info["llvm_objdump"]["error"] = f"running {tool} --version failed: {e}"
    return info


def mcpu_for(target: dict | None) -> str | None:
    """gfx target for --mcpu from metadata target string or ELF mach."""
    if target:
        t = target.get("amdhsa_target") if isinstance(target, dict) else None
        if isinstance(t, str) and "gfx" in t:
            # 'amdgcn-amd-amdhsa-unknown-gfx1151' / target-ids with ':feat'
            tail = t.rsplit("-", 1)[-1].split(":")[0]
            if tail.startswith("gfx"):
                return tail
        name = target.get("mach_name") if isinstance(target, dict) else None
        if name:
            return name
    return None


def disassemble(tool: str, code_object_bytes: bytes, mcpu: str | None,
                label: str) -> DisasmResult:
    res = DisasmResult(tool_path=tool, tool_version=None)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="halogen_co_", suffix=".o",
                                         delete=False) as tf:
            tf.write(code_object_bytes)
            tmp_path = tf.name
        argv = [tool, "-d", "--arch-name=amdgcn"]
        if mcpu:
            argv.append(f"--mcpu={mcpu}")
        argv.append(tmp_path)
        # retry without --mcpu if the cpu name is unknown to this llvm
        variants = [argv]
        if mcpu:
            variants.append([tool, "-d", "--arch-name=amdgcn", tmp_path])
        last_err = None
        for cmd in variants:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except (OSError, subprocess.SubprocessError) as e:
                res.error = f"llvm-objdump invocation failed: {e}"
                return res
            if proc.returncode == 0:
                res.commands.append(" ".join(_redact_tmp(cmd, label)))
                _parse(proc.stdout, res)
                return res
            last_err = (proc.stderr or proc.stdout or "").strip().splitlines()
            last_err = last_err[0] if last_err else f"exit {proc.returncode}"
        res.error = f"llvm-objdump failed: {last_err}"
        res.commands.append(" ".join(_redact_tmp(variants[-1], label)))
        return res
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _redact_tmp(argv, label):
    return [label if "halogen_co_" in a else a for a in argv]


def _parse(text: str, res: DisasmResult) -> None:
    cur_sym = None
    cur_base = None
    last_off = None
    for line in text.splitlines():
        m = _SYM_RE.match(line)
        if m:
            vma = int(m.group(1), 16)
            cur_sym = m.group(2)
            cur_base = vma
            last_off = None
            res.headers.append((vma, cur_sym))
            res.per_symbol.setdefault(cur_sym, [])
            continue
        if cur_sym is None:
            continue
        m = _INSN_RE.match(line)
        if m:
            mnem, ops, vma_s, hexb = m.groups()
            if mnem in ("Disassembly", "of") or mnem.startswith("."):
                continue
            vma = int(vma_s, 16)
            res.per_symbol[cur_sym].append(
                Insn(
                    offset=vma - (cur_base if cur_base is not None else vma),
                    vma=vma,
                    mnemonic=mnem,
                    operands=ops.strip(),
                    raw_bytes=hexb.lower(),
                )
            )
            last_off = res.per_symbol[cur_sym][-1].offset
            continue
        m = _INSN_LOOSE_RE.match(line)
        if m and "<" not in line and "//" not in line:
            mnem, ops = m.groups()
            if mnem.startswith(".") or mnem in ("Disassembly",):
                continue
            # no address comment: keep the instruction with a sequential
            # best-effort offset so counts stay honest instead of dropping
            off = 0 if last_off is None else last_off + 4
            res.per_symbol[cur_sym].append(
                Insn(offset=off, vma=(cur_base or 0) + off,
                     mnemonic=mnem, operands=ops.strip(), raw_bytes="")
            )
            last_off = off
