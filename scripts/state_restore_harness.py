#!/usr/bin/env python3
"""NVMe/SWA whole-context continuation harness (stdlib only).

Compares an uninterrupted HOT continuation against a
save -> erase/free -> restore -> same continuation run under identical
request arguments, against an EXTERNALLY managed llama-server
(`--base-url`); this harness never spawns a server, builds, or downloads
anything except the pinned bounded fixture via `fetch-fixture`.

What is measured (all observations emitted as one JSON report):
  * per-request `timings.prompt_n` / `timings.cache_n` (physical prefill
    suffix bound) and the global `llamacpp:prompt_tokens_total` /
    `llamacpp:prompt_tokens_cached_total` counter deltas around each op;
  * generated token ids and, where accessible, per-token
    `completion_probabilities` (top logprobs) for hot vs restored arms;
  * logical position accounting: prompt_n + cache_n == len(prompt array)
    plus `usage.prompt_tokens` / `usage.prompt_tokens_details.cached_tokens`
    when the server reports them;
  * slot-save/restore/erase bookkeeping counts (`n_saved`, `n_restored`,
    `n_read`, `n_erased`) against exact expected token counts;
  * RAM prompt-cache disabled: launch-recipe declaration checks plus a
    behavioral cold probe after erase asserting `cache_n == 0` (no false
    warm hit).

No RSS-drop claim: freeing hot state means sequence-cache ownership /
cells are reusable (proved by the cold probe and by restore succeeding
afterward); the preallocated KV pool resident size need not fall.

Exit codes: 0 all assertions passed, 1 assertion failure, 2 transport or
usage failure (report still written with `error` set).

Commands:
  fetch-fixture   pinned, size-capped, sha256-verified stream into the
                  git-ignored cache (never a moving HF branch)
  run             the full instrumented comparison
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "state-restore-fixture.json"


class HarnessError(Exception):
    """Transport / server / fixture failure (not an assertion failure)."""


# --------------------------------------------------------------------------
# config + fixtures
# --------------------------------------------------------------------------

def load_config(path: str | None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read config {cfg_path}: {exc}") from exc
    cfg["_path"] = str(cfg_path)
    return cfg


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_cache_path(cfg: dict) -> Path:
    return REPO_ROOT / cfg["fixture"]["cache_path"]


def verify_fixture(cfg: dict) -> dict:
    fx = cfg["fixture"]
    dest = fixture_cache_path(cfg)
    if not dest.exists():
        return {"status": "absent", "path": str(dest),
                "note": "server loads the fixture itself; local file optional"}
    size = dest.stat().st_size
    if size != fx["size_bytes"]:
        return {"status": "size_mismatch", "path": str(dest),
                "expected_bytes": fx["size_bytes"], "actual_bytes": size}
    digest = sha256_file(dest)
    if digest != fx["sha256"]:
        return {"status": "sha256_mismatch", "path": str(dest),
                "expected_sha256": fx["sha256"], "actual_sha256": digest}
    return {"status": "verified", "path": str(dest),
            "bytes": size, "sha256": digest}


def fetch_fixture(cfg: dict, force: bool) -> dict:
    """Stream the pinned revision into the ignored cache with a hard size
    cap (expected size, exactly) and sha256 verification; atomic rename."""
    fx = cfg["fixture"]
    if f"/resolve/{fx['revision']}/" not in fx["resolve_url"]:
        raise HarnessError(
            f"resolve_url is not pinned to revision {fx['revision']}: "
            f"{fx['resolve_url']}")
    dest = fixture_cache_path(cfg)
    if not force:
        status = verify_fixture(cfg)
        if status["status"] == "verified":
            status["fetched"] = False
            return status

    cap = int(fx["size_bytes"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    got = 0
    try:
        req = urllib.request.Request(
            fx["resolve_url"], headers={"User-Agent": "state-restore-harness"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            length = resp.headers.get("Content-Length")
            if length is not None and int(length) > cap:
                raise HarnessError(
                    f"server declares {length} bytes > cap {cap}")
            with open(part, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    got += len(chunk)
                    if got > cap:
                        raise HarnessError(
                            f"stream exceeded size cap {cap}; aborting")
                    out.write(chunk)
                    digest.update(chunk)
    except HarnessError:
        part.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError) as exc:
        part.unlink(missing_ok=True)
        raise HarnessError(f"fixture fetch failed: {exc}") from exc

    if got != cap:
        part.unlink(missing_ok=True)
        raise HarnessError(f"size mismatch: got {got}, expected {cap}")
    hexdigest = digest.hexdigest()
    if hexdigest != fx["sha256"]:
        part.unlink(missing_ok=True)
        raise HarnessError(
            f"sha256 mismatch: got {hexdigest}, expected {fx['sha256']}")
    os.replace(part, dest)
    return {"status": "verified", "fetched": True, "path": str(dest),
            "bytes": got, "sha256": hexdigest,
            "license": fx["license"]}


# --------------------------------------------------------------------------
# HTTP + metrics
# --------------------------------------------------------------------------

def http(base: str, method: str, path: str, payload=None, timeout: float = 60):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:2000]
        except OSError:
            pass
        raise HarnessError(f"{method} {path} -> HTTP {exc.code}: {detail}") \
            from exc
    except (OSError, urllib.error.URLError) as exc:
        raise HarnessError(f"{method} {path} failed: {exc}") from exc

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "json" in ctype:
        return json.loads(body)
    return body.decode("utf-8", "replace")


def scrape_counters(base: str, timeout: float = 60) -> dict:
    """Parse llamacpp:* counters from the Prometheus text endpoint."""
    try:
        text = http(base, "GET", "/metrics", timeout=timeout)
    except HarnessError as exc:
        if "501" in str(exc):
            raise HarnessError(
                "/metrics unavailable (501): launch with --metrics") from exc
        raise
    if not isinstance(text, str):
        raise HarnessError("/metrics did not return a text body")
    counters = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue  # skip labeled series (e.g. spec_decode per-position)
        name, value = parts
        try:
            counters[name] = float(value)
        except ValueError:
            continue
    for required in ("llamacpp:prompt_tokens_total",
                     "llamacpp:prompt_tokens_cached_total"):
        if required not in counters:
            raise HarnessError(f"required counter {required} missing")
    return counters


def counter_delta(before: dict, after: dict) -> dict:
    return {name: after[name] - before.get(name, 0.0) for name in after}


# --------------------------------------------------------------------------
# token corpus (real token ids from the server's own tokenizer)
# --------------------------------------------------------------------------

def build_corpus(base: str, needed: int, timeout: float, line: str) -> dict:
    """Deterministic corpus; token ids come from POST /tokenize so history
    and suffix are sent as arrays and never re-tokenized by BPE."""
    lines = max(64, needed // 12)
    last_seen = 0
    for _ in range(5):
        text = "\n".join(f"{i:06d} {line}" for i in range(lines))
        resp = http(base, "POST", "/tokenize",
                    {"content": text, "add_special": False}, timeout)
        tokens = resp.get("tokens")
        if not isinstance(tokens, list):
            raise HarnessError(f"/tokenize returned no tokens: {resp!r:.400}")
        last_seen = len(tokens)
        if last_seen >= needed:
            return {"tokens": tokens, "corpus_lines": lines,
                    "corpus_tokens": last_seen,
                    "corpus_sha256": hashlib.sha256(text.encode()).hexdigest()}
        lines *= 2
    raise HarnessError(
        f"corpus stayed short after retries: {last_seen} < {needed}")


# --------------------------------------------------------------------------
# report / assertions
# --------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, ok: bool, detail="") -> bool:
        entry = {"name": name, "ok": bool(ok), "detail": str(detail)[:2000]}
        self.checks.append(entry)
        print(f"  [{'ok' if ok else 'FAIL'}] {name}"
              + (f": {detail}" if detail else ""), file=sys.stderr, flush=True)
        return entry["ok"]

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def prune_completion(resp: dict) -> dict:
    """Keep every counter/timing/logprob field, drop only the bulky text."""
    out = {k: v for k, v in resp.items() if k != "content"}
    if isinstance(resp.get("content"), str):
        out["content_chars"] = len(resp["content"])
    return out


def generated_ids(resp: dict) -> list:
    """Generated token ids from `return_tokens` (raw ids, never detokenized
    then re-tokenized)."""
    tokens = resp.get("tokens")
    if not isinstance(tokens, list):
        return []
    return [t for t in tokens if isinstance(t, int)]


# --------------------------------------------------------------------------
# recipe declaration checks
# --------------------------------------------------------------------------

def check_recipe(rep: Report, variant: dict) -> list:
    args = list(variant["launch_args"])

    def has_pair(flag: str, value: str | None = None) -> bool:
        for i, tok in enumerate(args):
            if tok == flag:
                return value is None or (
                    i + 1 < len(args) and args[i + 1] == value)
        return False

    checks = [
        ("recipe.cache_ram_disabled", has_pair("--cache-ram", "0"),
         "--cache-ram 0 (RAM prompt cache off, no false warm hits)"),
        ("recipe.no_cache_idle_slots",
         has_pair("--no-cache-idle-slots"),
         "--no-cache-idle-slots (cache-idle-slots requires cache-ram)"),
        ("recipe.no_swa_full", not any("--swa-full" == a for a in args),
         "--swa-full not forced (would mask the pre-patch bug)"),
        ("recipe.metrics_enabled", has_pair("--metrics"),
         "--metrics (physical counter deltas)"),
        ("recipe.slots_enabled", has_pair("--slots"), "--slots"),
        ("recipe.slot_save_path", has_pair("--slot-save-path"),
         "--slot-save-path (save/erase/restore enabled)"),
        ("recipe.offline_local_model",
         has_pair("--offline") and has_pair("--model")
         and "-hf" not in args and "--hf-repo" not in args
         and "--hf-file" not in args,
         "--model + --offline, no -hf/--hf-repo moving-branch download"),
        ("recipe.threads_1_cpu_first", has_pair("--threads", "1")
         and has_pair("--n-gpu-layers", "0"),
         "--threads 1 and --n-gpu-layers 0 (CPU first)"),
        ("recipe.parallel_1", has_pair("--parallel", "1"),
         "--parallel 1 (single slot gets the full ctx)"),
        ("recipe.ctx_matches_variant", has_pair("--ctx-size", str(variant["n_ctx"])),
         f"--ctx-size {variant['n_ctx']}"),
        ("recipe.batch_matches_variant",
         has_pair("--batch-size", str(variant["batch_size"]))
         and has_pair("--ubatch-size", str(variant["ubatch_size"])),
         f"--batch-size {variant['batch_size']} / "
         f"--ubatch-size {variant['ubatch_size']} (names the replay bound)"),
    ]
    for name, ok, detail in checks:
        rep.check(name, ok, detail)
    return checks


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run(base: str, cfg: dict, variant_name: str) -> tuple[dict, int]:
    if variant_name not in cfg["variants"]:
        raise HarnessError(
            f"unknown variant {variant_name!r}; "
            f"have {sorted(cfg['variants'])}")
    variant = cfg["variants"][variant_name]
    det = cfg["determinism"]
    slot_id = int(cfg["slot_id"])
    timeout = float(variant["request_timeout_s"])
    rep = Report()

    # Named bounds ---------------------------------------------------------
    # Continuation prompt = saved_state (LCP) + suffix. Physical prefill of
    # a correct restore replays only:
    #   suffix_tokens                     newly appended tokens
    # + batch_size (final-fill gap)       near-end checkpoint is created
    #                                      before the last decode batch
    #                                      (server-context.cpp:3550-3564),
    #                                      so it trails the prompt end by at
    #                                      most one batch
    # + n_predict (generated tail)        G0 was generated after the warm
    #                                      prefill, so it is past the
    #                                      checkpoint too
    # + replay_boundary_tokens (=1)       documented forced last-token
    #                                     replay [TAG_PROMPT_LOGITS]
    #                                     (server-context.cpp:3641-3644)
    boundary = int(variant["replay_boundary_tokens"])
    replay_bound = (int(variant["batch_size"])
                    + int(variant["n_predict"]) + boundary)
    bounds = {
        "history_tokens": variant["history_tokens"],
        "suffix_tokens": variant["suffix_tokens"],
        "replay_boundary_tokens": boundary,
        "final_fill_bound_tokens": variant["batch_size"],
        "generated_bound_tokens": variant["n_predict"],
        "replay_bound_tokens": replay_bound,
        "prompt_n_min": variant["suffix_tokens"],
        "prompt_n_max": variant["suffix_tokens"] + replay_bound,
        "restored_prompt_n_max": variant["suffix_tokens"] + replay_bound,
        "derivation": (
            "suffix + batch_size (checkpoint final-fill gap) + n_predict "
            "(generated tail) + replay_boundary_tokens (documented forced "
            "last-token replay); full re-prefill = history+G0+suffix is far "
            "above this bound"),
    }

    report: dict = {
        "schema_version": 1,
        "variant": variant_name,
        "base_url": base,
        "config_path": cfg["_path"],
        "bounds": bounds,
        "determinism": det,
        "fixture": dict(cfg["fixture"], **verify_fixture(cfg)),
        "notes": [
            "Hot-state freed means sequence-cache ownership/cells are "
            "reusable (cold probe re-populates the slot; restore then "
            "succeeds); preallocated KV-pool RSS need not fall — no RSS "
            "drop is claimed.",
            "Timings are verbatim server timings plus client-observed "
            "wall_ms; no GPU time or p95 is fabricated.",
            "Logprob comparison uses greedy temperature-0 outputs; the "
            "pinned README documents batch-shape logit nondeterminism "
            "(upstreams/strix-llama.cpp README, cache-prompt note), hence "
            "the configurable logprob_abs_tol.",
            "Fixture license recorded as declared upstream metadata "
            "(WTFPL), not a legal approval.",
        ],
        "observations": {},
        "metrics": {},
        "comparison": {},
        "assertions": rep.checks,
    }

    try:
        # Launch-recipe declaration checks -------------------------------------
        snapshot_dir = REPO_ROOT / cfg["snapshot"]["dir"]
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        report["launch_recipe"] = {
            "variant": variant_name,
            "args": variant["launch_args"],
            "snapshot_dir": str(snapshot_dir),
            "snapshot_file": cfg["snapshot"]["filename"].format(
                variant=variant_name),
        }
        rep.check("recipe.snapshot_dir_configured",
                  bool(cfg["snapshot"].get("dir")),
                  f"snapshot dir {cfg['snapshot'].get('dir')}")
        check_recipe(rep, variant)

        # Fixture observation ---------------------------------------------------
        report["fixture"]["local_status"] = report["fixture"].pop("status", "absent")
        local = report["fixture"]["local_status"]
        rep.check("fixture.local_consistent", local in ("absent", "verified"),
                  f"local cache: {local}"
                  + ("" if local == "absent" else " (size+sha256 match pin)"))

        # Health / props / slots ------------------------------------------------
        deadline = time.monotonic() + 90.0
        health = None
        while True:
            try:
                health = http(base, "GET", "/health", timeout=15)
                break
            except HarnessError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)
        rep.check("server.health_ok", True, f"/health -> {health!r}"[:200])
        report["observations"]["health"] = health

        props = http(base, "GET", "/props", timeout=30)
        report["observations"]["props"] = props
        slot_ctx = (props.get("default_generation_settings", {}) or {}).get("n_ctx")
        total_slots = props.get("total_slots")
        rep.check("server.ctx_matches_variant", slot_ctx == variant["n_ctx"],
                  f"slot n_ctx {slot_ctx} == variant {variant['n_ctx']}")
        rep.check("server.single_slot", total_slots == 1,
                  f"total_slots {total_slots} (recipe --parallel 1)")

        slots = http(base, "GET", "/slots", timeout=30)
        report["observations"]["slots_before"] = slots
        rep.check("server.slot_id_available",
                  isinstance(slots, list)
                  and any(s.get("id") == slot_id for s in slots),
                  f"slot {slot_id} present")

        snapshot_file = cfg["snapshot"]["filename"].format(variant=variant_name)

        def op(label, method, path, payload=None, to=60):
            before = scrape_counters(base, to)
            t0 = time.monotonic()
            body = http(base, method, path, payload, to)
            wall_ms = round((time.monotonic() - t0) * 1000.0, 3)
            after = scrape_counters(base, to)
            delta = counter_delta(before, after)
            if isinstance(body, dict):
                rec = {"request": payload, **prune_completion(body),
                       "metrics_delta": delta, "wall_ms": wall_ms}
            else:
                rec = {"request": payload, "body": body,
                       "metrics_delta": delta, "wall_ms": wall_ms}
            report["observations"][label] = rec
            return rec, delta

        def complete(label, payload, to=timeout):
            return op(label, "POST", "/completion", payload, to)

        # Expected logical sizes ------------------------------------------------
        history_n = int(variant["history_tokens"])
        suffix_n = int(variant["suffix_tokens"])
        n_predict = int(variant["n_predict"])

        corpus = build_corpus(base, history_n + suffix_n, timeout,
                              cfg["corpus"]["line"])
        history = corpus["tokens"][:history_n]
        suffix = corpus["tokens"][history_n:history_n + suffix_n]
        rep.check("tokenize.sufficient",
                  len(history) == history_n and len(suffix) == suffix_n,
                  f"corpus {corpus['corpus_tokens']} tokens "
                  f"(history {len(history)}, suffix {len(suffix)})")
        report["tokenization"] = {k: v for k, v in corpus.items()
                                  if k != "tokens"} | {
            "history_first": history[0], "history_last": history[-1],
            "suffix_first": suffix[0], "suffix_last": suffix[-1]}

        baseline = scrape_counters(base, timeout)
        request_deltas: dict[str, dict] = {}

        # 0. clean slate --------------------------------------------------------
        _, d0 = op("erase_initial", "POST",
                   f"/slots/{slot_id}?action=erase")
        rep.check("erase_initial.no_physical_prompt",
                  d0["llamacpp:prompt_tokens_total"] == 0,
                  f"delta {d0['llamacpp:prompt_tokens_total']}")
        request_deltas["erase_initial"] = d0

        # 1. warm: prefill history, generate G0 (hot state on slot) ------------
        warm_payload = {
            "prompt": history, "id_slot": slot_id,
            "cache_prompt": bool(det["cache_prompt"]),
            "temperature": det["temperature"], "seed": det["seed"],
            "n_predict": n_predict, "return_tokens": bool(det["return_tokens"]),
            "n_probs": det["n_probs"],
        }
        warm, dw = complete("warm", warm_payload)
        g0 = generated_ids(warm)
        prompt_n = warm.get("timings", {}).get("prompt_n")
        cache_n = warm.get("timings", {}).get("cache_n")
        saved_state = history + g0
        rep.check("warm.fresh_slot_full_prefill",
                  prompt_n == history_n and cache_n == 0,
                  f"prompt_n {prompt_n} == history {history_n}, cache_n {cache_n}")
        rep.check("warm.generated_within_n_predict", len(g0) <= n_predict,
                  f"G0 length {len(g0)} <= n_predict {n_predict}")
        rep.check("warm.metrics_delta_physical",
                  dw["llamacpp:prompt_tokens_total"] == prompt_n,
                  f"delta {dw['llamacpp:prompt_tokens_total']} == prompt_n "
                  f"{prompt_n}")
        rep.check("warm.metrics_delta_cached",
                  dw["llamacpp:prompt_tokens_cached_total"] == cache_n,
                  f"delta {dw['llamacpp:prompt_tokens_cached_total']} == cache_n "
                  f"{cache_n}")
        request_deltas["warm"] = dw
        report["observations"]["warm"]["generated_tokens"] = g0

        # 2. save: prefix + generated history ----------------------------------
        save_payload = {"filename": snapshot_file}
        save, ds = op("save", "POST", f"/slots/{slot_id}?action=save",
                      save_payload)
        expected_saved = len(saved_state) - boundary
        rep.check("save.n_saved_exact",
                  save.get("n_saved") == expected_saved,
                  f"n_saved {save.get('n_saved')} == len(history)+len(G0) "
                  f"- replay_boundary_tokens = {len(saved_state)} - {boundary} "
                  f"(each processing pass persists prompt+generated minus the "
                  f"replay boundary token — the forced last-token replay "
                  f"[TAG_PROMPT_LOGITS] re-feeds it, "
                  f"server-context.cpp:3641-3644; hot arm cache_n = len-1 "
                  f"corroborates)")
        rep.check("save.n_written_positive", (save.get("n_written") or 0) > 0,
                  f"n_written {save.get('n_written')} bytes")
        rep.check("save.no_physical_prompt",
                  ds["llamacpp:prompt_tokens_total"] == 0,
                  f"delta {ds['llamacpp:prompt_tokens_total']}")
        request_deltas["save"] = ds

        # 3. HOT arm: uninterrupted continuation ------------------------------
        continuation_prompt = saved_state + suffix
        continuation_payload = {
            "prompt": continuation_prompt, "id_slot": slot_id,
            "cache_prompt": bool(det["cache_prompt"]),
            "temperature": det["temperature"], "seed": det["seed"],
            "n_predict": n_predict, "return_tokens": bool(det["return_tokens"]),
            "n_probs": det["n_probs"],
        }
        hot, dh = complete("hot_continuation", continuation_payload)
        request_deltas["hot_continuation"] = dh
        g1 = generated_ids(hot)
        report["observations"]["hot_continuation"]["generated_tokens"] = g1

        def completion_bounds(label: str, resp: dict, delta: dict) -> None:
            t = resp.get("timings") or {}
            u = resp.get("usage") or {}
            details = (u.get("prompt_tokens_details") or {})
            pn, cn = t.get("prompt_n"), t.get("cache_n")
            prompt_len = len(continuation_prompt)
            rep.check(f"{label}.prefill_suffix_bound",
                      pn is not None and bounds["prompt_n_min"] <= pn
                      <= bounds["prompt_n_max"],
                      f"prompt_n {pn} in [{bounds['prompt_n_min']}, "
                      f"{bounds['prompt_n_max']}] "
                      f"(suffix {suffix_n} + bound {replay_bound}); "
                      f"full re-prefill would be {prompt_len}")
            rep.check(f"{label}.position_identity",
                      pn is not None and cn is not None
                      and pn + cn == prompt_len,
                      f"prompt_n {pn} + cache_n {cn} == len(prompt) "
                      f"{prompt_len}")
            if u.get("prompt_tokens") is not None:
                rep.check(f"{label}.usage_prompt_tokens",
                          u["prompt_tokens"] == prompt_len,
                          f"usage.prompt_tokens {u['prompt_tokens']} == "
                          f"{prompt_len}")
            if details.get("cached_tokens") is not None:
                rep.check(f"{label}.usage_cached_tokens",
                          details["cached_tokens"] == cn,
                          f"prompt_tokens_details.cached_tokens "
                          f"{details['cached_tokens']} == cache_n {cn}")
            rep.check(f"{label}.metrics_delta_physical",
                      pn is not None
                      and delta["llamacpp:prompt_tokens_total"] == pn,
                      f"global delta {delta['llamacpp:prompt_tokens_total']} == "
                      f"prompt_n {pn}")
            rep.check(f"{label}.metrics_delta_cached",
                      cn is not None
                      and delta["llamacpp:prompt_tokens_cached_total"] == cn,
                      f"global delta "
                      f"{delta['llamacpp:prompt_tokens_cached_total']} == "
                      f"cache_n {cn}")

        completion_bounds("hot", hot, dh)

        # 4. erase: free hot state; exact slot bookkeeping ---------------------
        erase, de = op("erase_after_hot", "POST",
                       f"/slots/{slot_id}?action=erase")
        expected_erased = (len(history) + len(g0) + len(suffix) + len(g1)
                           - boundary)
        rep.check("erase.n_erased_exact_slot_bookkeeping",
                  erase.get("n_erased") == expected_erased,
                  f"n_erased {erase.get('n_erased')} == len(slot prompt) "
                  f"- replay_boundary_tokens = "
                  f"{len(history) + len(g0) + len(suffix) + len(g1)} - "
                  f"{boundary} (history {history_n}+G0 {len(g0)}+suffix "
                  f"{suffix_n}+G1 {len(g1)}; bookkeeping holds the final "
                  f"processing pass minus the replay boundary token "
                  f"[TAG_PROMPT_LOGITS] server-context.cpp:3641-3644; "
                  f"cleared via prompt_clear, not RSS)")
        rep.check("erase.no_physical_prompt",
                  de["llamacpp:prompt_tokens_total"] == 0,
                  f"delta {de['llamacpp:prompt_tokens_total']}")
        request_deltas["erase_after_hot"] = de
        after_erase_slots = http(base, "GET", "/slots", timeout=30)
        report["observations"]["slots_after_erase"] = after_erase_slots
        live = next((s for s in after_erase_slots
                     if isinstance(s, dict) and s.get("id") == slot_id), None)
        rep.check("erase.slot_idle", live is not None
                  and not live.get("is_processing"),
                  f"slot {slot_id} is_processing="
                  f"{None if live is None else live.get('is_processing')}")

        # 5. cold probe: RAM cache must not rehydrate the erased prefix --------
        if variant.get("cold_probe"):
            probe_payload = dict(continuation_payload,
                                 prompt=saved_state,
                                 n_predict=int(variant["cold_probe_n_predict"]))
            probe, dp = complete("cold_probe_after_erase", probe_payload)
            request_deltas["cold_probe_after_erase"] = dp
            pt, ct = (probe.get("timings") or {}).get("prompt_n"), \
                (probe.get("timings") or {}).get("cache_n")
            rep.check("ram_cache.disabled_no_false_warm_hit",
                      ct == 0 and pt == len(saved_state),
                      f"after erase, cold prefill of saved state: cache_n {ct} "
                      f"== 0 and prompt_n {pt} == len(saved) "
                      f"{len(saved_state)} (full physical re-eval; "
                      f"--cache-ram 0, no RAM rehydration)")

        # 6. restore -----------------------------------------------------------
        restore, dr = op("restore", "POST",
                         f"/slots/{slot_id}?action=restore", {"filename": snapshot_file})
        rep.check("restore.roundtrip_counts",
                  restore.get("n_restored") == save.get("n_saved")
                  and restore.get("n_read") == save.get("n_written"),
                  f"n_restored {restore.get('n_restored')} == n_saved "
                  f"{save.get('n_saved')}, n_read {restore.get('n_read')} == "
                  f"n_written {save.get('n_written')}")
        rep.check("restore.no_physical_prompt",
                  dr["llamacpp:prompt_tokens_total"] == 0,
                  f"delta {dr['llamacpp:prompt_tokens_total']} (state load only)")
        request_deltas["restore"] = dr

        # 7. RESTORED arm: identical continuation, identical args --------------
        restored, drest = complete("restored_continuation", continuation_payload)
        request_deltas["restored_continuation"] = drest
        g2 = generated_ids(restored)
        report["observations"]["restored_continuation"]["generated_tokens"] = g2
        completion_bounds("restored", restored, drest)

        # 8. comparison --------------------------------------------------------
        hot_t = hot.get("timings") or {}
        res_t = restored.get("timings") or {}

        def first_probs(resp: dict):
            probs = resp.get("completion_probabilities")
            if not isinstance(probs, list) or not probs:
                return None, None
            first = probs[0] or {}
            tops = first.get("top_logprobs")
            top1 = None
            if isinstance(tops, list) and tops:
                top1 = tops[0] if isinstance(tops[0], dict) else None
            return first, top1

        def max_logprob_diff(a: dict, b: dict):
            """Max |d logprob| over every generated position (both arms emit
            the same token count when deterministic equality holds)."""
            pa = a.get("completion_probabilities")
            pb = b.get("completion_probabilities")
            if not isinstance(pa, list) or not isinstance(pb, list) \
                    or not pa or len(pa) != len(pb):
                return None
            diffs = []
            for x, y in zip(pa, pb):
                try:
                    diffs.append(abs(float(x["logprob"]) - float(y["logprob"])))
                except (KeyError, TypeError, ValueError):
                    return None
            return max(diffs)

        hot_p, hot_top1 = first_probs(hot)
        res_p, res_top1 = first_probs(restored)
        logprob_diff = max_logprob_diff(hot, restored)

        comparison = {
            "identical_request": (
                report["observations"]["hot_continuation"].get("request")
                == report["observations"]["restored_continuation"].get("request")),
            "continuation_starts_at_position": len(saved_state),
            "prompt_n": {"hot": hot_t.get("prompt_n"),
                         "restored": res_t.get("prompt_n"),
                         "equal": hot_t.get("prompt_n")
                         == res_t.get("prompt_n")},
            "cache_n": {"hot": hot_t.get("cache_n"),
                        "restored": res_t.get("cache_n")},
            "generated_tokens": {"hot": g1, "restored": g2, "equal": g1 == g2},
            "logits_accessible": hot_p is not None and res_p is not None,
            "logprob_abs_diff_max": logprob_diff,
            "first_token_top_logprobs": {
                "hot": (hot_p or {}).get("top_logprobs"),
                "restored": (res_p or {}).get("top_logprobs"),
                "hot_top1": hot_top1, "restored_top1": res_top1,
            },
        }
        report["comparison"] = comparison

        rep.check("compare.request_identical_hot_vs_restored",
                  report["observations"]["hot_continuation"]["request"]
                  == report["observations"]["restored_continuation"]["request"],
                  "both arms POST the same JSON under identical state args")
        rep.check("compare.prompt_n_parity",
                  comparison["prompt_n"]["equal"],
                  f"restored prompt_n {comparison['prompt_n']['restored']} == "
                  f"hot prompt_n {comparison['prompt_n']['hot']} "
                  f"(pre-patch full re-prefill breaks this parity)")
        rep.check("compare.generated_tokens_equal",
                  comparison["generated_tokens"]["equal"],
                  f"hot {g1[:16]}... vs restored {g2[:16]}... "
                  f"(temperature 0, seed 42)")
        if comparison["logits_accessible"]:
            tol = float(det["logprob_abs_tol"])
            rep.check("compare.logprob_within_tol",
                      logprob_diff is not None and logprob_diff <= tol,
                      f"max |d logprob| over all generated positions "
                      f"{logprob_diff} <= tol {tol}")
        else:
            rep.check("compare.logprob_accessibility_fallback", True,
                      "completion_probabilities not returned by this build; "
                      "token-id equality asserted, logits recorded as "
                      "unavailable")

        # 9. cumulative global counters ---------------------------------------
        final = scrape_counters(base, timeout)
        total_prompt = final["llamacpp:prompt_tokens_total"] \
            - baseline["llamacpp:prompt_tokens_total"]
        total_cached = final["llamacpp:prompt_tokens_cached_total"] \
            - baseline["llamacpp:prompt_tokens_cached_total"]
        sum_prompt = sum(d["llamacpp:prompt_tokens_total"]
                         for d in request_deltas.values())
        sum_cached = sum(d["llamacpp:prompt_tokens_cached_total"]
                         for d in request_deltas.values())
        report["metrics"] = {
            "baseline": baseline,
            "final": final,
            "global_prompt_tokens_delta": total_prompt,
            "global_prompt_tokens_cached_delta": total_cached,
            "sum_per_request_prompt_n": sum_prompt,
            "sum_per_request_cache_n": sum_cached,
            "per_request_deltas": {k: v for k, v in request_deltas.items()},
        }
        rep.check("metrics.cumulative_physical",
                  total_prompt == sum_prompt,
                  f"llamacpp:prompt_tokens_total delta {total_prompt} == "
                  f"sum of per-request prompt_n {sum_prompt} (physical)")
        rep.check("metrics.cumulative_cached",
                  total_cached == sum_cached,
                  f"llamacpp:prompt_tokens_cached_total delta {total_cached} == "
                  f"sum of per-request cache_n {sum_cached}")

        # 10. final erase: leave the slot clean --------------------------------
        erase_final, dfin = op("erase_final", "POST",
                               f"/slots/{slot_id}?action=erase")
        expected_final = (len(history) + len(g0) + len(suffix) + len(g2)
                          - boundary)
        rep.check("erase_final.n_erased_exact",
                  erase_final.get("n_erased") == expected_final,
                  f"n_erased {erase_final.get('n_erased')} == "
                  f"{len(history) + len(g0) + len(suffix) + len(g2)} - "
                  f"{boundary} (final pass minus replay boundary token, "
                  f"server-context.cpp:3641-3644)")
        rep.check("erase_final.no_physical_prompt",
                  dfin["llamacpp:prompt_tokens_total"] == 0,
                  f"delta {dfin['llamacpp:prompt_tokens_total']}")
        request_deltas["erase_final"] = dfin
        report["metrics"]["per_request_deltas"] = dict(request_deltas)

        # 11. hot-state-freed semantics (no RSS claim) -------------------------
        report["observations"]["hot_state_freed_evidence"] = {
            "sequence_cells_reusable": (
                "cold probe re-populated the erased slot with full physical "
                "prefill, then restore succeeded over it — sequence-cache "
                "ownership returned to the pool"),
            "rss_claimed": False,
            "rss_note": (
                "preallocated KV pool RSS need not fall after erase; freeing "
                "hot state is observable as cell/ownership reuse, not as a "
                "resident-size drop"),
        }

        failed = [c for c in rep.checks if not c["ok"]]
        report["summary"] = {"total": len(rep.checks),
                             "failed_count": len(failed),
                             "failed": [c["name"] for c in failed],
                             "ok": rep.ok}
        report["assertions"] = rep.checks
        return report, 0 if rep.ok else 1
    except HarnessError as exc:
        report["error"] = str(exc)
        failed = [c for c in rep.checks if not c["ok"]]
        report["summary"] = {"total": len(rep.checks),
                             "failed_count": len(failed),
                             "failed": [c["name"] for c in failed],
                             "ok": False,
                             "error": str(exc)}
        report["assertions"] = rep.checks
        return report, 2


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def write_report(report: dict, path: str | None) -> str:
    text = json.dumps(report, indent=2, default=str)
    if path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
        return str(dest)
    print(text)
    return "<stdout>"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="state_restore_harness",
        description="Externally-managed-server NVMe/SWA save/erase/restore "
                    "continuation harness (stdlib only).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser(
        "fetch-fixture",
        help="pinned size-capped sha256-verified fixture download")
    p_fetch.add_argument("--config", default=None)
    p_fetch.add_argument("--force", action="store_true",
                         help="re-download even if verified")

    p_run = sub.add_parser(
        "run", help="hot vs save/erase/restore continuation comparison")
    p_run.add_argument("--base-url", default=None,
                       help="externally managed server (or env "
                            "NVME_CONTINUATION_BASE_URL)")
    p_run.add_argument("--variant", default="small",
                       help="config variant: small (default) | slow "
                            "(100K+4K)")
    p_run.add_argument("--config", default=None)
    p_run.add_argument("--report", default=None,
                       help="report JSON path (default: stdout)")

    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except HarnessError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2

    if args.command == "fetch-fixture":
        try:
            result = fetch_fixture(cfg, args.force)
        except HarnessError as exc:
            print(f"fetch-fixture: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") == "verified" else 2

    base = args.base_url or os.environ.get("NVME_CONTINUATION_BASE_URL")
    if not base:
        print("run: --base-url (or NVME_CONTINUATION_BASE_URL) is required; "
              "this harness never spawns a server", file=sys.stderr)
        return 2

    report = None
    rc = 2
    try:
        report, rc = run(base, cfg, args.variant)
    except HarnessError as exc:
        report = {
            "schema_version": 1,
            "variant": args.variant,
            "base_url": base,
            "error": str(exc),
            "summary": {"ok": False, "error": str(exc)},
            "assertions": [],
        }
        print(f"run: {exc}", file=sys.stderr)
        rc = 2
    dest = write_report(report, args.report)
    summary = report.get("summary", {})
    print(f"report -> {dest}; ok={summary.get('ok')} "
          f"failed={summary.get('failed_count', 'n/a')} "
          f"error={report.get('error')}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
