"""Synthetic structural tests for the extractor-side taxonomy labeler.

Framework: stdlib unittest (run by `python3 -m unittest tests.test_labeling`;
no pytest, no network, no model calls). Every fixture is hand-built from
NormalizedEvent values shaped exactly like the parsers emit them — hints come
from events.inspect_tool_arguments, results carry bounded text plus full
length/hash metadata — so no real trace content, raw payload or local path
appears in this file.

Coverage (docs/datasets.md "Taxonomies", "Tagging rules and boundary cases",
"Downweight classes"; schema contract schemas/episode.schema.json):

- boundary cases: debugging vs test_interpretation, compiler_interpretation
  vs debugging, verification vs test_interpretation, tool_use vs shell
- primary-tag rule: action/no-action selection, lexicographic tiebreak,
  long_context never primary, recovery-only primary, omission when empty
- languages: extensions, syntax markers, `other` for out-of-set subject code,
  generated/non-subject/bare-mention omission
- difficulty: the D0..D4 rule table and monotonicity (more span/tools/
  recovery/verification never lowers the level)
- downweight_classes: per-event attribution with count == downweighted_events,
  plus the omit-rather-than-guess path
- schema conformance of a fully labeled document (the repo's minimal JSON
  Schema subset validator from tests.test_build) and closed-enum equality
  against schemas/episode.schema.json (22 capabilities / 9 languages /
  5 difficulty levels / 5 downweight classes)
- determinism: two independent segment+label runs serialize byte-identically;
  relabeling the same episodes is idempotent
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from mimo_halo.traces.common import sha256_text  # noqa: E402
from mimo_halo.traces.events import (  # noqa: E402
    Episode,
    NormalizedEvent,
    SessionTrace,
    inspect_tool_arguments,
)
from mimo_halo.traces.labeling import (  # noqa: E402
    CAPABILITIES,
    DIFFICULTY_LEVELS,
    DOWNWEIGHT_CLASSES,
    LANGUAGES,
    choose_primary,
    detect_languages,
    episode_facts,
    label_episodes,
)
from mimo_halo.traces.session import segment_episodes  # noqa: E402
from tests.test_build import assert_schema  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_SCHEMA = json.loads((_REPO / "schemas" / "episode.schema.json").read_text())
_MAX_TEXT = 8192  # events.MAX_TEXT_CHARS: text bounded, length/hash full


# ---------------------------------------------------------------- fixtures

def _msg(seq: int, typ: str, text: str) -> NormalizedEvent:
    return NormalizedEvent(seq=seq, type=typ, text=text)


def _call(seq: int, name: str, args, cid: str, **over) -> NormalizedEvent:
    digest, chars, verif, spam = inspect_tool_arguments(name, args)
    ev = NormalizedEvent(
        seq=seq, type="tool_call", tool_call_id=cid, tool_name=name,
        arguments_sha256=digest, arguments_chars=chars,
        verification_hint=verif, spam_hint=spam, tool_kind="call",
    )
    for key, value in over.items():
        setattr(ev, key, value)
    return ev


def _result(seq: int, cid: str, text: str, error: bool = False,
            name: str | None = None) -> NormalizedEvent:
    # bounded text, full-length/hash metadata — exactly what the pipeline keeps
    return NormalizedEvent(
        seq=seq, type="tool_result", tool_call_id=cid, tool_name=name,
        text=text[:_MAX_TEXT], result_is_error=error,
        raw_text_sha256=sha256_text(text), raw_text_chars=len(text),
        tool_kind="result",
    )


def _trace(events: list[NormalizedEvent]) -> SessionTrace:
    trace = SessionTrace(source="synthetic", session_id="s-lbl",
                         source_path_sha256="f" * 64)
    trace.events = list(events)
    return trace


def _label(events: list[NormalizedEvent]) -> tuple[SessionTrace, list[Episode]]:
    trace = _trace(events)
    return trace, label_episodes(trace, segment_episodes(trace))


def _debug_events() -> list[NormalizedEvent]:
    """Failing test located a defect; a later edit fixed it (recovery too)."""
    return [
        _msg(0, "user", "the parser drops retries"),
        _msg(1, "assistant", "hypothesis: the retry counter resets mid-parse"),
        _call(2, "Bash", {"command": "pytest -q tests/test_parser.py"}, "c1"),
        _result(3, "c1", "FAILED tests/test_parser.py::test_parse - assert 2 == 3",
                error=True, name="Bash"),
        _call(4, "Edit", {"file_path": "src/parser.py", "old_string": "a",
                          "new_string": "b"}, "c2"),
        _result(5, "c2", "applied", name="Edit"),
        _call(6, "Bash", {"command": "pytest -q tests/test_parser.py"}, "c3"),
        _result(7, "c3", "3 passed", name="Bash"),
    ]


def _compiler_events() -> list[NormalizedEvent]:
    return [
        _msg(0, "user", "the build broke"),
        _msg(1, "assistant", "the diagnostic itself names the cause"),
        _call(2, "Bash", {"command": "cargo check"}, "c1"),
        _result(3, "c1",
                "error[E0308]: mismatched types\nexpected `i32`, found `&str`",
                error=True, name="Bash"),
        _call(4, "Edit", {"file_path": "src/lib.rs", "old_string": "x",
                          "new_string": "y"}, "c2"),
        _result(5, "c2", "applied", name="Edit"),
    ]


def _confirm_events() -> list[NormalizedEvent]:
    """A passing suite run that only confirms an already-made change."""
    return [
        _msg(0, "user", "confirm the fix still holds"),
        _msg(1, "assistant", "re-running the suite to confirm"),
        _call(2, "Bash", {"command": "pytest -q"}, "c1"),
        _result(3, "c1", "3 passed", name="Bash"),
    ]


def _infer_events() -> list[NormalizedEvent]:
    """A failing suite run read to infer expected behavior (no fix yet)."""
    return [
        _msg(0, "user", "what should this function return?"),
        _msg(1, "assistant", "reading the failing test to infer intent"),
        _call(2, "Bash", {"command": "pytest -q tests/test_seam.py"}, "c1"),
        _result(3, "c1", "assert 2 == 3\nFAILED tests/test_seam.py::test_seam",
                error=True, name="Bash"),
    ]


def _shell_pipeline_events() -> list[NormalizedEvent]:
    """A grep|sort|uniq pipeline through the shell is shell, not tool_use."""
    return [
        _msg(0, "user", "dedupe the report"),
        _msg(1, "assistant", "one shell pipeline does it"),
        _call(2, "Bash", {"command": "grep -rn TODO src | sort | uniq -c"}, "c1"),
        _result(3, "c1", "42 src/a.py: TODO", name="Bash"),
    ]


def _mcp_events() -> list[NormalizedEvent]:
    """An MCP tool invocation is tool_use, never shell (no map substring hits)."""
    return [
        _msg(0, "user", "file the report"),
        _msg(1, "assistant", "delegating to the MCP server"),
        _call(2, "mcp__wiki__ask", {"query": "status"}, "c1"),
        _result(3, "c1", "filed", name="mcp__wiki__ask"),
    ]


def _rich_events() -> list[NormalizedEvent]:
    """One turn exercising lockfile downweight + every boundary flavor."""
    return [
        _msg(0, "user", "the parser drops retries, deps moved under it"),
        _msg(1, "assistant", "hypothesis: retry counter resets; check src/parser.py"),
        _call(2, "Grep", {"pattern": "lockfileVersion", "path": "."}, "c1"),
        _result(3, "c1", "lockfileVersion: 3\npackages:\n  left-pad: 1.0.0",
                name="Grep"),
        _call(4, "Bash", {"command": "pytest -q tests/test_parser.py"}, "c2"),
        _result(5, "c2", "FAILED tests/test_parser.py::test_parse - assert 2 == 3",
                error=True, name="Bash"),
        _call(6, "Edit", {"file_path": "src/parser.py", "old_string": "a",
                          "new_string": "b"}, "c3"),
        _result(7, "c3", "applied", name="Edit"),
        _call(8, "Bash", {"command": "pytest -q tests/test_parser.py"}, "c4"),
        _result(9, "c4", "3 passed", name="Bash"),
    ]


def _labels(ep: Episode) -> dict:
    """Labelled fields of an episode as they serialize (present ones only)."""
    doc = ep.to_public()
    return {k: doc[k] for k in
            ("primary_capability", "languages", "difficulty", "downweight_classes")
            if k in doc}


# ------------------------------------------------- docs boundary cases

class BoundaryCaseTests(unittest.TestCase):
    def test_debugging_primary_with_test_interpretation_secondary(self):
        _, eps = _label(_debug_events())
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        self.assertEqual(ep.primary_capability, "debugging")
        # using a failing test to locate+fix keeps test_interpretation
        # as a secondary tag; the unnamed cause rules out compiler_interpretation
        self.assertIn("test_interpretation", ep.capabilities)
        self.assertNotIn("compiler_interpretation", ep.capabilities)
        # recovery rides along as the secondary tag, never the primary here
        self.assertIn("recovery", ep.capabilities)
        self.assertNotEqual(ep.primary_capability, "recovery")
        self.assertTrue(ep.recovered)

    def test_compiler_diagnostic_beats_debugging(self):
        _, eps = _label(_compiler_events())
        ep = eps[0]
        # the message itself resolved the cause: no hypothesis formed
        self.assertEqual(ep.primary_capability, "compiler_interpretation")
        self.assertNotIn("debugging", ep.capabilities)

    def test_confirm_run_is_verification_not_test_interpretation(self):
        _, eps = _label(_confirm_events())
        ep = eps[0]
        self.assertEqual(ep.primary_capability, "verification")
        # confirming a change never implies reading tests for intent
        self.assertNotIn("test_interpretation", ep.capabilities)

    def test_failing_test_read_is_test_interpretation_not_verification(self):
        _, eps = _label(_infer_events())
        ep = eps[0]
        self.assertEqual(ep.primary_capability, "test_interpretation")
        self.assertNotEqual(ep.primary_capability, "verification")

    def test_shell_pipeline_is_shell_not_tool_use(self):
        _, eps = _label(_shell_pipeline_events())
        ep = eps[0]
        self.assertEqual(ep.primary_capability, "shell")
        self.assertNotIn("tool_use", ep.capabilities)

    def test_mcp_tool_is_tool_use_not_shell(self):
        _, eps = _label(_mcp_events())
        ep = eps[0]
        self.assertEqual(ep.primary_capability, "tool_use")
        self.assertNotIn("shell", ep.capabilities)


# ------------------------------------------------- primary-tag rule

class PrimaryTagRuleTests(unittest.TestCase):
    def test_action_without_error_context_is_implementation(self):
        events = [
            _msg(0, "user", "add the flag"),
            _msg(1, "assistant", "decision: where the flag threads through"),
            _call(2, "Edit", {"file_path": "a.py", "old_string": "x",
                              "new_string": "y"}, "c1"),
            _result(3, "c1", "applied", name="Edit"),
        ]
        _, eps = _label(events)
        self.assertEqual(eps[0].primary_capability, "implementation")

    def test_lexicographic_tiebreak_is_deterministic(self):
        facts = episode_facts([])
        self.assertEqual(choose_primary(["git", "database"], facts), "database")
        self.assertEqual(choose_primary(["git", "database"], facts),
                         choose_primary(["database", "git"], facts))

    def test_long_context_is_never_primary(self):
        facts = episode_facts([])
        self.assertIsNone(choose_primary(["long_context"], facts))
        self.assertEqual(choose_primary(["long_context", "git"], facts), "git")

    def test_recovery_primary_only_when_restoring_is_the_point(self):
        facts = episode_facts([
            _call(2, "Bash", {"command": "git status"}, "c1"),
            _result(3, "c1", "dirty", error=True, name="Bash"),
        ])
        # only vehicle tags beside recovery -> recovery may carry the turn
        self.assertEqual(choose_primary(["recovery", "shell"], facts), facts
                         and choose_primary(["recovery", "shell"], facts))
        self.assertEqual(choose_primary(["recovery", "shell"], facts), "recovery")
        # a decision-bearing capability beside it keeps recovery secondary
        self.assertEqual(choose_primary(["recovery", "implementation"], facts),
                         "implementation")

    def test_omitted_when_no_capability_evidence(self):
        events = [
            _msg(0, "user", "hello"),
            _msg(1, "assistant", "just thinking, no tools touched"),
        ]
        _, eps = _label(events)
        ep = eps[0]
        self.assertIsNone(ep.primary_capability)
        doc = ep.to_public()
        self.assertNotIn("primary_capability", doc)  # omit-null, never guessed

    def test_language_omitted_without_code_evidence(self):
        events = [
            _msg(0, "user", "what changed?"),
            _msg(1, "assistant", "reading the changelog and design notes"),
        ]
        _, eps = _label(events)
        self.assertIsNone(eps[0].languages)
        self.assertNotIn("languages", eps[0].to_public())


# ------------------------------------------------- languages

class LanguageDetectionTests(unittest.TestCase):
    def _langs(self, *texts: str):
        events = [_msg(i, "assistant", t) for i, t in enumerate(texts)]
        return detect_languages(events)

    def test_named_extensions(self):
        self.assertEqual(self._langs("review src/components/App.tsx"),
                         ["typescript"])
        self.assertEqual(self._langs("run scripts/deploy.sh"), ["shell"])
        self.assertEqual(self._langs("query analytics/events.sql"), ["sql"])

    def test_syntax_markers_without_extension(self):
        self.assertEqual(self._langs("def parse(line):\n    return line"),
                         ["python"])
        self.assertEqual(self._langs("package main\nfunc main() {}"), ["go"])
        self.assertEqual(self._langs("auto v = std::vector<int>{};"), ["cpp"])
        self.assertEqual(self._langs("#!/usr/bin/env bash"), ["shell"])
        self.assertEqual(self._langs("#!/bin/sh"), ["shell"])

    def test_multi_language_ordered_by_closed_set(self):
        self.assertEqual(self._langs("port main.ts", "then run.py"),
                         ["typescript", "python"])

    def test_out_of_set_subject_code_is_other(self):
        self.assertEqual(self._langs("app/models/user.rb"), ["other"])
        self.assertEqual(self._langs("src/buf.h"), ["other"])  # C alone -> other
        # .h beside .hpp reads as the same C++ subject, not other
        self.assertEqual(self._langs("src/buf.h", "include/v2.hpp"), ["cpp"])
        self.assertEqual(self._langs("#include <vector>"), ["other"])

    def test_omitted_for_bare_language_names_and_non_subject_formats(self):
        # a language name is a mention, never subject evidence
        self.assertIsNone(self._langs("switch the rust toolchain next"))
        # json/markdown are non-subject formats (never in the code sets)
        self.assertIsNone(self._langs("read changelog.md and package.json"))
        # generated artifacts are opaque, not subject evidence
        self.assertIsNone(self._langs("serve app.min.js output"))


# ------------------------------------------------- difficulty

class DifficultyRuleTests(unittest.TestCase):
    @staticmethod
    def _ep(span=4, tools=0, decision=False, recovered=False, verif=0,
            caps=()) -> Episode:
        return Episode(
            episode_id="e", turn_index=0, sidechain=False,
            started_at=None, ended_at=None, event_seq=list(range(span)),
            tool_calls=tools, has_decision=decision,
            verification_events=verif, recovered=recovered,
            downweighted_events=0, capabilities=list(caps), weight=1.0,
        )

    def test_rule_table_levels(self):
        from mimo_halo.traces.labeling import assign_difficulty
        table = [
            # (episode, expected level, rule row)
            (self._ep(), "D0", "otherwise (mechanical / no decision)"),
            (self._ep(decision=True), "D1", "has_decision"),
            (self._ep(span=16, tools=4, decision=True), "D2",
             "has_decision and (span >= 16 or tool_calls >= 8)"),
            (self._ep(span=20, tools=6, decision=True), "D2",
             "has_decision and tool_calls >= 8"),
            (self._ep(decision=True, recovered=True), "D3",
             "recovered or long_context"),
            (self._ep(caps=["long_context"]), "D3",
             "recovered or long_context"),
            (self._ep(span=45, tools=9, decision=True, recovered=True, verif=1,
                      caps=["long_context"]), "D4",
             "long_context and recovered and verification_events >= 1"),
        ]
        for ep, expected, rule in table:
            with self.subTest(rule=rule):
                self.assertEqual(assign_difficulty(ep), expected)

    def test_monotonicity_span_never_lowers(self):
        from mimo_halo.traces.labeling import assign_difficulty
        levels = [assign_difficulty(self._ep(span=n, tools=2, decision=True))
                  for n in range(0, 65)]
        index = {lv: i for i, lv in enumerate(DIFFICULTY_LEVELS)}
        ranks = [index[lv] for lv in levels]
        self.assertEqual(ranks, sorted(ranks))  # non-decreasing in span

    def test_monotonicity_added_features_never_lower(self):
        from mimo_halo.traces.labeling import assign_difficulty
        from mimo_halo.traces.labeling import _DIFFICULTY_SPAN
        base = assign_difficulty(self._ep(decision=True))
        more_tools = assign_difficulty(self._ep(decision=True, tools=8))
        with_recovery = assign_difficulty(
            self._ep(span=_DIFFICULTY_SPAN, tools=8, decision=True,
                     recovered=True))
        long_context = self._ep(span=45, tools=8, decision=True,
                                recovered=True)
        long_context.capabilities = ["long_context"]
        with_context = assign_difficulty(long_context)
        verified = assign_difficulty(
            self._ep(span=45, tools=8, decision=True, recovered=True,
                     verif=1, caps=["long_context"]))
        chain = [base, more_tools, with_recovery, with_context, verified]
        index = {lv: i for i, lv in enumerate(DIFFICULTY_LEVELS)}
        ranks = [index[lv] for lv in chain]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(ranks[-1], 4)  # the full feature set lands on D4


# ------------------------------------------------- downweight classes

class DownweightClassTests(unittest.TestCase):
    def test_lockfile_read_class_and_count(self):
        _, eps = _label([
            _msg(0, "user", "deps moved"),
            _msg(1, "assistant", "reading the lock artifact"),
            _call(2, "Grep", {"pattern": "lockfileVersion"}, "c1"),
            _result(3, "c1", "lockfileVersion: 3\npackages:", name="Grep"),
        ])
        ep = eps[0]
        self.assertEqual(ep.downweighted_events, 1)  # the spam-flagged call
        self.assertEqual(ep.downweight_classes, ["lockfile"])
        self.assertEqual(len(ep.downweight_classes), ep.downweighted_events)

    def test_repeated_build_runs_each_counted(self):
        _, eps = _label([
            _msg(0, "user", "build it"),
            _msg(1, "assistant", "rebuilding until green"),
            _call(2, "Bash", {"command": "cargo build --release"}, "c1"),
            _result(3, "c1", "Compiling...", name="Bash"),
            _call(4, "Bash", {"command": "cargo build --release"}, "c2"),
            _result(5, "c2", "Compiling...", name="Bash"),
        ])
        ep = eps[0]
        self.assertEqual(ep.downweighted_events, 2)
        self.assertEqual(ep.downweight_classes,
                         ["repeated_build_spam", "repeated_build_spam"])
        self.assertEqual(len(ep.downweight_classes), ep.downweighted_events)

    def test_enormous_grep_then_duplicate_content(self):
        huge = "rg hit line\n" * 1600  # 17600 chars: over the 16384 bound
        _, eps = _label([
            _msg(0, "user", "search everything"),
            _msg(1, "assistant", "two identical oversized searches"),
            _call(2, "Grep", {"pattern": "TODO"}, "c1"),
            _result(3, "c1", huge, name="Grep"),
            _call(4, "Grep", {"pattern": "TODO"}, "c2"),
            _result(5, "c2", huge, name="Grep"),  # same content hash
        ])
        ep = eps[0]
        self.assertEqual(ep.downweighted_events, 4)
        self.assertEqual(ep.downweight_classes, [
            "enormous_grep_output", "enormous_grep_output",
            "duplicate_file_content", "duplicate_file_content",
        ])
        self.assertEqual(len(ep.downweight_classes), ep.downweighted_events)

    def test_generated_bundle_result_class(self):
        bundle = ("console.log(1);\n//# sourceMappingURL=app.min.js\n"
                  + "y\n" * 8200)  # over the 16384 bound; marker names the artifact
        _, eps = _label([
            _msg(0, "user", "ship the bundle"),
            _msg(1, "assistant", "reading generated output"),
            _call(2, "Read", {"file_path": "dist/app.min.js"}, "c1"),
            _result(3, "c1", bundle, name="Read"),
        ])
        ep = eps[0]
        self.assertEqual(ep.downweighted_events, 2)
        self.assertEqual(ep.downweight_classes,
                         ["generated_bundle", "generated_bundle"])
        self.assertEqual(len(ep.downweight_classes), ep.downweighted_events)

    def test_unattributable_spam_is_omitted_not_guessed(self):
        _, eps = _label([
            _msg(0, "user", "look around"),
            _msg(1, "assistant", "a plain read with no class evidence"),
            _call(2, "Read", {"file_path": "src/app.py"}, "c1"),
            _result(3, "c1", "def main():\n    pass", name="Read"),
        ])
        ep = eps[0]
        # the read is spam-flagged, but none of the five classes is observable
        self.assertEqual(ep.downweighted_events, 1)
        self.assertIsNone(ep.downweight_classes)
        self.assertNotIn("downweight_classes", ep.to_public())

    def test_count_invariant_holds_across_labeled_fixtures(self):
        fixtures = [_debug_events(), _compiler_events(), _confirm_events(),
                    _infer_events(), _shell_pipeline_events(), _mcp_events(),
                    _rich_events()]
        for events in fixtures:
            with self.subTest(first_seq=events[0].seq):
                _, eps = _label(events)
                for ep in eps:
                    if ep.downweight_classes is not None:
                        self.assertEqual(len(ep.downweight_classes),
                                         ep.downweighted_events)
                        self.assertTrue(set(ep.downweight_classes)
                                        <= set(DOWNWEIGHT_CLASSES))


# ------------------------------------------------- schema + closed enums

class SchemaConformanceTests(unittest.TestCase):
    def test_closed_enums_match_schema_exactly(self):
        ep_schema = _SCHEMA["properties"]["episodes"]["items"]
        props = ep_schema["properties"]
        self.assertEqual(props["primary_capability"]["enum"], list(CAPABILITIES))
        self.assertEqual(props["capabilities"]["items"]["enum"], list(CAPABILITIES))
        self.assertEqual(props["languages"]["items"]["enum"], list(LANGUAGES))
        self.assertEqual(props["difficulty"]["enum"], list(DIFFICULTY_LEVELS))
        self.assertEqual(_SCHEMA["$defs"]["downweight_class"]["enum"],
                         list(DOWNWEIGHT_CLASSES))
        # quoted sizes of the closed sets: 22 / 9 / 5 / 5
        self.assertEqual(
            (len(CAPABILITIES), len(LANGUAGES), len(DIFFICULTY_LEVELS),
             len(DOWNWEIGHT_CLASSES)), (22, 9, 5, 5))

    def test_fully_labeled_document_validates_against_schema(self):
        trace, eps = _label(_rich_events())
        ep = eps[0]
        # fully labeled: every optional label carries a value
        self.assertIn("primary_capability", _labels(ep))
        self.assertIn("languages", _labels(ep))
        self.assertIn("difficulty", _labels(ep))
        self.assertIn("downweight_classes", _labels(ep))
        calls = sum(1 for e in trace.events if e.type == "tool_call")
        results = sum(1 for e in trace.events if e.type == "tool_result")
        doc = {
            "schema_version": "1.0.0",
            "kind": "mimo-halo-normalized-trace",
            "session": {
                "source": "pi",
                "session_id": "s-lbl",
                "source_path_sha256": "f" * 64,
                "event_count": len(trace.events),
                "tool_call_count": calls,
                "tool_result_count": results,
                "unmatched_tool_calls": 0,
                "unmatched_tool_results": 0,
                "has_branching": False,
                "parent_session_ids": [],
                "models": [],
            },
            "events": [e.to_public(include_text=False) for e in trace.events],
            "episodes": [ep.to_public() for ep in eps],
        }
        assert_schema(doc, _SCHEMA)

    def test_emitted_labels_are_members_of_their_enums(self):
        _, eps = _label(_rich_events())
        ep = eps[0]
        labels = _labels(ep)
        self.assertIn(labels["primary_capability"], CAPABILITIES)
        self.assertTrue(set(labels["languages"]) <= set(LANGUAGES))
        self.assertIn(labels["difficulty"], DIFFICULTY_LEVELS)
        self.assertTrue(set(labels["downweight_classes"])
                        <= set(DOWNWEIGHT_CLASSES))
        self.assertIn(labels["primary_capability"], ep.capabilities)


# ------------------------------------------------- determinism

class DeterminismTests(unittest.TestCase):
    def test_two_independent_runs_serialize_byte_identically(self):
        def run() -> str:
            _, eps = _label(_rich_events())
            return json.dumps([ep.to_public() for ep in eps],
                              sort_keys=True, separators=(",", ":"))
        first, second = run(), run()
        self.assertEqual(first, second)
        self.assertEqual(sha256_text(first), sha256_text(second))

    def test_relabeling_the_same_episodes_is_idempotent(self):
        trace, eps = _label(_rich_events())
        before = json.dumps([ep.to_public() for ep in eps], sort_keys=True)
        label_episodes(trace, eps)
        after = json.dumps([ep.to_public() for ep in eps], sort_keys=True)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
