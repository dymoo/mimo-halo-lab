"""Black-box parity tests for source-driven NFC tokenizer metadata.

The tests intentionally exercise the pinned converter and the built
``llama-tokenize`` executable through temporary, vocab-only GGUF fixtures.  The
source tokenizer is copied from ``MIMO_HALO_TOKENIZER_SRC`` or the checked-in
scratch fixture; neither source assets nor the upstream checkout are mutated.

The native command line is the supported seam:

    llama-tokenize --model <vocab.gguf> --threads 1 --no-bos --no-escape \
        --ids --stdin --offline

Special-token parsing is left at its tokenizer-tool default (enabled).  Do not
add ``--parse-special``: this pinned CLI rejects that spelling.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
STRIX = REPO_ROOT / "upstreams" / "strix-llama.cpp"
CONVERTER = STRIX / "convert_hf_to_gguf.py"
DEFAULT_TOKENIZER_BIN = STRIX / "build-metal" / "bin" / "llama-tokenize"
TOKENIZER_BIN_ENV = "MIMO_HALO_TOKENIZER_BIN"
TOKENIZER_SRC_ENV = "MIMO_HALO_TOKENIZER_SRC"
FALLBACK_TOKENIZER_DIR = REPO_ROOT / "scratch" / "loadability" / "tiny_a"
NFC_KV = "tokenizer.ggml.normalizer.nfc"

# Keep these as semantic boundaries rather than a broad/generated corpus.  The
# reference tokenizer is the expected-value source; no token ids are frozen.
NFC_CASES = (
    ("canonical_mark_reordering", "a\u0315\u0300"),
    ("canonical_mark_blocking", "a\u0301\u0300"),
    ("canonical_chained_composition", "a\u030a\u0301"),
    ("hangul_jamo", "\u1100\u1161\u11a8"),
    ("composition_exclusion", "\u0915\u093c"),
    ("supplementary_unicode", "x\U0001f600\U0001d11ey"),
    ("embedded_nul", "e\u0301\x00z"),
    ("special_marker_adjacency", "<|im_start|>e\u0301<|im_end|>e\u0301"),
    ("ascii_and_control", "ASCII\tcontrol\n\x01"),
)

EQUIVALENT_SPELLINGS = (
    ("e_acute", "\u00e9", "e\u0301"),
    ("canonical_reordering", "\u00e0\u0315", "a\u0315\u0300"),
    ("canonical_blocking", "\u00e1\u0300", "a\u0301\u0300"),
    ("canonical_chained", "\u01fb", "a\u030a\u0301"),
    ("hangul", "\uac01", "\u1100\u1161\u11a8"),
)

# The no-opt-in fixture is deliberately exercised at the same boundaries that
# would otherwise be changed by normalization.  ASCII/control remains a useful
# unchanged-path witness too.
LEGACY_CASES = (
    ("decomposed_e_acute", "e\u0301"),
    ("hangul_jamo", "\u1100\u1161\u11a8"),
    ("embedded_nul", "e\u0301\x00z"),
    ("special_marker_adjacency", "<|im_start|>e\u0301<|im_end|>e\u0301"),
    ("ascii_and_control", "ASCII\tcontrol\n\x01"),
)


class TokenizerNfcTest(unittest.TestCase):
    """Converter -> GGUF metadata -> native CLI -> HF reference parity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_path: Path | None = None

        source = cls._find_tokenizer_source()
        if source is None:
            raise unittest.SkipTest(
                "source tokenizer assets absent: set MIMO_HALO_TOKENIZER_SRC "
                "to a local checkpoint directory or provide "
                "scratch/loadability/tiny_a"
            )
        cls.source = source

        missing_source = [
            name
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
            if not (source / name).is_file()
        ]
        if missing_source:
            message = (
                "source tokenizer prerequisite files absent in "
                f"{source}: {', '.join(missing_source)}"
            )
            if os.environ.get(TOKENIZER_SRC_ENV) is not None:
                raise AssertionError(f"{TOKENIZER_SRC_ENV} is invalid: {message}")
            raise unittest.SkipTest(message)

        if not CONVERTER.is_file() or not (STRIX / "conversion").is_dir() or not (
            STRIX / "gguf-py"
        ).is_dir():
            raise unittest.SkipTest(
                "pinned converter prerequisites absent: expected convert_hf_to_gguf.py, "
                "conversion/, and gguf-py/ under upstreams/strix-llama.cpp"
            )

        missing_converter_deps = [
            name
            for name in ("torch", "transformers")
            if importlib.util.find_spec(name) is None
        ]
        if missing_converter_deps:
            raise unittest.SkipTest(
                "pinned converter/reference Python prerequisites absent: install "
                + ", ".join(missing_converter_deps)
            )

        cls.tokenizer_bin = cls._find_tokenizer_binary()

        with (source / "tokenizer.json").open(encoding="utf-8") as fp:
            source_tokenizer = json.load(fp)
        normalizer = source_tokenizer.get("normalizer")
        if not isinstance(normalizer, dict) or normalizer.get("type") != "NFC":
            raise AssertionError(
                "the source fixture must declare the standalone tokenizer "
                f"normalizer {{'type': 'NFC'}}, got {normalizer!r}"
            )

        cls.tmp_path = Path(tempfile.mkdtemp(prefix="mimo-tokenizer-nfc-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp_path, ignore_errors=True)
        cls.converter_tree = cls.tmp_path / "converter"
        cls.nfc_source = cls.tmp_path / "source-nfc"
        cls.legacy_source = cls.tmp_path / "source-no-nfc"
        cls.nfc_gguf = cls.tmp_path / "nfc-vocab.gguf"
        cls.legacy_gguf = cls.tmp_path / "legacy-vocab.gguf"

        cls._copy_converter_tree(cls.converter_tree)
        cls._copy_source_fixture(source, cls.nfc_source)
        cls._copy_source_fixture(source, cls.legacy_source, remove_normalizer=True)

        cls._convert_vocab_only(cls.nfc_source, cls.nfc_gguf)
        cls._convert_vocab_only(cls.legacy_source, cls.legacy_gguf)

        # This is intentionally checked independently of token output: parity
        # must be source-driven, not an accidental global native normalization.
        cls.nfc_metadata = cls._read_nfc_metadata(cls.nfc_gguf)
        cls.legacy_metadata = cls._read_nfc_metadata(cls.legacy_gguf)

        cls.nfc_reference = cls._load_reference(cls.nfc_source)
        cls.legacy_reference = cls._load_reference(cls.legacy_source)


    @staticmethod
    def _find_tokenizer_source() -> Path | None:
        env_source = os.environ.get(TOKENIZER_SRC_ENV)
        if env_source is not None:
            candidate = Path(env_source)
            if not (candidate / "tokenizer.json").is_file():
                raise AssertionError(
                    f"{TOKENIZER_SRC_ENV} is explicitly set to {candidate}, "
                    "but tokenizer.json is missing"
                )
            return candidate

        if (FALLBACK_TOKENIZER_DIR / "tokenizer.json").is_file():
            return FALLBACK_TOKENIZER_DIR
        return None

    @classmethod
    def _find_tokenizer_binary(cls) -> Path:
        override = os.environ.get(TOKENIZER_BIN_ENV)
        if override is not None:
            candidate = Path(override)
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise AssertionError(
                    f"{TOKENIZER_BIN_ENV} points to a missing or non-executable "
                    f"binary: {candidate}"
                )
            return candidate

        if not DEFAULT_TOKENIZER_BIN.is_file() or not os.access(
            DEFAULT_TOKENIZER_BIN, os.X_OK
        ):
            raise unittest.SkipTest(
                "built tokenizer CLI absent: expected "
                f"{DEFAULT_TOKENIZER_BIN} or set {TOKENIZER_BIN_ENV}"
            )
        return DEFAULT_TOKENIZER_BIN

    @classmethod
    def _copy_converter_tree(cls, destination: Path) -> None:
        destination.mkdir(parents=True)
        shutil.copyfile(CONVERTER, destination / "convert_hf_to_gguf.py")
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(STRIX / "conversion", destination / "conversion", ignore=ignore)
        shutil.copytree(STRIX / "gguf-py", destination / "gguf-py", ignore=ignore)

    @staticmethod
    def _copy_source_fixture(
        source: Path, destination: Path, *, remove_normalizer: bool = False
    ) -> None:
        destination.mkdir(parents=True)
        # These are the tokenizer/config inputs only.  In particular, never
        # copy model.safetensors or alter the caller's source checkpoint.
        for name in (
            "config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
        ):
            src = source / name
            if src.is_file():
                shutil.copyfile(src, destination / name)

        tokenizer_json = source / "tokenizer.json"
        if remove_normalizer:
            with tokenizer_json.open(encoding="utf-8") as fp:
                document = json.load(fp)
            document.pop("normalizer", None)
            with (destination / "tokenizer.json").open("w", encoding="utf-8") as fp:
                json.dump(document, fp, ensure_ascii=False, separators=(",", ":"))
                fp.write("\n")
        else:
            shutil.copyfile(tokenizer_json, destination / "tokenizer.json")

    @classmethod
    def _convert_vocab_only(cls, model_dir: Path, output: Path) -> None:
        command = [
            sys.executable,
            str(cls.converter_tree / "convert_hf_to_gguf.py"),
            str(model_dir),
            "--outfile",
            str(output),
            "--outtype",
            "f16",
            "--vocab-only",
        ]
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if process.returncode != 0 or not output.is_file():
            raise AssertionError(
                "vocab-only conversion failed (source-driven NFC fixture): "
                f"exit={process.returncode}\n"
                f"stdout={process.stdout[-4000:]}\n"
                f"stderr={process.stderr[-4000:]}"
            )

    @classmethod
    def _read_nfc_metadata(cls, gguf_path: Path) -> bool | None:
        gguf_path_root = str(cls.converter_tree / "gguf-py")
        if gguf_path_root not in sys.path:
            sys.path.insert(0, gguf_path_root)
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf_path))
        field = reader.fields.get(NFC_KV)
        if field is None:
            return None
        return bool(field.contents())

    @classmethod
    def _read_tokenizer_model(cls, gguf_path: Path) -> str:
        gguf_py_root = str(cls.converter_tree / "gguf-py")
        if gguf_py_root not in sys.path:
            sys.path.insert(0, gguf_py_root)
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf_path))
        return str(reader.fields["tokenizer.ggml.model"].contents())

    @staticmethod
    def _load_reference(model_dir: Path):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=True
        )

    def setUp(self) -> None:
        self._native_cache: dict[tuple[str, str | bytes], list[int]] = {}
        self._reference_cache: dict[tuple[str, str], list[int]] = {}

    def _reference_ids(self, fixture: str, text: str) -> list[int]:
        key = (fixture, text)
        if key not in self._reference_cache:
            tokenizer = (
                self.nfc_reference if fixture == "nfc" else self.legacy_reference
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            self._reference_cache[key] = [int(token_id) for token_id in ids]
        return self._reference_cache[key]

    def _native_ids(self, fixture: str, text: str | bytes) -> list[int]:
        key = (fixture, text)
        if key in self._native_cache:
            return self._native_cache[key]

        model = self.nfc_gguf if fixture == "nfc" else self.legacy_gguf
        command = [
            str(self.tokenizer_bin),
            "--model",
            str(model),
            "--threads",
            "1",
            "--no-bos",
            "--no-escape",
            "--ids",
            "--stdin",
            "--offline",
        ]
        process = subprocess.run(
            command,
            input=text.encode("utf-8") if isinstance(text, str) else text,
            capture_output=True,
            timeout=120,
            cwd=str(self.tmp_path),
        )
        stdout = process.stdout.decode("utf-8", errors="replace")
        stderr = process.stderr.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise AssertionError(
                f"llama-tokenize failed for {fixture}: exit={process.returncode}\n"
                f"stderr={stderr[-4000:]}"
            )

        parsed: list[int] | None = None
        for line in stdout.splitlines():
            candidate = line.strip()
            if not (candidate.startswith("[") and candidate.endswith("]")):
                continue
            try:
                value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, list) and all(isinstance(item, int) for item in value):
                parsed = [int(item) for item in value]
        if parsed is None:
            raise AssertionError(
                f"llama-tokenize emitted no parseable --ids list for {fixture}:\n"
                f"stdout={stdout[-4000:]}\n"
                f"stderr={stderr[-4000:]}"
            )

        self._native_cache[key] = parsed
        return parsed

    def test_source_nfc_metadata_and_native_parity_cover_unicode_boundaries(self):
        self.assertIs(self.nfc_metadata, True, "source NFC must emit a true GGUF flag")
        self.assertIsNot(self.legacy_metadata, True, "no-NFC source must not opt in")
        self.assertEqual(self._read_tokenizer_model(self.nfc_gguf), "gpt2")

        for label, text in NFC_CASES:
            with self.subTest(label=label):
                expected = self._reference_ids("nfc", text)
                actual = self._native_ids("nfc", text)
                self.assertEqual(actual, expected)

    def test_nfc_metadata_is_refused_for_sentencepiece_writer(self):
        tokenizer_dir = self.tmp_path / "source-spm-nfc"
        tokenizer_dir.mkdir()
        tokenizer_json = {
            "version": "1.0",
            "normalizer": {"type": "NFC"},
            "model": {
                "type": "Unigram",
                "vocab": [["<unk>", 0.0], ["a", -1.0]],
                "unk_id": 0,
            },
        }
        (tokenizer_dir / "tokenizer.json").write_text(
            json.dumps(tokenizer_json), encoding="utf-8"
        )

        from gguf import GGUFWriter, SpecialVocab

        special_vocab = SpecialVocab(tokenizer_dir)
        self.assertIs(special_vocab.normalizer_nfc, True)
        writer = GGUFWriter(path=None, arch="llama")
        writer.add_tokenizer_model("llama")
        with self.assertRaisesRegex(ValueError, "only supported for BPE"):
            special_vocab.add_to_gguf(writer)
        self.assertNotIn(NFC_KV, writer.kv_data[0])

    def test_nfc_equivalent_spellings_share_reference_and_native_ids(self):
        for label, composed, decomposed in EQUIVALENT_SPELLINGS:
            with self.subTest(label=label):
                reference_composed = self._reference_ids("nfc", composed)
                reference_decomposed = self._reference_ids("nfc", decomposed)
                self.assertEqual(reference_composed, reference_decomposed)
                self.assertEqual(
                    self._native_ids("nfc", composed), reference_composed
                )
                self.assertEqual(
                    self._native_ids("nfc", decomposed), reference_decomposed
                )

    def test_composition_exclusion_is_not_overcomposed(self):
        # U+0958 has a canonical decomposition whose recomposition is excluded.
        # NFC therefore maps the scalar to the same sequence as U+0915 U+093C.
        # The native tokenizer must match that HF result rather than composing
        # the sequence back to an excluded scalar.
        decomposed = "\u0915\u093c"
        excluded = "\u0958"
        reference_decomposed = self._reference_ids("nfc", decomposed)
        reference_excluded = self._reference_ids("nfc", excluded)
        self.assertEqual(reference_decomposed, reference_excluded)
        self.assertEqual(
            self._native_ids("nfc", decomposed), reference_decomposed
        )
        self.assertEqual(self._native_ids("nfc", excluded), reference_excluded)

    def test_no_nfc_fixture_preserves_legacy_behavior_and_special_boundaries(self):
        self.assertIsNot(self.legacy_metadata, True)
        for label, text in LEGACY_CASES:
            with self.subTest(label=label):
                expected = self._reference_ids("legacy", text)
                actual = self._native_ids("legacy", text)
                self.assertEqual(actual, expected)

        # This is the observed minimal mismatch and guards the opt-in boundary:
        # adding NFC changes decomposed e+acute, while the no-NFC fixture keeps
        # its old BPE spelling.  No concrete token id is frozen here.
        decomposed = "e\u0301"
        self.assertNotEqual(
            self._reference_ids("legacy", decomposed),
            self._reference_ids("nfc", decomposed),
        )
        self.assertNotEqual(
            self._native_ids("legacy", decomposed),
            self._native_ids("nfc", decomposed),
        )

    def test_invalid_utf8_is_replaced_without_crashing_and_legacy_path_is_unchanged(self):
        invalid_utf8 = b"\xc3("
        replacement_text = invalid_utf8.decode("utf-8", errors="replace")

        # The reference tokenizer takes Unicode strings, so decode malformed
        # source bytes with the same U+FFFD replacement policy as the native
        # tokenizer input boundary before asking HF for the expected NFC ids.
        expected_nfc = self._reference_ids("nfc", replacement_text)
        self.assertEqual(self._native_ids("nfc", invalid_utf8), expected_nfc)

        # NFC is opt-in.  The legacy tokenizer path must continue to interpret
        # this malformed input exactly as it does explicit replacement text.
        self.assertEqual(
            self._native_ids("legacy", invalid_utf8),
            self._native_ids("legacy", replacement_text),
        )


if __name__ == "__main__":
    unittest.main()
