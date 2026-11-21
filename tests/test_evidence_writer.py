"""Reviewed evidence must never be silently replaced by a later render."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import batch_review_sheets
import semantic_sheets

class EvidenceDirectoryProtection(unittest.TestCase):
    def test_existing_review_evidence_is_unchanged_even_for_empty_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "staging"
            (stage / "candidates").mkdir(parents=True)
            (stage / "visual_observations.jsonl").write_text("", encoding="utf8")
            (stage / "candidates/example.jsonl").write_text("", encoding="utf8")
            output = root / "reports/previous_review"
            output.mkdir(parents=True)
            (output / "index.jsonl").write_text('{"id":"already-reviewed"}\n', encoding="utf8")
            (output / "page_0000.jpg").write_bytes(b"immutable evidence sentinel")
            before = {f.name: f.read_bytes() for f in output.iterdir()}
            argv = ["batch_review_sheets.py", "--source", "example",
                    "--out", "previous_review", "--limit", "0"]
            with patch.object(batch_review_sheets, "ROOT", root), patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    batch_review_sheets.main()
            self.assertEqual(before, {f.name: f.read_bytes() for f in output.iterdir()})

class SemanticEvidenceDirectoryProtection(unittest.TestCase):
    def test_existing_semantic_evidence_is_unchanged_even_for_empty_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "staging/candidates").mkdir(parents=True)
            (root / "staging/candidates/example.jsonl").write_text("", encoding="utf8")
            output = root / "reports/previous_semantic_review"
            output.mkdir(parents=True)
            (output / "index_0000.jsonl").write_text('{"id":"already-reviewed"}\\n', encoding="utf8")
            (output / "page_0000.jpg").write_bytes(b"immutable semantic evidence")
            before = {f.name: f.read_bytes() for f in output.iterdir()}
            argv = ["semantic_sheets.py", "--source", "example",
                    "--prefix", "previous_semantic_review", "--count", "0"]
            with patch.object(semantic_sheets, "ROOT", root), patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    semantic_sheets.main()
            self.assertEqual(before, {f.name: f.read_bytes() for f in output.iterdir()})

if __name__ == "__main__":
    unittest.main()
