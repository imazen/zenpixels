"""Black-box retention controls; synthetic bytes contain no third-party profiles."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).with_name("retain_originals.py")


class RetentionControls(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="icc-retain-control-", dir=scratch))
        self.original = b"original witness"
        (self.root / "original.icc").write_bytes(self.original)
        self.receipt = self.root / "receipt.json"
        self.document = {"rows": [{
            "file": "original.icc", "size": len(self.original),
            "sha256": hashlib.sha256(self.original).hexdigest(),
            "source": "[source](https://example.com/profile)",
            "normalized_hash": "0000000000000000", "murmur3_32": "00000000",
        }]}
        self.receipt.write_text(json.dumps(self.document))
        self.archive = self.root / "archive"

    def run_tool(self, *args, success=True):
        result = subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def test_relative_manifest_links_and_corruption_refusal(self):
        self.run_tool(self.receipt, self.archive)
        manifest = self.archive / "manifest.json"
        self.run_tool(manifest)
        row = json.loads(manifest.read_text())["rows"][0]
        self.assertEqual(row["source"], self.document["rows"][0]["source"])
        obj = self.archive / row["file"]
        self.assertEqual(obj.read_bytes(), self.original)
        damaged = b"changed witness!"
        self.assertEqual(len(damaged), len(self.original))
        obj.write_bytes(damaged)
        self.run_tool(manifest, success=False)
        self.run_tool(self.receipt, self.archive, success=False)
        self.assertEqual(obj.read_bytes(), damaged)

    def test_missing_later_original_prevents_partial_archive(self):
        missing = dict(self.document["rows"][0], file="missing.icc")
        self.document["rows"].append(missing)
        self.receipt.write_text(json.dumps(self.document))
        self.run_tool(self.receipt, self.archive, success=False)
        self.assertFalse(self.archive.exists())

    def test_conflicting_manifest_is_preserved(self):
        self.run_tool(self.receipt, self.archive)
        manifest = self.archive / "manifest.json"
        before = manifest.read_bytes()
        self.document["rows"][0]["active"] = False
        self.receipt.write_text(json.dumps(self.document))
        self.run_tool(self.receipt, self.archive, success=False)
        self.assertEqual(manifest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
