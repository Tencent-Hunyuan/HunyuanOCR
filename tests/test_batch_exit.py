import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BatchExitTest(unittest.TestCase):
    def run_batch(self, image_dir, out_dir):
        return subprocess.run(
            [
                sys.executable,
                "inference/vLLM/batch_infer.py",
                "--image-dir",
                str(image_dir),
                "--out-dir",
                str(out_dir),
                "--concurrency",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_failed_images_produce_nonzero_exit_after_recording_all_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            out_dir = Path(tmp) / "output"
            for name in ("a.png", "b.png"):
                (image_dir / name).symlink_to(image_dir / "missing")
            result = self.run_batch(image_dir, out_dir)
            records = [
                json.loads(line)
                for line in (out_dir / "results.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["image"] for row in records}, {"a.png", "b.png"})
            self.assertTrue(
                all(
                    not row["ok"] and "FileNotFoundError" in row["error"]
                    for row in records
                )
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_directory_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            out_dir = Path(tmp) / "output"
            result = self.run_batch(image_dir, out_dir)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((out_dir / "results.jsonl").read_text(), "")

    def test_completed_image_is_skipped_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            (image_dir / "page.png").symlink_to(image_dir / "missing")
            out_dir = Path(tmp) / "output"
            out_dir.mkdir()
            (out_dir / "page.md").write_text("previous result")
            result = self.run_batch(image_dir, out_dir)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((out_dir / "page.md").read_text(), "previous result")
            self.assertEqual((out_dir / "results.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
