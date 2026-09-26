"""Synthetic fixtures exercise failure handling without the real dataset."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bottle_vae.data_check import FOLDERS, check_dataset, main, totals_from_counts

SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DataCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in FOLDERS:
            directory = self.root / folder
            directory.mkdir(parents=True)
            name = "000_mask.png" if folder.startswith("ground_truth") else "000.png"
            (directory / name).write_bytes(SIGNATURE)

    def test_matching_pairs(self):
        counts, issues = check_dataset(self.root)
        self.assertEqual(issues, [])
        self.assertEqual(totals_from_counts(counts)["defective test"], 3)

    def test_missing_folder(self):
        folder = self.root / "train/good"
        (folder / "000.png").unlink()
        folder.rmdir()
        counts, issues = check_dataset(self.root)
        self.assertIsNone(counts["train/good"])
        self.assertIn("Missing folder: train/good", issues)

    def test_wrong_mask_name_even_with_equal_counts(self):
        folder = self.root / "ground_truth/broken_large"
        (folder / "000_mask.png").rename(folder / "999_mask.png")
        _, issues = check_dataset(self.root)
        self.assertTrue(any("Missing mask:" in issue for issue in issues))
        self.assertTrue(any("Orphan mask:" in issue for issue in issues))

    def test_invalid_png_and_unexpected_training_class(self):
        (self.root / "test/good/000.png").write_bytes(b"not a PNG")
        (self.root / "train/defect").mkdir()
        _, issues = check_dataset(self.root)
        self.assertTrue(any("Invalid PNG signature:" in issue for issue in issues))
        self.assertIn("Unexpected entry: train/defect", issues)

    def test_count_mismatch_exit(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = main(["--data-root", str(self.root)])
        self.assertEqual(status, 1)
        self.assertIn("MISMATCH", output.getvalue())

    def test_missing_root_exit(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = main(["--data-root", str(self.root / "absent")])
        self.assertEqual(status, 1)
        self.assertIn("Missing folder: train/good", output.getvalue())


if __name__ == "__main__":
    unittest.main()
