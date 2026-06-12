import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datas" / "final_gpg_pulses"
PULSE_COLUMNS = ["alpha", "beta", "gamma", "kappa", "detuning"]
SEQUENCE_COLUMNS = ["pulse_index", *PULSE_COLUMNS]


class FinalGpgPulseDataTests(unittest.TestCase):
    def test_required_files_exist(self):
        for name in [
            "README.md",
            "all_pulses.csv",
            "best_metrics.csv",
            "best_metrics.json",
            "manifest.csv",
        ]:
            self.assertTrue((DATA_DIR / name).exists(), name)

    def test_all_pulses_has_five_parameter_columns(self):
        with (DATA_DIR / "all_pulses.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertIsNotNone(reader.fieldnames)
            for column in PULSE_COLUMNS:
                self.assertIn(column, reader.fieldnames)
            first = next(reader)
            for column in PULSE_COLUMNS:
                float(first[column])

    def test_manifest_sequences_exist_with_expected_header(self):
        with (DATA_DIR / "manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertGreater(len(rows), 0)
        for row in rows:
            sequence_path = DATA_DIR / row["file"]
            self.assertTrue(sequence_path.exists(), row["file"])
            with sequence_path.open(newline="") as sequence_handle:
                reader = csv.reader(sequence_handle)
                self.assertEqual(next(reader), SEQUENCE_COLUMNS)

    def test_best_metrics_json_is_well_formed(self):
        metrics = json.loads((DATA_DIR / "best_metrics.json").read_text())
        self.assertIsInstance(metrics, list)
        self.assertGreater(len(metrics), 0)


if __name__ == "__main__":
    unittest.main()
