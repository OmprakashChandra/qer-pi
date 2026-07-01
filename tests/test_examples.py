import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = [
    ROOT / "examples",
    ROOT / "paper_plots",
]


class ExampleNotebookTests(unittest.TestCase):
    def test_example_notebooks_are_valid_json_and_executed_cleanly(self):
        notebooks = [
            path
            for notebook_dir in NOTEBOOK_DIRS
            for path in sorted(notebook_dir.glob("*.ipynb"))
        ]
        self.assertGreaterEqual(len(notebooks), 3)

        for path in notebooks:
            with self.subTest(path=path.relative_to(ROOT)):
                notebook = json.loads(path.read_text())
                self.assertEqual(notebook["nbformat"], 4)
                self.assertIn("cells", notebook)
                output_count = 0
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertIsNotNone(cell.get("execution_count"))
                        for output in cell.get("outputs", []):
                            self.assertNotEqual(output.get("output_type"), "error")
                            self.assertFalse(
                                output.get("output_type") == "stream"
                                and output.get("name") == "stderr"
                            )
                            output_count += 1
                self.assertGreater(output_count, 0)


if __name__ == "__main__":
    unittest.main()
