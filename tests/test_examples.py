import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = [
    ROOT / "examples",
    ROOT / "paper_plots",
]


class ExampleNotebookTests(unittest.TestCase):
    def test_example_notebooks_are_valid_json_and_unexecuted(self):
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
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell.get("execution_count"))
                        self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
