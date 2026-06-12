import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    from qer.codewords import bgmcode_kets_in_top_block, piqs_block_dims, piqs_total_dim
except ModuleNotFoundError as exc:
    np = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"optional dependency missing: {IMPORT_ERROR}")
class CodewordTests(unittest.TestCase):
    def test_bgm_codewords_are_normalized_and_orthogonal(self):
        ket0, ket1, num_qubits = bgmcode_kets_in_top_block(1, 1, 1)

        self.assertEqual(num_qubits, 3)
        self.assertAlmostEqual(float(np.linalg.norm(ket0)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(ket1)), 1.0)
        self.assertAlmostEqual(abs(np.vdot(ket0, ket1)), 0.0)

    def test_piqs_reduced_dimensions(self):
        self.assertEqual(piqs_block_dims(3), [4, 2])
        self.assertEqual(piqs_total_dim(3), 6)


if __name__ == "__main__":
    unittest.main()
