import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import qutip
    from qer.optimisation import optimise
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"optional dependency missing: {IMPORT_ERROR}")
class OptimisationTests(unittest.TestCase):
    def test_scs_default_solves_identity_channel(self):
        logical0 = qutip.basis(2, 0)
        logical1 = qutip.basis(2, 1)
        fidelity = optimise(
            logical0,
            logical1,
            [qutip.qeye(2)],
            solver="scs",
            solver_opts={"eps": 1e-5, "max_iters": 2500},
        )

        self.assertGreater(fidelity, 0.99)
        self.assertLessEqual(fidelity, 1.001)


if __name__ == "__main__":
    unittest.main()
