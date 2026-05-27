"""Refine spike points in the noiseless exact-AD GPG recovery sweep cache."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    p for p in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents] if (p / "python" / "codes").exists()
)
sys.path.insert(0, str(REPO_ROOT / "python"))

from codes import gpgs  # noqa: E402
from codes.bk_recovery import petz_recovery_kraus  # noqa: E402
from codes.codewords import bgmcode_kets_in_top_block  # noqa: E402
from codes.noisemodel import noisemodel  # noqa: E402


SWEEP_CACHE_PATH = (
    REPO_ROOT / "datas" / "noiseless_gpgs_pulses" / "cache" / "gpg_exact_ad_sweep_cache.pkl"
)
SWEEP_FIG_DIR = REPO_ROOT / "plots" / "AD"
SWEEP_FIG_BASENAME = "noiseless_gpg_implementation"


def bgm_exact_ad_problem():
    """Return the BGM `(3,3,1)` exact-AD/noiseless-GPG refinement problem."""
    b, g, m = 3, 3, 1
    num_qubits = 2 * b * m + g
    ket0, ket1, _ = bgmcode_kets_in_top_block(b, g, m, return_qutip=True)
    rho = (ket0 * ket0.dag() + ket1 * ket1.dag()) / 2

    def exact_ad(p):
        return noisemodel(
            "global symmetric amplitude damping",
            num_qubits,
            float(p),
            1.0,
            return_rep="super",
            dynamics="exact",
        )

    def approx_petz(p):
        approx_kraus = noisemodel(
            "global symmetric amplitude damping",
            num_qubits,
            float(p),
            1.0,
            return_rep="kraus",
            dynamics="approx",
        )
        approx_kraus = gpgs.restrict_operators_to_dimension(approx_kraus, num_qubits + 1)
        return petz_recovery_kraus(approx_kraus, rho)

    return num_qubits, rho, (ket0, ket1), exact_ad, approx_petz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p",
        type=float,
        nargs="*",
        help="Specific p values to refine. Defaults to points with large GPG penalty.",
    )
    parser.add_argument("--min-penalty", type=float, default=1e-6)
    parser.add_argument("--state-tol", type=float, default=1e-4)
    parser.add_argument("--synth-tol", type=float, default=0.08)
    parser.add_argument("--mode", choices=["quick", "normal", "deep"], default="normal")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_qubits, rho, logical_kets, exact_ad, approx_petz = bgm_exact_ad_problem()
    cache = gpgs.load_gpg_sweep_cache(SWEEP_CACHE_PATH)

    points = args.p if args.p else gpgs.default_spike_points(cache, args.min_penalty)
    if not points:
        print("No spike points matched the thresholds.")
        return

    backup_path = SWEEP_CACHE_PATH.with_suffix(
        f".before_refine_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    )
    if not args.dry_run:
        shutil.copy2(SWEEP_CACHE_PATH, backup_path)
        print(f"Backed up cache to {backup_path}")

    summary_rows = []
    for p in points:
        key = gpgs.p_cache_key(p)
        old_point = gpgs.cache_point_for_p(cache, p, cache_path=SWEEP_CACHE_PATH)
        old_gpg = float(old_point["metrics"]["GPG infidelity"])
        old_penalty = float(old_point["metrics"]["GPG - exact infidelity penalty"])

        new_point, suspect_keys = gpgs.refine_noiseless_gpg_recovery_point(
            cache,
            p,
            rho,
            exact_ad,
            approx_petz,
            state_tol=args.state_tol,
            synth_tol=args.synth_tol,
            mode=args.mode,
            max_targets=args.max_targets,
            logical_kets=logical_kets,
            reference_weight=num_qubits,
            cache_path=SWEEP_CACHE_PATH,
            log=lambda msg: print(msg, flush=True),
        )
        new_gpg = float(new_point["metrics"]["GPG infidelity"])
        new_penalty = float(new_point["metrics"]["GPG - exact infidelity penalty"])
        improved = new_gpg < old_gpg

        summary_rows.append(
            {
                "p": float(p),
                "targets": len(suspect_keys),
                "old GPG infidelity": old_gpg,
                "new GPG infidelity": new_gpg,
                "old penalty": old_penalty,
                "new penalty": new_penalty,
                "accepted": improved,
            }
        )

        if improved and not args.dry_run:
            cache["points"][key] = new_point
            gpgs.save_gpg_sweep_cache(cache, SWEEP_CACHE_PATH)
            print(f"  accepted improved point p={p:.12g}", flush=True)
        else:
            print(f"  kept previous point p={p:.12g}", flush=True)

    if not args.dry_run:
        fig_path = gpgs.redraw_gpg_recovery_sweep_figure(
            cache,
            SWEEP_FIG_DIR,
            basename=SWEEP_FIG_BASENAME,
        )
        print(f"Redrew sweep figure at {fig_path}")

    summary = pd.DataFrame(summary_rows)
    print("\nRefinement summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
