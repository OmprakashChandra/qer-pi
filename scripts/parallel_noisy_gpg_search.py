"""Parallel guarded search for high-budget detuned noisy-GPG recovery points.

Workers run independent random-seed attempts and write candidate pickle files.
Only the parent process updates the main cache, CSV, and plot, so a worse
candidate cannot overwrite the current best point.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import json
import math
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    p for p in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents] if (p / "src" / "qer").exists()
)
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from qer import gpgs  # noqa: E402
from qer.bk_recovery import petz_recovery_kraus  # noqa: E402
from qer.codewords import bgmcode_kets_in_top_block  # noqa: E402
from qer.noisemodel import noisemodel  # noqa: E402
from qer.optimisation import no_recovery  # noqa: E402


CACHE_PATH = (
    REPO_ROOT
    / "datas"
    / "noisy_gpgs_pulses"
    / "cache"
    / "detuned_usual_gpg_p5e-4_p1e-3_Csweep.pkl"
)
CSV_PATH = (
    REPO_ROOT
    / "datas"
    / "noisy_gpgs_pulses"
    / "detuned_usual_gpg_p5e-4_p1e-3_Csweep_metrics.csv"
)
PLOT_DIR = REPO_ROOT / "plots" / "AD" / "noisy_gpg_implementation"
PNG_PATH = PLOT_DIR / "detuned_usual_gpg_cooperativity_sweep.png"
PDF_PATH = PLOT_DIR / "detuned_usual_gpg_cooperativity_sweep.pdf"
NOISELESS_CACHE_PATH = (
    REPO_ROOT / "datas" / "noiseless_gpgs_pulses" / "cache" / "gpg_exact_ad_sweep_cache.pkl"
)
ERROR_FREE_REF_PATH = (
    REPO_ROOT
    / "datas"
    / "noisy_gpgs_pulses"
    / "cache"
    / "error_free_usual_gpg_refs_p5e-4_p1e-3.pkl"
)
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "improving_noisy_gpg.ipynb"
CANDIDATE_DIR = CACHE_PATH.parent / "parallel_candidates"

PLOT_P_VALUES = [5e-4, 1e-3]
PLOT_COOPERATIVITIES = [1e6, 1e7, 1e8, 1e9, 1e10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=5e-4)
    parser.add_argument("--c", type=float, default=1e10, help="Cooperativity to polish.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--restarts", type=int, default=14)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--extra-pulses", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--stop-after-accepts", type=int, default=None)
    parser.add_argument("--target-inf", type=float, default=None)
    parser.add_argument("--target-eps", type=float, default=None)
    parser.add_argument("--base-seed", type=int, default=2_900_000)
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--no-notebook-preview", action="store_true")
    return parser.parse_args()


def cache_key(p: float, cooperativity: float) -> str:
    """Historical key used by the notebook plot cache."""
    return (
        f"p={float(p):.12e}|C={float(cooperativity):.12e}|mode=detuned"
        f"|restarts=3|maxiter=500|complete-prior-v1"
    )


def default_target_window(cooperativity: float) -> tuple[float | None, float]:
    if math.isclose(float(cooperativity), 1e9, rel_tol=0.0, abs_tol=1.0):
        return 1e-6, 2e-6
    if math.isclose(float(cooperativity), 1e10, rel_tol=0.0, abs_tol=10.0):
        return 5e-7, 2e-7
    return None, 0.0


def bgm_problem():
    b, g, m = 3, 3, 1
    num_qubits = 2 * b * m + g
    ket0, ket1, _ = bgmcode_kets_in_top_block(b, g, m, return_qutip=True)
    rho = (ket0 * ket0.dag() + ket1 * ket1.dag()) / 2

    def exact_global_ad(p):
        return noisemodel(
            "global symmetric amplitude damping",
            num_qubits,
            float(p),
            1.0,
            return_rep="super",
            dynamics="exact",
        )

    def approximate_petz_recovery(p):
        kraus = noisemodel(
            "global symmetric amplitude damping",
            num_qubits,
            float(p),
            1.0,
            return_rep="kraus",
            dynamics="approx",
        )
        return petz_recovery_kraus(
            gpgs.restrict_operators_to_dimension(kraus, num_qubits + 1),
            rho,
        )

    return num_qubits, rho, (ket0, ket1), exact_global_ad, approximate_petz_recovery


def load_noiseless_prior(p: float) -> dict[tuple[str, int], Any]:
    prior: dict[tuple[str, int], Any] = {}
    for path in [ERROR_FREE_REF_PATH, NOISELESS_CACHE_PATH]:
        if not path.exists():
            continue
        with path.open("rb") as f:
            old_cache = pickle.load(f)
        points = sorted(
            old_cache["points"].values(),
            key=lambda point: abs(point["metrics"]["p"] - float(p)),
        )
        for point in points:
            for key, sequence in point["pulse_sequences"].items():
                prior.setdefault(key, sequence)
    return prior


def load_error_free_refs(p_values: list[float]) -> dict[float, float]:
    refs: dict[float, float] = {}
    if ERROR_FREE_REF_PATH.exists():
        with ERROR_FREE_REF_PATH.open("rb") as f:
            ref_cache = pickle.load(f)
        for point in ref_cache.get("points", {}).values():
            p_ref = float(point["metrics"]["p"])
            for p in p_values:
                if abs(p_ref - float(p)) <= 1e-15:
                    refs[float(p)] = float(point["metrics"]["GPG infidelity"])

    missing = [float(p) for p in p_values if float(p) not in refs]
    if missing:
        with NOISELESS_CACHE_PATH.open("rb") as f:
            old_cache = pickle.load(f)
        old_points = list(old_cache["points"].values())
        for p in missing:
            nearest = min(old_points, key=lambda point: abs(float(point["metrics"]["p"]) - p))
            refs[p] = float(nearest["metrics"]["GPG infidelity"])
    return refs


def no_recovery_infidelities(
    p_values: list[float],
    *,
    noise_type: str = "global symmetric amplitude damping",
) -> dict[float, float]:
    """Return no-recovery entanglement infidelities for the BGM code."""
    num_qubits, rho, _, _, _ = bgm_problem()
    out: dict[float, float] = {}
    for p in p_values:
        kraus = noisemodel(
            noise_type,
            num_qubits,
            float(p),
            1.0,
            return_rep="kraus",
            dynamics="exact",
        )
        ambient_dim = kraus[0].shape[0]
        rho_for_channel = (
            rho
            if ambient_dim == rho.shape[0]
            else gpgs._embed_top_block_rho(rho, ambient_dim)
        )
        out[float(p)] = abs(1.0 - float(no_recovery(rho_for_channel, kraus)))
    return out


def settings_for_attempt(
    *,
    restarts: int,
    maxiter: int,
    extra_pulses: int,
    seed_offset: int,
):
    def settings(label: str, eig_index: int, counter: int) -> dict[str, int]:
        out = gpgs.default_gpg_recovery_settings(label, eig_index, counter).copy()
        out["pulses"] += int(extra_pulses)
        out["restarts"] = int(restarts)
        out["maxiter"] = int(maxiter)
        out["seed"] = int(seed_offset) + 1009 * int(counter) + 37 * int(eig_index)
        return out

    return settings


def run_worker(config: dict[str, Any], attempt: int, seed_offset: int) -> dict[str, Any]:
    """Run one independent candidate attempt and write the point to a file."""
    try:
        num_qubits, rho, logical_kets, exact_ad, recovery_ops = bgm_problem()
        with CACHE_PATH.open("rb") as f:
            cache = pickle.load(f)
        key = cache_key(config["p"], config["cooperativity"])
        old_point = cache["points"][key]
        prior = load_noiseless_prior(config["p"])
        prior.update(old_point.get("pulse_sequences", {}))

        t0 = time.perf_counter()
        point = gpgs.run_gpg_recovery(
            rho,
            exact_ad,
            recovery_ops,
            float(config["p"]),
            logical_kets=logical_kets,
            reference_weight=num_qubits,
            prior_sequences=prior,
            settings_fn=settings_for_attempt(
                restarts=config["restarts"],
                maxiter=config["maxiter"],
                extra_pulses=config["extra_pulses"],
                seed_offset=seed_offset,
            ),
            gpg_mode="detuned",
            cooperativity=float(config["cooperativity"]),
            log=None,
        )
        wall = float(time.perf_counter() - t0)
        point["metrics"]["baseline p"] = float(config["p"])
        point["metrics"]["below baseline"] = bool(
            point["metrics"]["GPG infidelity"] < float(config["p"])
        )
        point["metrics"]["wall seconds"] = wall
        point["parallel_search"] = {
            "attempt": attempt,
            "seed_offset": seed_offset,
            "workers": config["workers"],
            "restarts": config["restarts"],
            "maxiter": config["maxiter"],
            "extra_pulses": config["extra_pulses"],
        }

        candidate_dir = Path(config["candidate_dir"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / (
            f"candidate_p{config['p']:.3e}_C{config['cooperativity']:.3e}"
            f"_attempt{attempt:05d}_seed{seed_offset}.pkl"
        )
        with candidate_path.open("wb") as f:
            pickle.dump(point, f)

        return {
            "ok": True,
            "attempt": attempt,
            "seed_offset": seed_offset,
            "candidate_path": str(candidate_path),
            "gpg_infidelity": float(point["metrics"]["GPG infidelity"]),
            "max_state": float(point["metrics"]["max state-prep infidelity"]),
            "wall": wall,
        }
    except Exception:
        return {
            "ok": False,
            "attempt": attempt,
            "seed_offset": seed_offset,
            "traceback": traceback.format_exc(),
        }


def save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)


def refresh_csv_and_plot(update_notebook_preview: bool) -> pd.DataFrame:
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)

    best_rows: dict[tuple[float, float], dict[str, Any]] = {}
    for key, point in cache.get("points", {}).items():
        metrics = point.get("metrics", {})
        if metrics.get("gpg_mode") not in (None, "detuned"):
            continue
        p = float(metrics.get("p", float("nan")))
        cooperativity = float(metrics.get("GPG cooperativity", float("nan")))
        if not any(abs(p - float(p0)) <= 1e-15 for p0 in PLOT_P_VALUES):
            continue
        if not any(abs(cooperativity - float(c0)) <= max(1.0, 1e-12 * float(c0)) for c0 in PLOT_COOPERATIVITIES):
            continue
        group_key = (p, cooperativity)
        infidelity = float(metrics["GPG infidelity"])
        if (
            group_key not in best_rows
            or infidelity < float(best_rows[group_key]["GPG infidelity"])
        ):
            row = dict(metrics)
            row["cache_key"] = key
            best_rows[group_key] = row

    metrics = pd.DataFrame(best_rows.values()).sort_values(["p", "GPG cooperativity"]).reset_index(drop=True)
    refs = load_error_free_refs(PLOT_P_VALUES)
    metrics["error-free GPG infidelity"] = metrics["p"].map(lambda value: refs[float(value)])
    no_recovery_refs = no_recovery_infidelities(PLOT_P_VALUES)
    metrics["no recovery infidelity"] = metrics["p"].map(
        lambda value: no_recovery_refs[float(value)]
    )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(CSV_PATH, index=False)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.1), dpi=180)
    colors = {5e-4: "tab:blue", 1e-3: "tab:orange"}
    for p, group in metrics.groupby("p"):
        group = group.sort_values("GPG cooperativity")
        color = colors.get(float(p))
        ax.loglog(
            group["GPG cooperativity"],
            group["GPG infidelity"],
            "o-",
            lw=1.6,
            ms=4.5,
            color=color,
            label=rf"detuned GPG, $p={float(p):.0e}$",
        )
        ax.axhline(
            refs[float(p)],
            color=color,
            ls=":",
            lw=1.2,
            alpha=0.9,
            label=rf"perfect-cavity GPG, $p={float(p):.0e}$",
        )
        ax.axhline(
            no_recovery_refs[float(p)],
            color=color,
            ls="--",
            lw=1.1,
            alpha=0.8,
            label=rf"no recovery, $p={float(p):.0e}$",
        )
    ax.set_xlabel(r"cooperativity $C$")
    ax.set_ylabel(r"Entanglement infidelity $(1-F)$")
    ax.grid(True, which="both", ls=":", lw=0.55, alpha=0.65)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, bbox_inches="tight")
    plt.close(fig)

    if update_notebook_preview:
        update_notebook_plot_preview(metrics)
    return metrics


def update_notebook_plot_preview(metrics: pd.DataFrame) -> None:
    if not NOTEBOOK_PATH.exists() or not PNG_PATH.exists():
        return

    nb = json.loads(NOTEBOOK_PATH.read_text())
    p5 = metrics[metrics["p"].sub(5e-4).abs() < 1e-15].copy()
    vals = {
        float(row["GPG cooperativity"]): float(row["GPG infidelity"])
        for _, row in p5.iterrows()
    }
    summary_src = [
        "## Latest noisy-GPG cooperativity plot\n",
        "\n",
        f"Updated: `{time.strftime('%Y-%m-%d %H:%M')}`. "
        "This is the only rendered plot preview kept in this notebook.\n",
        "\n",
        "Latest polished `p=5e-4` high-C points:\n",
        "\n",
        f"- `C=1e8`: `{vals.get(1e8, float('nan')):.6e}`\n",
        f"- `C=1e9`: `{vals.get(1e9, float('nan')):.6e}`\n",
        f"- `C=1e10`: `{vals.get(1e10, float('nan')):.6e}`\n",
    ]
    preview_src = [
        "# Latest plot preview; no sweep is run in this cell.\n",
        "from pathlib import Path\n",
        "from IPython.display import Image, display\n",
        "\n",
        "plot_path = Path('../../plots/AD/noisy_gpg_implementation/"
        "detuned_usual_gpg_cooperativity_sweep.png')\n",
        "display(Image(filename=str(plot_path)))\n",
    ]
    png_b64 = base64.b64encode(PNG_PATH.read_bytes()).decode("ascii")
    preview_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": preview_src,
        "outputs": [
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": png_b64,
                    "text/plain": ["<IPython.core.display.Image object>"],
                },
            }
        ],
    }

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and src.startswith(
            "## Latest noisy-GPG cooperativity plot"
        ):
            cell["source"] = summary_src
            break

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "Latest plot preview" in src:
            nb["cells"][i] = preview_cell
            break

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1))


def cleanup_candidates(results: list[dict[str, Any]], accepted_path: str | None, keep: bool) -> None:
    if keep:
        return
    for result in results:
        path = result.get("candidate_path")
        if path is None or path == accepted_path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    target_inf, target_eps = default_target_window(args.c)
    if args.target_inf is not None:
        target_inf = args.target_inf
    if args.target_eps is not None:
        target_eps = args.target_eps
    threshold = None if target_inf is None else float(target_inf) + float(target_eps)

    key = cache_key(args.p, args.c)
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    if key not in cache["points"]:
        raise KeyError(f"No cached point for p={args.p:g}, C={args.c:g}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = CACHE_PATH.with_name(
        CACHE_PATH.stem + f".before_parallel_search_{stamp}" + CACHE_PATH.suffix
    )
    shutil.copy2(CACHE_PATH, backup)
    print(f"Backup: {backup}", flush=True)
    if threshold is None:
        print(f"No target window configured for C={args.c:g}.", flush=True)
    else:
        print(
            f"Target window for C={args.c:g}: target={target_inf:.6e}, "
            f"eps={target_eps:.6e}, threshold={threshold:.6e}",
            flush=True,
        )

    config = {
        "p": float(args.p),
        "cooperativity": float(args.c),
        "workers": int(args.workers),
        "restarts": int(args.restarts),
        "maxiter": int(args.maxiter),
        "extra_pulses": int(args.extra_pulses),
        "candidate_dir": str(CANDIDATE_DIR),
    }

    accepted = 0
    attempts_done = 0
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        while args.max_attempts is None or attempts_done < args.max_attempts:
            with CACHE_PATH.open("rb") as f:
                cache = pickle.load(f)
            old_point = cache["points"][key]
            old_inf = float(old_point["metrics"]["GPG infidelity"])

            batch_size = args.workers
            if args.max_attempts is not None:
                batch_size = min(batch_size, args.max_attempts - attempts_done)
            if batch_size <= 0:
                break

            print(
                f"\nLaunching batch of {batch_size} attempts from current best {old_inf:.6e}",
                flush=True,
            )
            submitted = []
            for _ in range(batch_size):
                attempts_done += 1
                seed_offset = (
                    int(args.base_seed)
                    + 100_000 * attempts_done
                    + int(round(float(args.c))).bit_length() * 1000
                )
                submitted.append(pool.submit(run_worker, config, attempts_done, seed_offset))

            results = []
            for future in futures.as_completed(submitted):
                result = future.result()
                results.append(result)
                if not result["ok"]:
                    print(
                        f"Attempt {result['attempt']} failed:\n{result['traceback']}",
                        flush=True,
                    )
                    continue
                print(
                    f"Attempt {result['attempt']} seed={result['seed_offset']} "
                    f"inf={result['gpg_infidelity']:.6e} "
                    f"max_state={result['max_state']:.6e} "
                    f"wall={result['wall'] / 60:.1f} min",
                    flush=True,
                )

            valid = [result for result in results if result.get("ok")]
            if not valid:
                cleanup_candidates(results, None, args.keep_candidates)
                continue

            best = min(valid, key=lambda result: result["gpg_infidelity"])
            accepted_path = None
            if best["gpg_infidelity"] < old_inf:
                accepted_path = best["candidate_path"]
                with Path(accepted_path).open("rb") as f:
                    best_point = pickle.load(f)
                cache["points"][key] = best_point
                save_cache(cache)
                accepted += 1
                print(
                    f"ACCEPTED: {old_inf:.6e} -> {best['gpg_infidelity']:.6e} "
                    f"(attempt {best['attempt']})",
                    flush=True,
                )
                metrics = refresh_csv_and_plot(not args.no_notebook_preview)
                print(
                    metrics[
                        [
                            "p",
                            "GPG cooperativity",
                            "GPG infidelity",
                            "max state-prep infidelity",
                        ]
                    ].to_string(index=False),
                    flush=True,
                )
                if threshold is not None and best["gpg_infidelity"] <= threshold:
                    print(
                        f"Reached target threshold: {best['gpg_infidelity']:.6e} "
                        f"<= {threshold:.6e}",
                        flush=True,
                    )
                    cleanup_candidates(results, accepted_path, args.keep_candidates)
                    break
                if (
                    args.stop_after_accepts is not None
                    and accepted >= args.stop_after_accepts
                ):
                    print(f"Reached stop-after-accepts={args.stop_after_accepts}.", flush=True)
                    cleanup_candidates(results, accepted_path, args.keep_candidates)
                    break
            else:
                print(
                    f"Kept old: best candidate {best['gpg_infidelity']:.6e} "
                    f">= old {old_inf:.6e}",
                    flush=True,
                )

            cleanup_candidates(results, accepted_path, args.keep_candidates)

    print("Parallel noisy-GPG search finished.", flush=True)


if __name__ == "__main__":
    main()
