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
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    p for p in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents] if (p / "src" / "qer").exists()
)
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qer-mpl"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


DEFAULT_PLOT_P_VALUES = [5e-4, 1e-3]
DEFAULT_PLOT_COOPERATIVITIES = [1e6, 1e7, 1e8, 1e9, 1e10]
_RUNTIME_READY = False


def load_runtime_dependencies() -> None:
    """Import scientific dependencies after CLI parsing."""
    global _RUNTIME_READY
    global plt, pd, gpgs
    global petz_recovery_kraus, bgmcode_kets_in_top_block
    global noisemodel, no_recovery
    if _RUNTIME_READY:
        return

    import matplotlib.pyplot as plt  # noqa: F401
    import pandas as pd  # noqa: F401

    from qer import gpgs  # noqa: F401
    from qer.bk_recovery import petz_recovery_kraus  # noqa: F401
    from qer.codewords import bgmcode_kets_in_top_block  # noqa: F401
    from qer.noisemodel import noisemodel  # noqa: F401
    from qer.optimisation import no_recovery  # noqa: F401

    _RUNTIME_READY = True


@dataclass(frozen=True)
class ScriptPaths:
    cache_path: Path
    csv_path: Path
    plot_dir: Path
    plot_stem: str
    candidate_dir: Path
    prior_cache_paths: tuple[Path, ...]
    reference_cache_paths: tuple[Path, ...]
    notebook_path: Path | None

    @property
    def png_path(self) -> Path:
        return self.plot_dir / f"{self.plot_stem}.png"

    @property
    def pdf_path(self) -> Path:
        return self.plot_dir / f"{self.plot_stem}.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Pickle cache containing detuned GPG recovery sweep points.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Output metrics CSV. Defaults to '<cache-stem>_metrics.csv' next to the cache.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for regenerated PDF/PNG plots. Defaults to '<cache-dir>/plots'.",
    )
    parser.add_argument(
        "--plot-stem",
        default="detuned_gpg_cooperativity_sweep",
        help="Filename stem for regenerated PDF/PNG plots.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="Scratch directory for worker candidate pickles. Defaults to '<cache-dir>/parallel_candidates'.",
    )
    parser.add_argument(
        "--prior-cache",
        type=Path,
        action="append",
        default=[],
        help="Optional detuned five-parameter pulse cache to warm-start from. May be repeated.",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        action="append",
        default=[],
        help="Optional cache providing error-free GPG reference lines. May be repeated.",
    )
    parser.add_argument(
        "--cache-key-style",
        choices=["canonical", "legacy"],
        default="canonical",
        help="Use current qer cache keys or the historical notebook cache key.",
    )
    parser.add_argument(
        "--notebook-path",
        type=Path,
        default=None,
        help="Optional notebook to update with a rendered plot preview.",
    )
    parser.add_argument(
        "--update-notebook-preview",
        action="store_true",
        help="Update --notebook-path after accepting an improved point.",
    )
    parser.add_argument("--plot-p", type=float, nargs="*", default=DEFAULT_PLOT_P_VALUES)
    parser.add_argument("--plot-c", type=float, nargs="*", default=DEFAULT_PLOT_COOPERATIVITIES)
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
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> ScriptPaths:
    cache_path = args.cache_path.expanduser()
    csv_path = (
        args.csv_path.expanduser()
        if args.csv_path is not None
        else cache_path.with_name(f"{cache_path.stem}_metrics.csv")
    )
    plot_dir = (
        args.plot_dir.expanduser()
        if args.plot_dir is not None
        else cache_path.parent / "plots"
    )
    candidate_dir = (
        args.candidate_dir.expanduser()
        if args.candidate_dir is not None
        else cache_path.parent / "parallel_candidates"
    )
    notebook_path = args.notebook_path.expanduser() if args.notebook_path is not None else None
    return ScriptPaths(
        cache_path=cache_path,
        csv_path=csv_path,
        plot_dir=plot_dir,
        plot_stem=args.plot_stem,
        candidate_dir=candidate_dir,
        prior_cache_paths=tuple(path.expanduser() for path in args.prior_cache),
        reference_cache_paths=tuple(path.expanduser() for path in args.reference_cache),
        notebook_path=notebook_path,
    )


def cache_key(p: float, cooperativity: float, style: str = "canonical") -> str:
    """Return the target cache key for a detuned GPG recovery point."""
    if style == "canonical":
        return gpgs.gpg_recovery_cache_key(
            p,
            gpg_mode="detuned",
            cooperativity=cooperativity,
        )
    if style != "legacy":
        raise ValueError("cache key style must be 'canonical' or 'legacy'.")
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


def load_detuned_prior(
    p: float,
    prior_cache_paths: tuple[Path, ...],
) -> dict[tuple[str, int], Any]:
    prior: dict[tuple[str, int], Any] = {}
    for path in prior_cache_paths:
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
                try:
                    sequence = gpgs.coerce_detuned_pulse_params(sequence)
                except ValueError as exc:
                    raise ValueError(
                        f"{path} contains a non-detuned prior sequence for {key!r}. "
                        "Detuned warm starts must have five columns: "
                        "alpha, beta, gamma, kappa, detuning."
                    ) from exc
                prior.setdefault(key, sequence)
    return prior


def load_error_free_refs(
    p_values: list[float],
    reference_cache_paths: tuple[Path, ...],
) -> dict[float, float]:
    refs: dict[float, float] = {}
    for path in reference_cache_paths:
        if not path.exists():
            continue
        with path.open("rb") as f:
            ref_cache = pickle.load(f)
        points = list(ref_cache.get("points", {}).values())
        for p in p_values:
            if float(p) in refs or not points:
                continue
            nearest = min(points, key=lambda point: abs(float(point["metrics"]["p"]) - float(p)))
            refs[float(p)] = float(nearest["metrics"]["GPG infidelity"])
    for p in p_values:
        refs.setdefault(float(p), float("nan"))
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
    load_runtime_dependencies()
    try:
        num_qubits, rho, logical_kets, exact_ad, recovery_ops = bgm_problem()
        cache_path = Path(config["cache_path"])
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        key = config["target_cache_key"]
        old_point = cache["points"][key]
        prior = load_detuned_prior(
            config["p"],
            tuple(Path(path) for path in config["prior_cache_paths"]),
        )
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


def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)


def refresh_csv_and_plot(
    paths: ScriptPaths,
    *,
    plot_p_values: list[float],
    plot_cooperativities: list[float],
    update_notebook_preview: bool,
) -> pd.DataFrame:
    with paths.cache_path.open("rb") as f:
        cache = pickle.load(f)

    best_rows: dict[tuple[float, float], dict[str, Any]] = {}
    for key, point in cache.get("points", {}).items():
        metrics = point.get("metrics", {})
        if metrics.get("gpg_mode") != "detuned":
            continue
        p = float(metrics.get("p", float("nan")))
        cooperativity = float(metrics.get("GPG cooperativity", float("nan")))
        if not any(abs(p - float(p0)) <= 1e-15 for p0 in plot_p_values):
            continue
        if not any(
            abs(cooperativity - float(c0)) <= max(1.0, 1e-12 * float(c0))
            for c0 in plot_cooperativities
        ):
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

    if not best_rows:
        raise RuntimeError("No detuned GPG cache rows matched --plot-p/--plot-c filters.")

    metrics = pd.DataFrame(best_rows.values()).sort_values(["p", "GPG cooperativity"]).reset_index(drop=True)
    refs = load_error_free_refs(plot_p_values, paths.reference_cache_paths)
    metrics["error-free GPG infidelity"] = metrics["p"].map(lambda value: refs[float(value)])
    no_recovery_refs = no_recovery_infidelities(plot_p_values)
    metrics["no recovery infidelity"] = metrics["p"].map(
        lambda value: no_recovery_refs[float(value)]
    )

    paths.csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.csv_path, index=False)

    paths.plot_dir.mkdir(parents=True, exist_ok=True)
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
        if math.isfinite(refs[float(p)]):
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
    fig.savefig(paths.pdf_path, bbox_inches="tight")
    fig.savefig(paths.png_path, bbox_inches="tight")
    plt.close(fig)

    if update_notebook_preview:
        update_notebook_plot_preview(metrics, paths.notebook_path, paths.png_path)
    return metrics


def update_notebook_plot_preview(
    metrics: pd.DataFrame,
    notebook_path: Path | None,
    png_path: Path,
) -> None:
    if notebook_path is None or not notebook_path.exists() or not png_path.exists():
        return

    nb = json.loads(notebook_path.read_text())
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
    rel_plot_path = os.path.relpath(png_path, notebook_path.parent)
    preview_src = [
        "# Latest plot preview; no sweep is run in this cell.\n",
        "from pathlib import Path\n",
        "from IPython.display import Image, display\n",
        "\n",
        f"plot_path = Path({rel_plot_path!r})\n",
        "display(Image(filename=str(plot_path)))\n",
    ]
    png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
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

    notebook_path.write_text(json.dumps(nb, indent=1))


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
    paths = resolve_paths(args)
    if not paths.cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {paths.cache_path}. "
            "Pass --cache-path pointing to an existing detuned GPG sweep cache."
        )
    if args.update_notebook_preview and paths.notebook_path is None:
        raise ValueError("--update-notebook-preview requires --notebook-path.")

    load_runtime_dependencies()

    target_inf, target_eps = default_target_window(args.c)
    if args.target_inf is not None:
        target_inf = args.target_inf
    if args.target_eps is not None:
        target_eps = args.target_eps
    threshold = None if target_inf is None else float(target_inf) + float(target_eps)

    key = cache_key(args.p, args.c, style=args.cache_key_style)
    with paths.cache_path.open("rb") as f:
        cache = pickle.load(f)
    if key not in cache["points"]:
        raise KeyError(
            f"No cached point for p={args.p:g}, C={args.c:g}. "
            f"Looked for key {key!r} in {paths.cache_path}."
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = paths.cache_path.with_name(
        paths.cache_path.stem + f".before_parallel_search_{stamp}" + paths.cache_path.suffix
    )
    shutil.copy2(paths.cache_path, backup)
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
        "candidate_dir": str(paths.candidate_dir),
        "cache_path": str(paths.cache_path),
        "target_cache_key": key,
        "prior_cache_paths": [str(path) for path in paths.prior_cache_paths],
    }

    accepted = 0
    attempts_done = 0
    paths.candidate_dir.mkdir(parents=True, exist_ok=True)

    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        while args.max_attempts is None or attempts_done < args.max_attempts:
            with paths.cache_path.open("rb") as f:
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
                save_cache(cache, paths.cache_path)
                accepted += 1
                print(
                    f"ACCEPTED: {old_inf:.6e} -> {best['gpg_infidelity']:.6e} "
                    f"(attempt {best['attempt']})",
                    flush=True,
                )
                metrics = refresh_csv_and_plot(
                    paths,
                    plot_p_values=[float(p) for p in args.plot_p],
                    plot_cooperativities=[float(c) for c in args.plot_c],
                    update_notebook_preview=args.update_notebook_preview,
                )
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
