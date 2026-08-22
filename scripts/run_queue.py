"""Colab-resilient experiment orchestrator.

SPEC.md section 7: open the notebook, run one cell, and whatever is unfinished
gets picked up and continued. Disconnection is the normal case, not an error
path -- a ``running`` run whose heartbeat has gone stale is reclaimed as
``pending``, which is how a dead session self-heals.

``build_manifest`` encodes the collapse study in priority order, so if time
runs out the scientifically important runs are already done.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwc.config import load_config  # noqa: E402
from rwc.train import Trainer, _atomic_write_text, gpu_name  # noqa: E402

MANIFEST_VERSION = 1
STALE_SECONDS = 20 * 60  # SPEC.md section 7
LOCK_STALE_SECONDS = 120

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

# Needle at W=64 with L=4: stacked local reach is 4*63 = 252, so distances
# 8..192 are solvable by local attention and 256..384 require memory.
SYNTH_BASE: Dict[str, Any] = {
    "data.task": "needle",
    "data.distances": [8, 16, 32, 48, 64, 96, 128, 192, 256, 384],
    # Under full-sequence CE the answer is 1 scored position of 511, so the
    # gradient carrying the task is ~0.2% of the loss and the model never leaves
    # the ln(n_values) plateau. Confirmed by the calibration ladder.
    "data.loss_on_answer_only": True,
}

LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0)
SHARINGS = ("shared", "separate")


def build_manifest() -> Dict[str, Any]:
    """The collapse study.

    Read-weighted consistency, the masking ablations and the seed/scale groups
    that served them are gone: the hypothesis was refuted by our own C1 and C3
    measurements, and the collapse mechanism explains why it could never have
    worked -- re-weighting WHICH dimensions to match is irrelevant once the loss
    has switched the memory channel off.
    """
    runs: List[Dict[str, Any]] = []

    def add(priority: int, group: str, name: str, **over: Any) -> None:
        runs.append({
            "id": name, "priority": priority, "group": group,
            "config": "tiny_synth.yaml",
            "overrides": {"run.name": name, **over},
        })

    # 1 -- the collapse, and its dependence on parameter sharing. Reproduces the
    # published lambda anomaly: shared severs between 0.03 and 0.1, separate
    # between 0.1 and 0.3.
    for sharing in SHARINGS:
        for lam in LAMBDAS:
            over = {"model.sharing": sharing}
            if lam > 0:
                over |= {"loss.consistency": "uniform", "loss.lam": lam}
            add(1, "collapse", f"p1_{sharing}_lam{lam}", **SYNTH_BASE, **over)

    # 2 -- the intervention. propagation_weight prices the recurrence being
    # switched off; measured to take rho from 0.002 to 0.680 and to turn the
    # lambda=0.1 plateau into learning at 3/3 seeds.
    # Seeds start at 1: the seed-0 control is p1_shared_lam0.1, an identical
    # config, and emitting it twice would train the same cell twice.
    for seed in (1, 2, 3, 4, 5):
        for tag, extra in (("ctl", {}), ("pp", {"loss.propagation_weight": 0.1})):
            add(2, "propagation", f"p2_{tag}_s{seed}", **SYNTH_BASE,
                **{"loss.consistency": "uniform", "loss.lam": 0.1,
                   "run.seed": seed}, **extra)

    # 3 -- reference points. SWA fixes the floor (no memory at all); full
    # attention fixes the ceiling and shows the task is solvable.
    add(3, "baselines", "p3_swa", **SYNTH_BASE, **{"model.kind": "swa"})
    add(3, "baselines", "p3_full", **SYNTH_BASE, **{"model.kind": "full"})

    ids = [r["id"] for r in runs]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate run ids in the manifest")

    for run in runs:
        run.update(status="pending", run_dir=None, claimed_at=None,
                   worker=None, error=None, config_hash=None)
    return {"version": MANIFEST_VERSION, "created": time.time(), "runs": runs}


# ---------------------------------------------------------------------------
# Manifest I/O with a lock
# ---------------------------------------------------------------------------


class ManifestLock:
    """Exclusive-create lock file, with a staleness escape hatch.

    Colab normally runs one worker, but a reconnect can briefly overlap with the
    session it replaced, and that is exactly when two writers would collide.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.with_suffix(".lock")

    def __enter__(self) -> "ManifestLock":
        deadline = time.time() + LOCK_STALE_SECONDS
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(time.time()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                age = time.time() - self.path.stat().st_mtime
                if age > LOCK_STALE_SECONDS or time.time() > deadline:
                    self.path.unlink(missing_ok=True)
                    continue
                time.sleep(1.0)

    def __exit__(self, *exc: Any) -> None:
        self.path.unlink(missing_ok=True)


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    _atomic_write_text(Path(path), json.dumps(manifest, indent=2))


def reclaim_stale(manifest: Dict[str, Any], *, now: Optional[float] = None) -> List[str]:
    """Return ``running`` runs whose heartbeat has gone stale to ``pending``."""
    now = now if now is not None else time.time()
    reclaimed = []
    for run in manifest["runs"]:
        if run["status"] != "running":
            continue
        beat = _heartbeat_time(run) or run.get("claimed_at") or 0.0
        if now - beat > STALE_SECONDS:
            run.update(status="pending", worker=None, claimed_at=None)
            reclaimed.append(run["id"])
    return reclaimed


def _heartbeat_time(run: Dict[str, Any]) -> Optional[float]:
    if not run.get("run_dir"):
        return None
    path = Path(run["run_dir"]) / "heartbeat"
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["t"])
    except Exception:
        return path.stat().st_mtime


def claim_next(
    manifest_path: Path, *, worker: str, results_dir: Path
) -> Optional[Dict[str, Any]]:
    """Atomically take the highest-priority pending run, or None."""
    with ManifestLock(manifest_path):
        manifest = load_manifest(manifest_path)
        reclaimed = reclaim_stale(manifest)
        if reclaimed:
            print(f"[queue] reclaimed stale runs: {', '.join(reclaimed)}", flush=True)

        pending = [r for r in manifest["runs"] if r["status"] == "pending"]
        if not pending:
            save_manifest(manifest_path, manifest)
            return None
        pending.sort(key=lambda r: (r["priority"], r["id"]))
        run = pending[0]
        run.update(
            status="running",
            worker=worker,
            claimed_at=time.time(),
            run_dir=str(results_dir / run["id"]),
        )
        save_manifest(manifest_path, manifest)
        return dict(run)


def mark(manifest_path: Path, run_id: str, **fields: Any) -> None:
    with ManifestLock(manifest_path):
        manifest = load_manifest(manifest_path)
        for run in manifest["runs"]:
            if run["id"] == run_id:
                run.update(fields)
        save_manifest(manifest_path, manifest)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def execute(run: Dict[str, Any], *, results_dir: Path) -> Dict[str, Any]:
    cfg = load_config(CONFIGS / run["config"], overrides=run["overrides"])
    run_dir = Path(run["run_dir"] or results_dir / run["id"])
    _retire_stale_checkpoint(run_dir, cfg.config_hash)
    trainer = Trainer(cfg, run_dir=run_dir)
    trainer.maybe_resume()  # zero manual intervention after a disconnect
    result = trainer.run()
    _atomic_write_text(run_dir / "result.json", json.dumps(result, indent=2))
    return result


def _retire_stale_checkpoint(run_dir: Path, want_hash: str) -> None:
    """Move aside a checkpoint written by a different config, and start clean.

    Re-running with --init after a config change gives every cell a new hash
    while its run directory still holds the *old* checkpoint. Trainer.maybe_resume
    correctly refuses to load it -- resuming into the wrong config would silently
    corrupt a grid cell -- but the run then fails instead of starting fresh, and
    the stale result.json stays on disk looking like a completed run. That
    happened to all three priority-1 cells on the first headline batch: their
    artifacts were from the previous commit and were mistaken for new results.

    The old files are renamed rather than deleted, so nothing is destroyed.
    """
    ckpt = run_dir / "checkpoint.pt"
    if not ckpt.exists():
        return
    try:
        import torch

        have = torch.load(ckpt, map_location="cpu", weights_only=False).get("config_hash")
    except Exception:
        have = None
    if have == want_hash:
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    retired = run_dir / f"stale-{have or 'unreadable'}-{stamp}"
    retired.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint.pt", "result.json", "metrics.csv", "heartbeat"):
        src = run_dir / name
        if src.exists():
            src.rename(retired / name)
    print(
        f"[queue] {run_dir.name}: checkpoint was written by config "
        f"{(have or '?')[:12]}, this cell wants {want_hash[:12]}. "
        f"Retired the old artifacts to {retired.name}/ and starting clean.",
        flush=True,
    )


def work(manifest_path: Path, results_dir: Path, *, max_runs: Optional[int] = None) -> int:
    worker = f"{socket.gethostname()}:{os.getpid()}"
    print(f"[queue] worker {worker} on {gpu_name()}", flush=True)
    done = 0
    while max_runs is None or done < max_runs:
        run = claim_next(manifest_path, worker=worker, results_dir=results_dir)
        if run is None:
            print("[queue] nothing pending. All done.", flush=True)
            break
        print(f"\n[queue] === p{run['priority']} {run['id']} ===", flush=True)
        try:
            result = execute(run, results_dir=results_dir)
            mark(manifest_path, run["id"], status="done", error=None,
                 config_hash=result["config_hash"])
        except KeyboardInterrupt:
            mark(manifest_path, run["id"], status="pending", worker=None)
            raise
        except Exception:
            err = traceback.format_exc()
            print(err, file=sys.stderr, flush=True)
            mark(manifest_path, run["id"], status="failed", error=err[-4000:])
        done += 1
    return 0


def summarise(manifest: Dict[str, Any]) -> str:
    counts: Dict[str, int] = {}
    for run in manifest["runs"]:
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    order = ("done", "running", "pending", "failed")
    return "  ".join(f"{k}={counts.get(k, 0)}" for k in order)


def summarise_results(manifest: Dict[str, Any], results_dir: Path) -> str:
    """Probe accuracy against chance, plus where CE left the plateau.

    Answers the only question calibration asks -- did this cell learn to
    retrieve at all -- without needing a notebook to look at the JSON.
    """
    lines = [
        f"{'run':<16} {'status':<8} {'mean/chance':>12} {'escaped':>9}  per-distance accuracy",
        "-" * 100,
    ]
    for run in sorted(manifest["runs"], key=lambda r: (r["priority"], r["id"])):
        rd = Path(run["run_dir"]) if run["run_dir"] else results_dir / run["id"]
        result_path = rd / "result.json"
        if not result_path.exists():
            lines.append(f"{run['id']:<16} {run['status']:<8} {'-':>12} {'-':>9}")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        buckets = result.get("probe_accuracy_by_distance", {})
        chance = 1.0 / result["config"]["data"]["n_values"]
        # Mean over buckets, not max: at 64 samples per bucket a single bucket
        # drifts +/- 0.04 by chance alone, which is larger than the effect.
        mean = (
            sum(v["acc"] for v in buckets.values()) / len(buckets) if buckets else 0.0
        )
        escaped = _escape_step(rd, chance)
        per = "  ".join(f"D{d}:{v['acc']:.2f}" for d, v in buckets.items())
        lines.append(
            f"{run['id']:<16} {run['status']:<8} {mean / chance:>11.2f}x "
            f"{str(escaped):>9}  {per}"
        )
    lines.append("")
    lines.append(
        "mean/chance <= ~1.2 means the cell never learned to retrieve. 'escaped' is "
        "the step at which CE\nfirst fell clearly below the ln(n_values) plateau; "
        "None means it never did."
    )
    return "\n".join(lines)


def _escape_step(run_dir: Path, chance: float) -> Optional[int]:
    """First step where CE drops clearly below the plateau (a phase transition)."""
    import csv as _csv
    import math as _math

    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    plateau = -_math.log(chance)
    with path.open(encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            try:
                if float(row["ce"]) < plateau * 0.95:
                    return int(row["step"])
            except (ValueError, KeyError):
                continue
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="results/manifest.json")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--init", action="store_true", help="(re)create the manifest")
    p.add_argument("--status", action="store_true", help="print status and exit")
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="use the calibration ladder instead of the SPEC grid",
    )
    p.add_argument(
        "--summary", action="store_true", help="print probe accuracy vs chance and exit"
    )
    args = p.parse_args(argv)

    manifest_path = Path(args.manifest)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.init or not manifest_path.exists():
        builder = build_calibration_manifest if args.calibrate else build_manifest
        save_manifest(manifest_path, builder())
        print(f"[queue] wrote manifest with "
              f"{len(load_manifest(manifest_path)['runs'])} runs", flush=True)

    if args.summary:
        print(summarise_results(load_manifest(manifest_path), results_dir))
        return 0

    if args.status:
        manifest = load_manifest(manifest_path)
        print(summarise(manifest))
        for run in sorted(manifest["runs"], key=lambda r: (r["priority"], r["id"])):
            print(f"  p{run['priority']} {run['status']:<8} {run['id']}")
        return 0

    return work(manifest_path, results_dir, max_runs=args.max_runs)


if __name__ == "__main__":
    raise SystemExit(main())
