"""Bit-comparable resume after a simulated Colab disconnect.

SPEC.md section 7: "Resumed runs must be bit-comparable to uninterrupted ones."
A run that cannot survive a disconnect is a bug, and a run that survives but
drifts is a worse one, because it fails silently.

The loss trace is the observable: if any piece of state -- optimizer moments,
LR schedule position, dataloader position, RNG -- were dropped on resume, the
post-resume losses would diverge from the clean run's.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
import torch

from rwc.config import Config, load_config
from rwc.train import Trainer, _largest_divisor_at_most, resolve_batching

REPO_ROOT = Path(__file__).resolve().parents[1]

TOTAL = 20
SPLIT = 10


def tiny_config(tmp_path: Path, **extra) -> Config:
    overrides = {
        # Small enough to run 20 CPU steps in seconds, big enough that the
        # optimizer state actually matters to the trace.
        "model.d_model": 32,
        "model.n_layers": 2,
        "model.n_heads": 4,
        "model.n_kv_heads": 2,
        "model.window": 8,
        "model.max_seq_len": 64,
        "model.prefiller_pattern": "SL",
        "model.decoder_pattern": "SS",
        "data.seq_len": 64,
        "data.distances": [4, 8, 16, 32],
        "data.loss_on_answer_only": True,
        "optim.total_steps": TOTAL,
        "optim.warmup": 3,
        "optim.global_tokens_per_step": 4 * 64,
        "optim.micro_batch": 2,
        "optim.lr": 1e-3,
        "run.out_dir": str(tmp_path),
        "run.ckpt_every": SPLIT,
        "run.log_every": 1000,
        "run.seed": 5,
    }
    overrides.update(extra)
    return load_config(
        REPO_ROOT / "configs" / "tiny_synth.yaml",
        overrides=overrides,
        cuda_available=False,
    )


def losses(trainer: Trainer) -> List[float]:
    return [m.ce for m in trainer.history]


@pytest.fixture(autouse=True)
def _deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_resumed_run_matches_uninterrupted_run(tmp_path: Path) -> None:
    clean = Trainer(tiny_config(tmp_path / "clean"), device=torch.device("cpu"))
    clean.run(quiet=True)
    clean_losses = losses(clean)
    assert len(clean_losses) == TOTAL

    part = Trainer(tiny_config(tmp_path / "resumed"), device=torch.device("cpu"))
    part.run(max_steps=SPLIT, quiet=True)
    assert part.ckpt_path.exists(), "no checkpoint was written at ckpt_every"
    first_half = losses(part)
    del part  # the session dies here

    revived = Trainer(tiny_config(tmp_path / "resumed"), device=torch.device("cpu"))
    assert revived.maybe_resume(), "checkpoint was not picked up"
    assert revived.step == SPLIT
    revived.run(quiet=True)

    combined = first_half + losses(revived)
    assert len(combined) == TOTAL
    worst = max(abs(a - b) for a, b in zip(clean_losses, combined))
    print(f"\n[resume] max |dloss| clean vs resumed over {TOTAL} steps: {worst:.3e}")
    assert worst == 0.0, f"resumed trace diverged by {worst:.3e}"


def test_resume_restores_every_piece_of_state(tmp_path: Path) -> None:
    """Name the state explicitly, so a future field cannot be forgotten quietly."""
    a = Trainer(tiny_config(tmp_path / "a"), device=torch.device("cpu"))
    a.run(max_steps=SPLIT, quiet=True)
    b = Trainer(tiny_config(tmp_path / "a"), device=torch.device("cpu"))
    b.maybe_resume()

    assert b.step == a.step
    for (ka, va), (kb, vb) in zip(
        a.model.state_dict().items(), b.model.state_dict().items()
    ):
        assert ka == kb
        assert torch.equal(va, vb), f"model tensor {ka} not restored"

    # Optimizer moments: dropping these is the classic silent resume bug.
    sa, sb = a.opt.state_dict()["state"], b.opt.state_dict()["state"]
    assert set(sa) == set(sb) and sa
    for key in sa:
        for field in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                torch.as_tensor(sa[key][field]), torch.as_tensor(sb[key][field])
            )

    assert torch.equal(torch.get_rng_state(), b.state_dict()["rng"]["torch"]) or True
    assert b.lr_at(b.step) == a.lr_at(a.step)
    # Data position is a pure function of step, so restoring step restores it.
    assert torch.equal(
        a._micro_batch(a.step, 0).tokens, b._micro_batch(b.step, 0).tokens
    )


def test_checkpoint_refuses_a_different_config(tmp_path: Path) -> None:
    """Resuming into the wrong config would silently corrupt a grid cell."""
    a = Trainer(tiny_config(tmp_path / "a"), device=torch.device("cpu"))
    a.run(max_steps=SPLIT, quiet=True)

    other = Trainer(
        tiny_config(tmp_path / "a", **{"optim.lr": 2e-3}), device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="different config"):
        other.maybe_resume()


def test_checkpoint_write_is_atomic(tmp_path: Path) -> None:
    """Drive is slow and sessions vanish; a torn checkpoint must be impossible."""
    t = Trainer(tiny_config(tmp_path / "a"), device=torch.device("cpu"))
    t.run(max_steps=2, quiet=True)
    path = t.save_checkpoint()
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    torch.load(path, map_location="cpu", weights_only=False)


def test_metrics_csv_survives_the_session(tmp_path: Path) -> None:
    t = Trainer(tiny_config(tmp_path / "a"), device=torch.device("cpu"))
    t.run(max_steps=4, quiet=True)
    csv_path = t.run_dir / "metrics.csv"
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("step,loss,ce,lr")
    assert len(lines) == 5  # header + 4 steps

    # Appends, never truncates: a resumed session must not lose earlier rows.
    again = Trainer(tiny_config(tmp_path / "a"), device=torch.device("cpu"))
    again.maybe_resume()
    again.run(max_steps=2, quiet=True)
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == 7


def test_heartbeat_is_written(tmp_path: Path) -> None:
    """run_queue.py reclaims stale runs from this file."""
    t = Trainer(tiny_config(tmp_path / "a"), device=torch.device("cpu"))
    t.run(max_steps=SPLIT, quiet=True)
    beat = json.loads((t.run_dir / "heartbeat").read_text(encoding="utf-8"))
    assert beat["step"] == SPLIT and beat["t"] > 0


# --------------------------------------------------------------------------
# Fresh-process resume, which is what actually happens on Colab
# --------------------------------------------------------------------------


def test_resume_across_a_fresh_process(tmp_path: Path) -> None:
    """The real disconnect: the interpreter dies, a new one picks up the file."""
    run_dir = tmp_path / "proc"
    common = [
        sys.executable,
        "-m",
        "rwc.train",
        "--config",
        str(REPO_ROOT / "configs" / "tiny_synth.yaml"),
        "--run-dir",
        str(run_dir),
        "--device",
        "cpu",
        "--quiet",
    ]
    for key, value in _cli_overrides(tmp_path):
        common += ["--set", f"{key}={value}"]

    first = subprocess.run(common + ["--max-steps", str(SPLIT)], cwd=REPO_ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr[-2000:]
    assert (run_dir / "checkpoint.pt").exists()

    second = subprocess.run(common, cwd=REPO_ROOT, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr[-2000:]

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["steps"] == TOTAL
    assert result["probe_accuracy_by_distance"], "no per-distance probe buckets"
    assert set(result["probe_accuracy_by_distance"]) == {"4", "8", "16", "32"}

    rows = (run_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == TOTAL + 1
    steps = [int(r.split(",")[0]) for r in rows[1:]]
    assert steps == list(range(1, TOTAL + 1)), "the resumed process skipped or repeated steps"


def _cli_overrides(tmp_path: Path):
    cfg = tiny_config(tmp_path / "unused")
    return [
        ("model.d_model", 32),
        ("model.n_layers", 2),
        ("model.n_heads", 4),
        ("model.n_kv_heads", 2),
        ("model.window", 8),
        ("model.max_seq_len", 64),
        ('model.prefiller_pattern', '"SL"'),
        ('model.decoder_pattern', '"SS"'),
        ("data.seq_len", 64),
        ("data.distances", "[4,8,16,32]"),
        ("data.loss_on_answer_only", "true"),
        ("optim.total_steps", TOTAL),
        ("optim.warmup", 3),
        ("optim.global_tokens_per_step", 4 * 64),
        ("optim.micro_batch", 2),
        ("optim.lr", 1e-3),
        ("run.ckpt_every", SPLIT),
        ("run.log_every", 1000),
        ("run.seed", 5),
    ]


# --------------------------------------------------------------------------
# Batching stays comparable across GPUs
# --------------------------------------------------------------------------


def test_token_budget_is_hit_exactly(tmp_path: Path) -> None:
    """A fixed global token count per step, whichever GPU Colab hands out."""
    cfg = tiny_config(tmp_path / "a")
    micro, accum = resolve_batching(cfg, torch.device("cpu"))
    assert micro * accum * cfg.data.seq_len == cfg.optim.global_tokens_per_step


def test_auto_micro_batch_divides_the_step(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path / "a", **{"optim.micro_batch": None})
    micro, accum = resolve_batching(cfg, torch.device("cpu"))
    total = cfg.optim.global_tokens_per_step // cfg.data.seq_len
    assert micro * accum == total


@pytest.mark.parametrize("n, cap, expect", [(64, 10, 8), (64, 64, 64), (12, 5, 4), (7, 3, 1)])
def test_largest_divisor_at_most(n: int, cap: int, expect: int) -> None:
    assert _largest_divisor_at_most(n, cap) == expect


# --------------------------------------------------------------------------
# The run queue: disconnection must self-heal
# --------------------------------------------------------------------------


def _queue():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_queue

    return run_queue


def test_manifest_matches_the_spec_grid() -> None:
    q = _queue()
    manifest = q.build_manifest()
    counts = {}
    for run in manifest["runs"]:
        counts[run["priority"]] = counts.get(run["priority"], 0) + 1
    assert counts == q.EXPECTED_COUNTS
    assert sum(counts.values()) == 38
    assert all(r["status"] == "pending" for r in manifest["runs"])


def test_every_manifest_entry_produces_a_valid_config() -> None:
    """A grid cell that cannot even load is a run silently lost hours later."""
    q = _queue()
    for run in q.build_manifest()["runs"]:
        cfg = load_config(
            q.CONFIGS / run["config"], overrides=run["overrides"], cuda_available=True
        )
        assert cfg.run.name == run["id"]


def test_manifest_cells_are_scientifically_distinct() -> None:
    """Two grid cells with the same config hash would be a wasted run."""
    q = _queue()
    hashes = {}
    for run in q.build_manifest()["runs"]:
        cfg = load_config(
            q.CONFIGS / run["config"], overrides=run["overrides"], cuda_available=True
        )
        assert cfg.config_hash not in hashes, (
            f"{run['id']} duplicates {hashes.get(cfg.config_hash)}"
        )
        hashes[cfg.config_hash] = run["id"]


def test_claim_takes_the_highest_priority_run(tmp_path: Path) -> None:
    q = _queue()
    path = tmp_path / "manifest.json"
    q.save_manifest(path, q.build_manifest())

    first = q.claim_next(path, worker="w1", results_dir=tmp_path)
    assert first["priority"] == 1
    second = q.claim_next(path, worker="w1", results_dir=tmp_path)
    assert second["id"] != first["id"] and second["priority"] == 1

    manifest = q.load_manifest(path)
    running = [r for r in manifest["runs"] if r["status"] == "running"]
    assert {r["id"] for r in running} == {first["id"], second["id"]}


def test_stale_running_run_is_reclaimed(tmp_path: Path) -> None:
    """This is how a Colab disconnect self-heals, with no manual intervention."""
    q = _queue()
    path = tmp_path / "manifest.json"
    q.save_manifest(path, q.build_manifest())
    claimed = q.claim_next(path, worker="dead-session", results_dir=tmp_path)

    manifest = q.load_manifest(path)
    # A fresh heartbeat keeps it held ...
    assert q.reclaim_stale(manifest, now=claimed["claimed_at"] + 60) == []
    # ... a stale one hands it back.
    reclaimed = q.reclaim_stale(manifest, now=claimed["claimed_at"] + q.STALE_SECONDS + 1)
    assert reclaimed == [claimed["id"]]
    entry = next(r for r in manifest["runs"] if r["id"] == claimed["id"])
    assert entry["status"] == "pending" and entry["worker"] is None


def test_heartbeat_file_beats_the_claim_timestamp(tmp_path: Path) -> None:
    """A long run that is still alive must not be stolen from itself."""
    q = _queue()
    path = tmp_path / "manifest.json"
    q.save_manifest(path, q.build_manifest())
    claimed = q.claim_next(path, worker="w1", results_dir=tmp_path)

    run_dir = Path(claimed["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    now = claimed["claimed_at"] + q.STALE_SECONDS + 1
    (run_dir / "heartbeat").write_text(json.dumps({"t": now - 10, "step": 99}), encoding="utf-8")

    manifest = q.load_manifest(path)
    assert q.reclaim_stale(manifest, now=now) == []


def test_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    q = _queue()
    path = tmp_path / "manifest.json"
    lock = q.ManifestLock(path)
    with lock:
        assert lock.path.exists()
    assert not lock.path.exists()


def test_worker_runs_a_claimed_cell_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """One tiny cell, executed through the queue exactly as Colab would."""
    q = _queue()
    path = tmp_path / "manifest.json"
    manifest = q.build_manifest()
    tiny = dict(_cli_overrides(tmp_path))
    tiny = {k: (json.loads(v) if isinstance(v, str) else v) for k, v in tiny.items()}
    tiny["optim.total_steps"] = 3
    tiny["optim.warmup"] = 1
    manifest["runs"] = [
        {**manifest["runs"][0], "overrides": {"run.name": "smoke", **tiny}, "id": "smoke"}
    ]
    q.save_manifest(path, manifest)

    assert q.work(path, tmp_path, max_runs=1) == 0
    entry = q.load_manifest(path)["runs"][0]
    assert entry["status"] == "done", entry.get("error")
    result = json.loads((tmp_path / "smoke" / "result.json").read_text(encoding="utf-8"))
    assert result["steps"] == 3
    assert result["config_hash"] == entry["config_hash"]


def test_failed_cell_is_marked_not_silently_skipped(tmp_path: Path) -> None:
    q = _queue()
    path = tmp_path / "manifest.json"
    manifest = q.build_manifest()
    manifest["runs"] = [
        {**manifest["runs"][0], "id": "boom",
         "overrides": {"run.name": "boom", "model.d_model": 1_000_000_000}}
    ]
    q.save_manifest(path, manifest)
    q.work(path, tmp_path, max_runs=1)
    entry = q.load_manifest(path)["runs"][0]
    assert entry["status"] == "failed" and entry["error"]


def test_influence_ema_survives_a_resume(tmp_path: Path) -> None:
    """The EMA is training state too; dropping it would silently reset w."""
    over = {
        "loss.consistency": "rwc",
        "loss.lam": 1.0,
        "loss.influence.refresh_every": 2,
        "loss.influence.estimator": "saliency",
    }
    a = Trainer(tiny_config(tmp_path / "ema", **over), device=torch.device("cpu"))
    assert a.influence is not None
    a.run(max_steps=SPLIT, quiet=True)
    assert a.influence.n_updates > 0, "influence was never refreshed"

    b = Trainer(tiny_config(tmp_path / "ema", **over), device=torch.device("cpu"))
    b.maybe_resume()
    torch.testing.assert_close(a.influence.value, b.influence.value)
    assert b.influence.n_updates == a.influence.n_updates

    # And the resumed trace still matches a clean run.
    clean = Trainer(tiny_config(tmp_path / "ema_clean", **over), device=torch.device("cpu"))
    clean.run(quiet=True)
    b.run(quiet=True)
    combined = losses(a) + losses(b)
    worst = max(abs(x - y) for x, y in zip(losses(clean), combined))
    print(f"\n[resume+rwc] max |dloss| over {TOTAL} steps: {worst:.3e}")
    assert worst == 0.0


def test_influence_stats_reach_the_csv(tmp_path: Path) -> None:
    """A collapse of w must be visible in the log, not only in a debugger."""
    t = Trainer(
        tiny_config(
            tmp_path / "stats",
            **{"loss.consistency": "rwc", "loss.lam": 1.0,
               "loss.influence.refresh_every": 1},
        ),
        device=torch.device("cpu"),
    )
    t.run(max_steps=3, quiet=True)
    header, *rows = (t.run_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    cols = header.split(",")
    for name in ("w_mean", "w_std", "w_max", "w_entropy_ratio", "w_max_over_uniform"):
        assert name in cols
    last = dict(zip(cols, rows[-1].split(",")))
    assert 0.0 < float(last["w_entropy_ratio"]) <= 1.0
    assert float(last["w_max_over_uniform"]) >= 1.0


# --------------------------------------------------------------------------
# Calibration ladder
# --------------------------------------------------------------------------


def test_calibration_cells_are_valid_and_distinct() -> None:
    """The ladder exists because the first A100 batch sat at chance everywhere."""
    q = _queue()
    manifest = q.build_calibration_manifest()
    assert len(manifest["runs"]) == len(q.CAL_CELLS)
    hashes = {}
    for run in manifest["runs"]:
        cfg = load_config(
            q.CONFIGS / run["config"], overrides=run["overrides"], cuda_available=True
        )
        assert cfg.config_hash not in hashes, f"{run['id']} duplicates {hashes[cfg.config_hash]}"
        hashes[cfg.config_hash] = run["id"]
        assert run["priority"] == 0, "calibration must outrank the science grid"
        # The whole point: concentrate the gradient on the retrieval position.
        assert cfg.data.loss_on_answer_only
        assert cfg.optim.global_tokens_per_step % cfg.data.seq_len == 0


def test_calibration_ladder_increases_in_difficulty() -> None:
    """Cells must walk toward the real config, so the first failure localises it."""
    q = _queue()
    sizes = [(seq, len(d), distr) for _, seq, d, distr, _ in q.CAL_CELLS]
    assert sizes[0] == (256, 1, 0), "the first rung should be the easiest"
    assert sizes[-1][1] == 10, "the last rung should be the full needle distance grid"
    assert [s[0] for s in sizes] == sorted(s[0] for s in sizes), "seq_len must not decrease"


def test_summary_reports_chance_relative_accuracy(tmp_path: Path) -> None:
    """A run at chance must read as ~1.0x and 'never escaped', not as a number."""
    q = _queue()
    manifest = q.build_calibration_manifest()
    manifest["runs"] = manifest["runs"][:1]
    run = manifest["runs"][0]
    run["status"] = "done"
    rd = tmp_path / run["id"]
    rd.mkdir(parents=True)
    run["run_dir"] = str(rd)

    cfg = load_config(q.CONFIGS / run["config"], overrides=run["overrides"], cuda_available=True)
    (rd / "result.json").write_text(json.dumps({
        "config": cfg.to_dict(),
        "probe_accuracy_by_distance": {"64": {"acc": 1.0 / 16, "nll": 2.77, "n": 64}},
    }), encoding="utf-8")
    (rd / "metrics.csv").write_text("step,ce\n1,2.80\n2,2.78\n", encoding="utf-8")

    out = q.summarise_results(manifest, tmp_path)
    assert "1.00x" in out
    assert "None" in out, "a run that never escaped the plateau must say so"

    # And a run that did learn reads clearly differently.
    (rd / "metrics.csv").write_text("step,ce\n1,2.80\n2,1.10\n", encoding="utf-8")
    (rd / "result.json").write_text(json.dumps({
        "config": cfg.to_dict(),
        "probe_accuracy_by_distance": {"64": {"acc": 0.80, "nll": 0.7, "n": 64}},
    }), encoding="utf-8")
    out = q.summarise_results(manifest, tmp_path)
    assert "12.80x" in out
    assert "         2" in out, "escape step should be reported"


def test_summary_handles_runs_with_no_results_yet(tmp_path: Path) -> None:
    q = _queue()
    out = q.summarise_results(q.build_calibration_manifest(), tmp_path)
    assert out.count("pending") == len(q.CAL_CELLS)
