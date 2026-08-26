"""The archived evidence must keep reproducing the published numbers.

`evidence/` holds the result.json of every run behind the write-up, without the
checkpoints. If a refactor changes how a cell is scored, or how arms are
assembled, the headline silently moves. These tests fail instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "evidence"


def _verify_module():
    spec = importlib.util.spec_from_file_location("_verify", EVIDENCE / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not (EVIDENCE / "verify.py").exists() or
    not list(EVIDENCE.glob("cpu_study/runs/*/result.json")),
    reason="evidence archive not present",
)


# --------------------------------------------------------------------------
# Fisher exact
# --------------------------------------------------------------------------
#
# The first implementation summed the hypergeometric tail from
# `col1 - (n1 - k1)`, which is not the lower end of the support. Whenever arm 1
# was perfect that made the range EMPTY and returned p = 0.0 -- for exactly the
# 10/10 vs 3/10 table the headline rests on.


@pytest.mark.parametrize(
    "k1,n1,k2,n2",
    [
        (10, 10, 3, 10),   # the headline table: arm 1 perfect
        (8, 8, 2, 7),      # perfect arm, unequal n
        (5, 10, 5, 10),    # no effect
        (0, 10, 0, 10),    # both empty
        (0, 5, 4, 5),      # effect in the opposite direction
        (1, 3, 0, 3),      # tiny n
        (7, 12, 2, 9),
    ],
)
def test_fisher_matches_scipy(k1: int, n1: int, k2: int, n2: int) -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    mod = _verify_module()
    _, expected = scipy_stats.fisher_exact(
        [[k1, n1 - k1], [k2, n2 - k2]], alternative="greater"
    )
    assert mod.fisher_greater(k1, n1, k2, n2) == pytest.approx(expected, rel=1e-9)


def test_fisher_is_not_zero_when_one_arm_is_perfect() -> None:
    """The exact shape of the bug: a real p-value, never 0."""
    mod = _verify_module()
    p = mod.fisher_greater(10, 10, 3, 10)
    assert 0.0 < p < 0.01


# --------------------------------------------------------------------------
# The headline
# --------------------------------------------------------------------------


def test_headline_reproduces_from_the_archive() -> None:
    mod = _verify_module()
    cpu = mod.load("cpu_study/runs/*/result.json")
    plain, res, _ = mod.collect_arms(cpu)

    assert len(plain) == 10, f"write-rule arm has {len(plain)} seeds, expected 10"
    assert len(res) == 10, f"identity arm has {len(res)} seeds, expected 10"
    assert sorted(plain) == list(range(10))
    assert sorted(res) == list(range(10))

    kp = sum(any(mod.learned(q) for _, q in c) for c in plain.values())
    kr = sum(any(mod.learned(q) for _, q in c) for c in res.values())
    assert (kp, kr) == (3, 10), f"headline moved: {kp}/10 vs {kr}/10"

    p = mod.fisher_greater(kr, 10, kp, 10)
    assert p == pytest.approx(1.55e-3, rel=0.02), f"p-value moved to {p:.3e}"


def test_duplicate_seeds_agree() -> None:
    """Two cells were run twice under different names; they must still agree."""
    mod = _verify_module()
    cpu = mod.load("cpu_study/runs/*/result.json")
    plain, res, notes = mod.collect_arms(cpu)
    assert notes, "expected the known duplicate cells to be detected"
    assert not any("DISAGREE" in n for n in notes), "\n".join(notes)


def test_every_archived_cell_carries_its_own_config() -> None:
    """Reproducibility depends on this: no cell may need an external config."""
    mod = _verify_module()
    for name, p in mod.load("cpu_study/runs/*/result.json").items():
        cfg = p.get("config", {})
        for key in ("model", "data", "optim", "loss", "run"):
            assert key in cfg, f"{name} is missing config.{key}"


# --------------------------------------------------------------------------
# The stratified test
# --------------------------------------------------------------------------
#
# It exists in two places -- scripts/cpu_ladder.py (which produces the numbers)
# and evidence/verify.py (which recomputes them from the archive). If those ever
# disagree, the paper and its verification would quietly report different
# p-values. These pin both to scipy and to each other.


def _ladder_module():
    spec = importlib.util.spec_from_file_location(
        "_ladder", REPO / "scripts" / "cpu_ladder.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ladder"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "strata",
    [
        [(5, 5, 1, 5)],
        [(10, 10, 3, 10)],
        [(5, 5, 1, 5), (5, 5, 3, 5)],
        [(5, 5, 1, 5), (5, 5, 3, 5), (5, 5, 2, 5)],
        [(10, 10, 3, 10), (10, 10, 6, 10), (5, 5, 2, 5), (5, 5, 0, 5), (5, 5, 0, 5)],
        [(3, 5, 3, 5), (2, 4, 2, 4)],
    ],
)
def test_stratified_test_agrees_between_producer_and_verifier(strata) -> None:
    mod = _verify_module()
    ladder = _ladder_module()
    assert mod.stratified_exact_p(strata) == pytest.approx(
        ladder.stratified_exact_p(strata), rel=1e-12
    )


@pytest.mark.parametrize("k1,n1,k2,n2", [(5, 5, 1, 5), (10, 10, 3, 10), (3, 5, 3, 5)])
def test_single_stratum_reduces_to_fisher(k1, n1, k2, n2) -> None:
    """One stratum must give exactly Fisher's one-sided exact p."""
    scipy_stats = pytest.importorskip("scipy.stats")
    mod = _verify_module()
    _, expected = scipy_stats.fisher_exact(
        [[k1, n1 - k1], [k2, n2 - k2]], alternative="greater"
    )
    assert mod.stratified_exact_p([(k1, n1, k2, n2)]) == pytest.approx(
        expected, rel=1e-9
    )


def test_stratifying_is_not_the_same_as_pooling() -> None:
    """The reason this function exists at all.

    Pooling strata into one 2x2 treats different model sizes as exchangeable
    repeats of one condition and reports a smaller p than the data support.
    """
    mod = _verify_module()
    strata = [(5, 5, 1, 5), (5, 5, 3, 5), (5, 5, 2, 5)]
    stratified = mod.stratified_exact_p(strata)
    pooled = mod.fisher_greater(
        sum(s[0] for s in strata), sum(s[1] for s in strata),
        sum(s[2] for s in strata), sum(s[3] for s in strata),
    )
    assert stratified > pooled, "stratifying must not be more permissive than pooling"


@pytest.mark.skipif(
    not list((REPO / "ladder_runs").glob("*/result.json")),
    reason="width ladder not present",
)
def test_width_ladder_headline_holds() -> None:
    """35/35 vs 11/35 across five rungs. Fails loudly if scoring drifts."""
    mod = _verify_module()
    rows = list(mod.load("*/result.json", REPO / "ladder_runs").values())
    if len(rows) < 70:
        pytest.skip(f"ladder incomplete ({len(rows)}/70)")
    strata = []
    for d in sorted({r["d_model"] for r in rows}):
        at = [r for r in rows if r["d_model"] == d]
        kr = sum(mod.learned(r) for r in at if r["arm"] == "res")
        nr = sum(1 for r in at if r["arm"] == "res")
        kp = sum(mod.learned(r) for r in at if r["arm"] == "plain")
        npl = sum(1 for r in at if r["arm"] == "plain")
        strata.append((kr, nr, kp, npl))
    assert sum(s[0] for s in strata) == 35, "identity arm is no longer 35/35"
    assert sum(s[1] for s in strata) == 35
    assert sum(s[2] for s in strata) == 11, "write-rule arm moved off 11/35"
    p = mod.stratified_exact_p(strata)
    assert p == pytest.approx(8.80e-11, rel=0.05), f"stratified p moved to {p:.2e}"
