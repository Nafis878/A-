"""The three probe tasks, generated on the fly from a seeded RNG.

SPEC.md section 6. Every sample carries the **retrieval distance** it was built
at, and results are reported per distance bucket rather than averaged, because
the slope against distance is the finding.

Generation is deterministic in ``(seed, step)``. That is the whole reason the
dataloader position is a single integer, which in turn is why resume can be
exact rather than approximate.

A correction to the SPEC, flagged in README assumption 16: the SPEC describes
delayed-recall as ``D > W``, "provably unreachable through local sliding-window
attention". That holds for one sliding layer. With ``L`` stacked sliding layers
information propagates ``L*(W-1)`` positions, so the provable bound is
``D > L*(W-1)``. That stricter condition is what is enforced here; using the
looser one would have let local attention solve part of the task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence

import torch
from torch import Tensor

from ..config import ConfigError, DataConfig, ModelConfig

__all__ = ["SyntheticVocab", "ProbeBatch", "SyntheticGenerator", "TASK_ALIASES"]

# carrier_copy is delayed-recall with a known read structure imposed on the
# model side (ModelConfig.carrier_gain_seed); the sequences are identical.
TASK_ALIASES = {"carrier_copy": "delayed_recall"}


@dataclass(frozen=True)
class SyntheticVocab:
    """Token id layout. Ids are contiguous so the tables stay small."""

    n_keys: int
    n_values: int
    n_filler: int

    PAD: int = 0
    SEP: int = 1
    QUERY: int = 2
    ANS: int = 3
    N_SPECIAL: int = 4

    @property
    def key_lo(self) -> int:
        return self.N_SPECIAL

    @property
    def value_lo(self) -> int:
        return self.key_lo + self.n_keys

    @property
    def filler_lo(self) -> int:
        return self.value_lo + self.n_values

    @property
    def size(self) -> int:
        return self.filler_lo + self.n_filler

    @classmethod
    def from_config(cls, data: DataConfig) -> "SyntheticVocab":
        return cls(n_keys=data.n_keys, n_values=data.n_values, n_filler=data.n_filler)


class ProbeBatch(NamedTuple):
    tokens: Tensor  # (B, T) int64
    targets: Tensor  # (B, T) int64, next-token targets; -100 where undefined
    answer_idx: Tensor  # (B,) index into targets holding the retrieved token
    distance: Tensor  # (B,) retrieval distance, the figure x-axis
    loss_mask: Tensor  # (B, T) bool, positions contributing to CE


class SyntheticGenerator:
    """On-the-fly probe generation, deterministic in ``(seed, step)``."""

    IGNORE_INDEX = -100

    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        *,
        seed: int = 0,
        n_distractors: int = 4,
    ) -> None:
        self.task = TASK_ALIASES.get(data.task or "", data.task or "")
        if self.task not in ("needle", "two_hop", "delayed_recall"):
            raise ConfigError(f"unsupported synthetic task {data.task!r}")
        self.data = data
        self.model = model
        self.seed = seed
        self.n_distractors = n_distractors
        self.vocab = SyntheticVocab.from_config(data)
        self.seq_len = data.seq_len
        self.distances = list(data.distances)

        if self.vocab.size > model.vocab_size:
            raise ConfigError(
                f"synthetic alphabet needs {self.vocab.size} ids but "
                f"model.vocab_size={model.vocab_size}"
            )
        # The query block occupies the last 3 positions; two-hop needs room for
        # two planted pairs before it.
        if self.task == "two_hop" and min(self.distances) < 5:
            raise ConfigError(
                f"two_hop needs distances >= 5 so the second hop fits strictly "
                f"between the first and the query; got {min(self.distances)}"
            )
        need = max(self.distances) + 2
        if need >= self.seq_len - 3:
            raise ConfigError(
                f"max distance {max(self.distances)} does not fit seq_len={self.seq_len}"
            )
        if self.task == "delayed_recall":
            reach = model.local_reach
            bad = [d for d in self.distances if d <= reach]
            if bad:
                raise ConfigError(
                    f"delayed_recall distances {bad} are within the stacked local "
                    f"reach L*(W-1)={reach}; they would be solvable without memory"
                )

    # ---- position bookkeeping --------------------------------------------

    @property
    def query_pos(self) -> int:
        """QUERY token position; the answer lands at ``query_pos + 2``."""
        return self.seq_len - 3

    def _generator(self, step: int) -> torch.Generator:
        # Mixed with a large odd constant so adjacent steps are not correlated.
        g = torch.Generator()
        g.manual_seed((self.seed * 1_000_003 + step) % (2**63 - 1))
        return g

    # ---- generation -------------------------------------------------------

    def batch(self, step: int, batch_size: int) -> ProbeBatch:
        """The batch for a given optimizer step. Pure function of (seed, step)."""
        g = self._generator(step)
        v = self.vocab
        t = self.seq_len

        tokens = torch.randint(
            v.filler_lo, v.size, (batch_size, t), generator=g, dtype=torch.long
        )
        # Cycle the distance grid so every batch covers every bucket evenly;
        # sampling i.i.d. would leave buckets unevenly represented at small B.
        offset = int(torch.randint(len(self.distances), (1,), generator=g))
        dist = torch.tensor(
            [self.distances[(offset + i) % len(self.distances)] for i in range(batch_size)]
        )

        if self.task == "two_hop":
            self._plant_two_hop(tokens, dist, g)
        else:
            self._plant_needle(tokens, dist, g)

        targets = torch.full_like(tokens, self.IGNORE_INDEX)
        targets[:, :-1] = tokens[:, 1:]
        answer_idx = torch.full((batch_size,), self.query_pos + 1, dtype=torch.long)

        loss_mask = targets != self.IGNORE_INDEX
        if self.data.loss_on_answer_only:
            loss_mask = torch.zeros_like(loss_mask)
            loss_mask[torch.arange(batch_size), answer_idx] = True
            targets = torch.where(loss_mask, targets, torch.full_like(targets, self.IGNORE_INDEX))

        return ProbeBatch(
            tokens=tokens,
            targets=targets,
            answer_idx=answer_idx,
            distance=dist,
            loss_mask=loss_mask,
        )

    def _plant_needle(self, tokens: Tensor, dist: Tensor, g: torch.Generator) -> None:
        """``... [K, V] ... [QUERY, K, V]`` with the needle at distance D."""
        v = self.vocab
        b = tokens.shape[0]
        q = self.query_pos

        keys = torch.randint(0, v.n_keys, (b, 1 + self.n_distractors), generator=g)
        values = torch.randint(0, v.n_values, (b, 1 + self.n_distractors), generator=g)

        for i in range(b):
            d = int(dist[i])
            p = q - d
            # Distractor keys must all differ from the needle key, or the query
            # would be ambiguous and the task unsolvable in principle.
            key_ids = _distinct(keys[i], v.n_keys, g)
            needle_key, needle_val = int(key_ids[0]), int(values[i, 0])

            occupied = {q, q + 1, q + 2, p, p + 1}
            tokens[i, p] = v.key_lo + needle_key
            tokens[i, p + 1] = v.value_lo + needle_val

            for j in range(1, self.n_distractors + 1):
                dp = _free_slot(q - 2, occupied, g)
                if dp is None:
                    break
                occupied.update({dp, dp + 1})
                tokens[i, dp] = v.key_lo + int(key_ids[j])
                tokens[i, dp + 1] = v.value_lo + int(values[i, j])

            tokens[i, q] = v.QUERY
            tokens[i, q + 1] = v.key_lo + needle_key
            tokens[i, q + 2] = v.value_lo + needle_val

    def _plant_two_hop(self, tokens: Tensor, dist: Tensor, g: torch.Generator) -> None:
        """``... [A, B] ... [B, C] ... [QUERY, A, C]``: two chained memory reads.

        The reported distance is the *farther* hop, since that is the binding
        constraint on how far back the memory has to reach.
        """
        v = self.vocab
        b = tokens.shape[0]
        q = self.query_pos

        for i in range(b):
            d1 = int(dist[i])
            # Second hop sits strictly between the two slots of the first hop
            # and the query: d2 in [2, d1-2], so p2 >= p1+2 and the pairs never
            # overlap. An overlap would silently truncate the A->B edge.
            d2 = int(torch.randint(2, max(3, d1 - 1), (1,), generator=g))
            p1, p2 = q - d1, q - d2
            assert p2 >= p1 + 2, f"two-hop pairs overlap at d1={d1}, d2={d2}"
            ids = _distinct(
                torch.randint(0, v.n_keys, (2 + self.n_distractors,), generator=g),
                v.n_keys,
                g,
            )
            a, bb = int(ids[0]), int(ids[1])
            c = int(torch.randint(0, v.n_values, (1,), generator=g))

            occupied = {q, q + 1, q + 2, p1, p1 + 1, p2, p2 + 1}
            tokens[i, p1] = v.key_lo + a
            tokens[i, p1 + 1] = v.key_lo + bb  # A -> B, both from the key alphabet
            tokens[i, p2] = v.key_lo + bb
            tokens[i, p2 + 1] = v.value_lo + c  # B -> C

            for j in range(2, 2 + self.n_distractors):
                dp = _free_slot(q - 2, occupied, g)
                if dp is None:
                    break
                occupied.update({dp, dp + 1})
                tokens[i, dp] = v.key_lo + int(ids[j])
                tokens[i, dp + 1] = v.value_lo + int(
                    torch.randint(0, v.n_values, (1,), generator=g)
                )

            tokens[i, q] = v.QUERY
            tokens[i, q + 1] = v.key_lo + a
            tokens[i, q + 2] = v.value_lo + c

    # ---- evaluation sets --------------------------------------------------

    def eval_batches(
        self, batch_size: int, *, n_per_distance: int = 128, seed_offset: int = 10**6
    ) -> List[ProbeBatch]:
        """Fixed batches, one distance per batch, so accuracy buckets cleanly."""
        out: List[ProbeBatch] = []
        for i, d in enumerate(self.distances):
            single = SyntheticGenerator(
                _with_distances(self.data, [d]),
                self.model,
                seed=self.seed + seed_offset + i,
                n_distractors=self.n_distractors,
            )
            done = 0
            while done < n_per_distance:
                n = min(batch_size, n_per_distance - done)
                out.append(single.batch(step=done, batch_size=n))
                done += n
        return out


def _with_distances(data: DataConfig, distances: Sequence[int]) -> DataConfig:
    import dataclasses

    return dataclasses.replace(data, distances=list(distances))


def _distinct(ids: Tensor, n_values: int, g: torch.Generator) -> Tensor:
    """Force a set of ids to be pairwise distinct, resampling collisions."""
    out = ids.clone().flatten()
    seen = set()
    for i in range(out.numel()):
        tries = 0
        while int(out[i]) in seen and tries < 100:
            out[i] = torch.randint(0, n_values, (1,), generator=g)
            tries += 1
        seen.add(int(out[i]))
    return out


def _free_slot(limit: int, occupied: set, g: torch.Generator) -> Optional[int]:
    """A position p with p and p+1 free, in [0, limit). None if none found."""
    for _ in range(50):
        p = int(torch.randint(0, max(1, limit), (1,), generator=g))
        if p not in occupied and (p + 1) not in occupied and p + 1 < limit:
            return p
    return None
