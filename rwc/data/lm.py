"""FineWeb-Edu tokenization to uint16 shards, and the shard loader.

Not yet implemented: milestone M3-LM (SPEC.md sections 6 and 9). The synthetic
domain is the headline (SPEC.md section 1); this is the secondary confirmation
and is built after the influence gate clears.

The loader's position is ``(shard_index, offset)`` rather than a step counter,
because shards have unequal length; ``state_dict``/``load_state_dict`` carry it
through a resume exactly as the synthetic generator's step counter does.
"""

from __future__ import annotations


class ShardLoader:
    """Streams uint16 token shards. Milestone M3-LM."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("milestone M3-LM: see SPEC.md section 9")


def build_tokenizer(*args, **kwargs):
    """Train a 16k BPE tokenizer on a FineWeb-Edu subset. Milestone M3-LM."""
    raise NotImplementedError("milestone M3-LM: see SPEC.md section 9")
