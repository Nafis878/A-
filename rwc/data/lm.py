"""Character-level language modelling from a plain text corpus.

Why character level, and why this replaces the FineWeb-Edu/BPE stub. The claim
under test is about a memory channel, not about tokenisation: does Maglev's write
rule carry information past the decoder's stacked local reach when the data is
natural language rather than a synthetic probe? A character corpus needs no BPE
training and no sharding, so the whole thing is CPU-feasible, and on text8 one
character is one byte -- which makes ``rwc.evaluate.bits_per_byte`` reduce to
``CE / ln 2`` and the numbers directly comparable to the char-LM literature.

The loader's contract is set by ``Trainer._micro_batch``, which calls
``batch(step * accum + micro, micro_batch)``. That index is a pure function of
the step, so as long as ``batch`` is a pure function of the index the data
position needs no separate state and a resume is exact for free -- the same
property ``SyntheticGenerator`` has. There is deliberately no cursor to save.

Held-out data is a DIFFERENT FILE (``text8.valid``) where one exists, or the tail
of the corpus otherwise, and the training slice is truncated so the two can never
overlap. Reporting bits-per-byte on data the model has trained on would make the
memory comparison meaningless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..config import ConfigError, DataConfig
from .synthetic import ProbeBatch

__all__ = ["CharShardLoader", "ShardLoader", "load_corpus"]

IGNORE_INDEX = -100

# text8's alphabet, fixed rather than inferred so that a corpus slice which
# happens to omit a rare letter cannot silently shift every id.
TEXT8_ALPHABET = " abcdefghijklmnopqrstuvwxyz"


def load_corpus(shard_dir: str, name: str = "text8") -> tuple:
    """Return ``(train_text, valid_text, source)`` from ``shard_dir``.

    Falls back through the corpora that fit on CPU. The name of whatever was
    actually used is returned and recorded in every result, because "which
    corpus" is not a detail a reader should have to infer.
    """
    root = Path(shard_dir)
    for stem in (name, "tiny_shakespeare", "corpus"):
        train = root / stem
        if not train.exists():
            continue
        text = train.read_text(encoding="utf-8", errors="ignore")
        valid_path = root / f"{stem}.valid"
        if valid_path.exists():
            return text, valid_path.read_text(encoding="utf-8", errors="ignore"), stem
        # No separate validation file: hold out the tail and truncate train so
        # the two cannot overlap.
        cut = int(len(text) * 0.95)
        return text[:cut], text[cut:], f"{stem} (tail split)"
    raise ConfigError(
        f"no corpus in {root}; expected one of text8, tiny_shakespeare, corpus"
    )


class CharShardLoader:
    """Streams fixed-length character windows, deterministic in the index."""

    def __init__(
        self,
        data: DataConfig,
        *,
        seed: int = 0,
        split: str = "train",
        max_chars: Optional[int] = None,
    ) -> None:
        if data.shard_dir is None:
            raise ConfigError("lm data requires data.shard_dir")
        train_text, valid_text, source = load_corpus(data.shard_dir)
        self.source = source
        self.split = split

        text = train_text if split == "train" else valid_text[: data.val_tokens]
        if max_chars is not None:
            text = text[:max_chars]

        alphabet = TEXT8_ALPHABET if source.startswith("text8") else "".join(
            sorted(set(train_text) | set(valid_text))
        )
        self.alphabet = alphabet
        self.vocab_size = len(alphabet)
        lookup = np.full(256, 0, dtype=np.uint8)
        for i, ch in enumerate(alphabet):
            lookup[ord(ch)] = i
        # uint8 keeps a 90M-character corpus at 90 MB; slices become int64 only
        # when a batch is actually built.
        raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
        self.ids = lookup[raw]

        self.seq_len = data.seq_len
        # +1 because each window needs one extra character to supply the target
        # for its final position; no position is wasted on an ignored label.
        self.n_chunks = (len(self.ids) - 1) // self.seq_len
        if self.n_chunks < 1:
            raise ConfigError(
                f"corpus has {len(self.ids)} chars, too few for seq_len={self.seq_len}"
            )
        # A fixed permutation of window starts. Sequential windows would put
        # adjacent text in one batch and correlate the gradient; a permutation
        # decorrelates it while staying a pure function of the index.
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(self.n_chunks)

    # -- the interface Trainer relies on ------------------------------------

    def batch(self, index: int, batch_size: int) -> ProbeBatch:
        """The batch at a given index. Pure function of ``(seed, index)``."""
        starts = [
            int(self.perm[(index * batch_size + j) % self.n_chunks]) * self.seq_len
            for j in range(batch_size)
        ]
        window = np.stack([self.ids[s : s + self.seq_len + 1] for s in starts])
        chunk = torch.from_numpy(window.astype(np.int64))

        tokens = chunk[:, : self.seq_len]
        targets = chunk[:, 1 : self.seq_len + 1]
        return ProbeBatch(
            tokens=tokens,
            targets=targets,
            # Full-sequence CE: this is language modelling, not answer-only
            # retrieval, so every position is scored.
            loss_mask=torch.ones_like(tokens, dtype=torch.bool),
            # Unused here, but ProbeBatch is the shared contract. The last
            # position is the only defensible "answer" index for an LM.
            answer_idx=torch.full((batch_size,), self.seq_len - 1, dtype=torch.long),
            distance=torch.zeros(batch_size, dtype=torch.long),
        )

    def __len__(self) -> int:
        return self.n_chunks

    def __repr__(self) -> str:
        return (
            f"CharShardLoader({self.source}, split={self.split!r}, "
            f"{len(self.ids)} chars, {self.n_chunks} windows of {self.seq_len}, "
            f"vocab {self.vocab_size})"
        )


# Trainer._build_data imports this name.
ShardLoader = CharShardLoader


def build_tokenizer(*args, **kwargs):
    """Not used: character level needs no learned tokenizer.

    Kept so the SPEC's M3-LM interface still resolves, and to make the choice
    explicit rather than looking like an oversight.
    """
    raise NotImplementedError(
        "character-level modelling needs no tokenizer; see CharShardLoader"
    )
