# examples/sherlock/extractors.py
from __future__ import annotations

import hashlib
from typing import Sequence

from kgraph.pipeline.embedding import EmbeddingGeneratorInterface


# ============================================================================
# Embedding generator (toy)
# ============================================================================


class SimpleEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Deterministic hash-based embedding generator (demo only)."""

    @property
    def dimension(self) -> int:
        return 32

    async def generate(self, text: str) -> tuple[float, ...]:
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        values = [b / 255.0 for b in h[: self.dimension]]
        mag = sum(v * v for v in values) ** 0.5
        if mag == 0:
            return tuple(0.0 for _ in values)
        return tuple(v / mag for v in values)

    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [await self.generate(t) for t in texts]
