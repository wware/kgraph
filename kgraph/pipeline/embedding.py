"""Embedding generation interface for the knowledge graph framework."""

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingGeneratorInterface(ABC):
    """Generate semantic vector embeddings for text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of generated embeddings."""

    @abstractmethod
    async def generate(self, text: str) -> tuple[float, ...]:
        """Generate an embedding for a single text string.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as a tuple of floats
        """

    @abstractmethod
    async def generate_batch(
        self,
        texts: Sequence[str],
    ) -> list[tuple[float, ...]]:
        """Generate embeddings for multiple texts efficiently.

        Default implementations may call generate() for each text.
        Override for batch optimization with embedding APIs.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors in the same order as input
        """
