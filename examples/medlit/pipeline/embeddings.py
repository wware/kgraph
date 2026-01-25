"""Embedding generation for medical entities.

Simple hash-based embedding generator for now. Can be enhanced with
biomedical embedding models (BioBERT, etc.) later.
"""

import os
from typing import Optional, Sequence

import httpx

from kgraph.pipeline.embedding import EmbeddingGeneratorInterface


class OllamaMedLitEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Real embedding generator using Ollama.

    Uses Ollama's embedding API to generate vectors for medical entities.
    Default model is nomic-embed-text which performs well on medical text.
    """

    def __init__(
        self,
        # model: str = "nomic-embed-text",
        model: str = "mxbai-embed-large",
        ollama_host: str | None = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Return embedding dimension for the model.

        Common dimensions:
        - nomic-embed-text: 768
        - mxbai-embed-large: 1024
        - bge-large: 1024
        """
        if self._dimension is None:
            # Lazy initialization - get dimension from first embedding
            raise RuntimeError("Dimension not yet determined. Call generate() at least once first.")
        return self._dimension

    async def generate(self, text: str) -> tuple[float, ...]:
        """Generate embedding for a single text using Ollama.

        Args:
            text: The text to generate an embedding for.

        Returns:
            Tuple of float values representing the embedding vector.

        Raises:
            httpx.HTTPError: If Ollama request fails
            ValueError: If response format is unexpected
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.ollama_host}/api/embed", json={"model": self.model, "input": text})
            response.raise_for_status()

            data = response.json()

            # Ollama returns {"embeddings": [[...]], ...}
            if "embeddings" not in data or not data["embeddings"]:
                raise ValueError(f"Unexpected response format: {data}")

            embedding = data["embeddings"][0]

            # Set dimension on first call
            if self._dimension is None:
                self._dimension = len(embedding)

            return tuple(embedding)

    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Sequence of texts to generate embeddings for.

        Returns:
            List of embedding tuples in the same order as input texts.
        """
        return [await self.generate(t) for t in texts]
