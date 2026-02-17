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
        self.ollama_host: str = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434"
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
            print("DEBUG: Inside async context manager")  # Add this
            self.model = "nomic-embed-text"
            print(f"DEBUG: Set model to {self.model}")  # Add this
            url = self.ollama_host + "/api/embeddings"
            print(f"DEBUG: URL is {url}")  # Add this
            try:
                response = await client.post(
                    url,
                    json={"model": self.model, "prompt": text},
                )
                print("EMBED STATUS:", response.status_code)
                if response.status_code != 200:
                    print("EMBED BODY:", response.text[:500])
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                import traceback

                print("EMBED EXCEPTION:", repr(e))
                traceback.print_exc()
                raise

            if "embedding" in data and isinstance(data["embedding"], list):
                embedding = data["embedding"]
            elif "embeddings" in data and data["embeddings"]:
                embedding = data["embeddings"][0]
            else:
                raise ValueError(f"Unexpected response format: {data}")

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
