"""LLM client abstraction for entity and relationship extraction.

Provides a unified interface for Ollama LLM integration.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional
import json

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMClientInterface(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text completion from a prompt.

        Args:
            prompt: The input prompt text.
            temperature: Sampling temperature (0.0-2.0). Lower = more deterministic.
            max_tokens: Maximum tokens to generate (None = model default).

        Returns:
            Generated text response.
        """

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any] | list[Any]:
        """Generate structured JSON response from a prompt.

        Args:
            prompt: The input prompt text (should request JSON output).
            temperature: Sampling temperature (0.0-2.0).

        Returns:
            Parsed JSON object (dict or list).

        Raises:
            ValueError: If response is not valid JSON.
        """


class OllamaLLMClient(LLMClientInterface):
    """Ollama LLM client implementation."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,  # Reduced default timeout
    ):
        """Initialize Ollama client.

        Args:
            model: Ollama model name (e.g., "llama3.1:8b", "meditron:70b")
            host: Ollama server URL
            timeout: Request timeout in seconds (default: 60)
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not installed. Install with: pip install ollama")

        self.model = model
        self.host = host
        self.timeout = timeout
        self._client = ollama.Client(host=host, timeout=timeout)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Ollama."""
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        # Run synchronous ollama client in thread pool to avoid blocking
        # Use asyncio.to_thread for cleaner async execution (Python 3.9+)
        def _generate():
            return self._client.generate(
                model=self.model,
                prompt=prompt,
                options=options,
            )
        
        try:
            # Add timeout wrapper
            response = await asyncio.wait_for(
                asyncio.to_thread(_generate),
                timeout=self.timeout
            )
            return response.get("response", "").strip()
        except asyncio.TimeoutError:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
        except Exception as e:
            print(f"Ollama generate error: {e}")
            raise

    async def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any] | list[Any]:
        """Generate JSON using Ollama."""
        response_text = await self.generate(prompt, temperature)

        # Try to extract JSON from response (LLMs sometimes add extra text)
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start == -1:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

        if json_start == -1:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")

        json_text = response_text[json_start:json_end]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}\nResponse: {json_text[:500]}")
