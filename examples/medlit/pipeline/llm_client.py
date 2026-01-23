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
            model: Ollama model name (e.g., "llama3.1:8b", "llama3.1:8b")
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
            response = await asyncio.wait_for(asyncio.to_thread(_generate), timeout=self.timeout)
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
        """Generate structured JSON response from a prompt."""

        def _generate():
            response = self._client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biomedical entity extraction expert. Extract entities and return ONLY valid JSON in the exact format requested. Never return the original text, only the extracted entities.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": temperature},
            )
            return response["message"]["content"]

        response_text = await asyncio.to_thread(_generate)
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        # Find and extract first complete JSON structure
        def find_matching_bracket(text: str, start: int, open_char: str, close_char: str) -> int:
            """Find the matching closing bracket for an opening bracket."""
            count = 1
            i = start + 1
            while i < len(text) and count > 0:
                if text[i] == open_char:
                    count += 1
                elif text[i] == close_char:
                    count -= 1
                i += 1
            return i if count == 0 else -1

        # Try to find JSON array first
        json_start = response_text.find("[")
        if json_start != -1:
            json_end = find_matching_bracket(response_text, json_start, "[", "]")
            if json_end != -1:
                json_text = response_text[json_start:json_end]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass

        # Try to find JSON object
        json_start = response_text.find("{")
        if json_start != -1:
            json_end = find_matching_bracket(response_text, json_start, "{", "}")
            if json_end != -1:
                json_text = response_text[json_start:json_end]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"No valid JSON found in response: {response_text[:200]}")
