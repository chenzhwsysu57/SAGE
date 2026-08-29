from __future__ import annotations

import os
import time
from urllib.parse import urlparse

from openai import OpenAI

from base.interfaces import BaseModel


class OpenAIModel(BaseModel):
    """Thin wrapper around OpenAI-compatible chat completions APIs (OpenAI, DeepSeek, etc.).

    Defaults to DeepSeek with thinking disabled for fair method comparison.
    Pass extra_body={"thinking": {"type": "enabled"}} to enable DeepSeek reasoning.

    Args:
        timeout: HTTP request timeout in seconds. 0 means no timeout.
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.7,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict | None = None,
        timeout: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self.extra_body = extra_body if extra_body is not None else {"thinking": {"type": "disabled"}}
        self._timeout = timeout

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass --api-key or set OPENAI_API_KEY env var."
            )

        client_kwargs: dict = {"api_key": resolved_key}
        if timeout > 0:
            client_kwargs["timeout"] = timeout
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_url:
            client_kwargs["base_url"] = resolved_url
            # Ollama's Qwen3 models use the `think` request field. The
            # DeepSeek-style `thinking.type=disabled` field is ignored by
            # Ollama, which can leave Qwen3 in a very slow reasoning mode.
            parsed_url = urlparse(resolved_url)
            if parsed_url.port == 11434 or parsed_url.hostname in {"ollama", "ollama.local"}:
                self.extra_body = {"think": False} if extra_body is None else extra_body
        self.client = OpenAI(**client_kwargs)

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        import openai

        kwargs: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "extra_body": self.extra_body,
        }

        last_exc = None
        for attempt in range(max_retries + 1):
            t0 = time.time()
            print(f"  [API] calling {self.model} ...", flush=True)
            try:
                response = self.client.chat.completions.create(**kwargs)
            except (openai.APIConnectionError, openai.APITimeoutError,
                    openai.RateLimitError, openai.InternalServerError) as e:
                dt = time.time() - t0
                last_exc = e
                if attempt < max_retries:
                    wait = 2 ** attempt * 5
                    print(f"  [API] FAILED after {dt:.0f}s ({type(e).__name__}), "
                          f"retry {attempt + 1}/{max_retries} in {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                print(f"  [API] FAILED after {dt:.0f}s: {type(e).__name__}: {e}", flush=True)
                raise
            except Exception as e:
                dt = time.time() - t0
                print(f"  [API] FAILED after {dt:.0f}s: {type(e).__name__}: {e}", flush=True)
                raise

            dt = time.time() - t0
            content = response.choices[0].message.content
            print(f"  [API] done in {dt:.0f}s (output length={len(content or '')})", flush=True)
            return content if content else ""

        # Shouldn't reach here, but just in case
        raise last_exc  # type: ignore[misc]
