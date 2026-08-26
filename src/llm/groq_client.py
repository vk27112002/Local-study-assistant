"""Validated Groq chat-completions client with bounded retry behavior."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os
import time

import requests
from dotenv import load_dotenv


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GENERATION_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2.0


def has_api_key() -> bool:
    key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY).strip()
    return bool(key and "your_" not in key.lower())


def _retry_delay(response, attempt: int) -> float:
    value = response.headers.get("retry-after")
    if value:
        try:
            return max(0.0, min(float(value), 60.0))
        except ValueError:
            try:
                target = parsedate_to_datetime(value)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return max(0.0, min((target - datetime.now(timezone.utc)).total_seconds(), 60.0))
            except (TypeError, ValueError, OverflowError):
                pass
    return min(BASE_BACKOFF_SECONDS * (2 ** attempt), 60.0)


def call_llm(prompt: str, max_tokens: int = 500, temperature: float = 0.2) -> str:
    """Send one prompt and return non-empty assistant text."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")

    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY).strip()
    if not has_api_key():
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    payload = {
        "model": GENERATION_MODEL,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": prompt.strip()}],
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=90,
            )
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError("Could not connect to Groq after all retries.") from exc
            time.sleep(min(BASE_BACKOFF_SECONDS * (2 ** attempt), 60.0))
            continue

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError("Groq returned HTTP 200 with invalid JSON.") from exc
            if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
                raise RuntimeError("Groq returned a malformed response without choices.")
            choice = data["choices"][0]
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Groq returned an empty assistant response.")
            return content.strip()

        retryable = response.status_code in {408, 409, 425, 429} or response.status_code >= 500
        if not retryable:
            try:
                error = response.json()
            except ValueError:
                error = response.text[:2000]
            raise RuntimeError(f"Groq request failed with HTTP {response.status_code}: {error}")
        if attempt == MAX_RETRIES:
            raise RuntimeError(f"Groq request failed after all retries (HTTP {response.status_code}).")
        time.sleep(_retry_delay(response, attempt))

    raise RuntimeError("Groq request failed unexpectedly.")


if __name__ == "__main__":
    print("Groq key configured:", has_api_key())
