import os
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq, RateLimitError

# ARCHITECTURE.md §4.4 point 2 — confirmed directly from the Groq dashboard, not estimated.
TPM_LIMITS = {
    "GROQ_MODEL_CHEAP": 70_000,  # groq/compound-mini
    "GROQ_MODEL_STRONG": 8_000,  # qwen/qwen3.8-27b
    "GROQ_MODEL_SYNTHESIS": 8_000,  # openai/gpt-oss-120b
}

# ARCHITECTURE.md §4.4 point 1 — four retries beyond the original attempt.
BACKOFF_SCHEDULE_S = [1, 2, 4, 8]

DEFAULT_EXPECTED_OUTPUT_TOKENS = 500


@dataclass
class LLMResult:
    ok: bool
    text: Optional[str] = None
    error: Optional[str] = None
    rate_limited: bool = False


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

_client_instance: Optional[Groq] = None

# Per-model rolling one-minute token usage window: {model_env_var: [(monotonic_ts, tokens), ...]}
_usage_windows: dict[str, list[tuple[float, int]]] = {}


def _client() -> Groq:
    global _client_instance
    if _client_instance is None:
        _client_instance = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client_instance


def _estimate_tokens(prompt: str, expected_output_tokens: int) -> int:
    return len(prompt) // 4 + expected_output_tokens


def _prune_window(window: list[tuple[float, int]], now: float) -> None:
    cutoff = now - 60
    while window and window[0][0] < cutoff:
        window.pop(0)


def _wait_for_tpm_headroom(model_env_var: str, prompt: str, expected_output_tokens: int) -> None:
    limit = TPM_LIMITS[model_env_var]
    estimate = _estimate_tokens(prompt, expected_output_tokens)
    if estimate > limit:
        # A single call bigger than the model's whole TPM budget — proceed and let 429
        # handling take over rather than spin forever waiting for headroom that can't exist.
        return

    window = _usage_windows.setdefault(model_env_var, [])
    while True:
        now = time.monotonic()
        _prune_window(window, now)
        used = sum(tokens for _, tokens in window)
        if used + estimate <= limit:
            return
        sleep_s = max(window[0][0] + 60 - now, 0.1)
        time.sleep(sleep_s)


def _record_usage(model_env_var: str, tokens: int) -> None:
    window = _usage_windows.setdefault(model_env_var, [])
    window.append((time.monotonic(), tokens))


def _extract_token_usage(response) -> Optional[int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return getattr(usage, "total_tokens", None)


def _retry_after_seconds(exc: RateLimitError) -> Optional[int]:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _call(
    model_env_var: str,
    prompt: str,
    timeout_s: int,
    expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS,
) -> LLMResult:
    model = os.environ[model_env_var]
    _wait_for_tpm_headroom(model_env_var, prompt, expected_output_tokens)

    last_error: Optional[Exception] = None

    for attempt in range(len(BACKOFF_SCHEDULE_S) + 1):
        try:
            response = _client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_s,
            )
            content = response.choices[0].message.content
            tokens_used = _extract_token_usage(response) or _estimate_tokens(
                prompt, expected_output_tokens
            )
            _record_usage(model_env_var, tokens_used)
            return LLMResult(ok=True, text=content)
        except RateLimitError as exc:
            last_error = exc
            if attempt < len(BACKOFF_SCHEDULE_S):
                sleep_s = _retry_after_seconds(exc) or BACKOFF_SCHEDULE_S[attempt]
                time.sleep(sleep_s)
            continue
        except Exception as exc:  # timeout, connection error, bad model string, etc.
            return LLMResult(ok=False, error=str(exc), rate_limited=False)

    return LLMResult(ok=False, error=str(last_error), rate_limited=True)


def call_cheap(
    prompt: str, *, timeout_s: int = 15, expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS
) -> LLMResult:
    return _call("GROQ_MODEL_CHEAP", prompt, timeout_s, expected_output_tokens)


def call_strong(
    prompt: str, *, timeout_s: int = 20, expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS
) -> LLMResult:
    return _call("GROQ_MODEL_STRONG", prompt, timeout_s, expected_output_tokens)


def call_synthesis(
    prompt: str, *, timeout_s: int = 30, expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS
) -> LLMResult:
    return _call("GROQ_MODEL_SYNTHESIS", prompt, timeout_s, expected_output_tokens)
