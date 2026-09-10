import os
import re
import time
import logging
from groq import Groq

logger = logging.getLogger("reasoning.groq")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


def chat_completion_with_retry(
    messages: list,
    model: str | None = None,
    temperature: float = 0,
    response_format: dict | None = None,
    max_tokens: int = 600,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
):
    """
    Executes a Groq chat completion with exponential backoff on HTTP 429
    (rate limits / TPM exhaustion) and transient network failures.
    """
    target_model = model or MODEL_NAME
    backoff = initial_backoff

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            return client.chat.completions.create(**kwargs)

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = any(
                term in err_str
                for term in ["429", "rate limit", "tpm", "tokens per minute", "tpd", "tokens per day", "rpm", "413"]
            )

            if is_rate_limit and attempt < max_retries:
                match = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", str(exc), re.IGNORECASE)
                if match:
                    mins = float(match.group(1)) if match.group(1) else 0.0
                    secs = float(match.group(2))
                    wait_time = mins * 60 + secs + 1.0
                else:
                    wait_time = backoff * (2 ** (attempt - 1))

                # If wait time is reasonable (up to 120s), wait for the rolling quota reset
                if wait_time <= 120.0:
                    logger.warning(
                        f"[Groq Retry] Rate limit on {target_model} (attempt {attempt}/{max_retries}). "
                        f"Sleeping {wait_time:.1f}s for quota reset..."
                    )
                    print(f"[Groq Retry] Rate limit on {target_model}. Sleeping {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # If wait time is longer, rotate models
                models_chain = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
                try:
                    curr_idx = models_chain.index(target_model)
                    alt_model = models_chain[(curr_idx + 1) % len(models_chain)]
                except ValueError:
                    alt_model = models_chain[0]

                if "120b" in alt_model and max_tokens < 800:
                    max_tokens = 800

                logger.warning(f"[Groq Fallback] Switching from {target_model} to {alt_model}...")
                print(f"[Groq Fallback] Switching to {alt_model}...")
                target_model = alt_model
                time.sleep(2.0)
                continue

            elif attempt < max_retries:
                time.sleep(1.0)
            else:
                logger.error(f"[Groq Error] Final failure after {attempt} attempts: {exc}")
                raise exc
