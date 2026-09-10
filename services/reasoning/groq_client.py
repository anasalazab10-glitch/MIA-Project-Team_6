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
            is_daily_limit = "tokens per day" in err_str or "tpd" in err_str
            is_rate_limit = any(
                term in err_str
                for term in ["429", "rate limit", "tpm", "tokens per minute", "413"]
            )
            is_json_err = "json_validate_failed" in err_str or "failed to validate json" in err_str

            if is_daily_limit and attempt < max_retries:
                # Do not sleep for hours if daily limit is reached on one model: switch immediately
                alt_model = "qwen/qwen3.8-27b" if "qwen3.8" not in target_model else "openai/gpt-oss-120b"
                logger.warning(f"[Groq Daily Limit] Switching from {target_model} to {alt_model}...")
                print(f"[Groq Daily Limit] Switching to {alt_model}...")
                target_model = alt_model
                time.sleep(1.0)
                continue

            if is_json_err and response_format is not None:
                # Retry once without response_format, letting prompt handle JSON instruction
                logger.warning(f"[Groq JSON Fallback] Retrying {target_model} without strict response_format...")
                try:
                    kwargs_no_rf = {
                        "model": target_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    return client.chat.completions.create(**kwargs_no_rf)
                except Exception:
                    pass

            if is_rate_limit and attempt < max_retries:
                wait_time = backoff
                match = re.search(r"try again in ([\d\.]+)s", str(exc), re.IGNORECASE)
                if match:
                    wait_time = max(wait_time, float(match.group(1)) + 0.5)
                else:
                    wait_time = backoff * (2 ** (attempt - 1))

                logger.warning(
                    f"[Groq Retry] Rate limited on {target_model} (attempt {attempt}/{max_retries}). "
                    f"Backing off for {wait_time:.2f}s... Error: {exc}"
                )
                print(f"[Groq Retry] Rate limit hit. Backing off for {wait_time:.2f}s...")
                time.sleep(wait_time)
            elif attempt < max_retries and not is_rate_limit:
                # If model is failing with 400 or transient error, try qwen3.8-27b as fallback
                if target_model != "qwen/qwen3.8-27b":
                    logger.warning(f"[Groq Fallback] Switching from {target_model} to qwen/qwen3.8-27b...")
                    target_model = "qwen/qwen3.8-27b"
                time.sleep(1.0)
            else:
                logger.error(f"[Groq Error] Final failure after {attempt} attempts: {exc}")
                raise exc
