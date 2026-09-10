import os
import re
import time
import logging
from groq import Groq

logger = logging.getLogger("reasoning.groq")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"), max_retries=0)

MODEL_NAME = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


def chat_completion_with_retry(
    messages: list,
    model: str | None = None,
    temperature: float = 0,
    response_format: dict | None = None,
    max_tokens: int = 800,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
):
    """
    Executes a Groq chat completion with exponential backoff on HTTP 429
    (rate limits / TPM exhaustion) and transient network failures.
    """
    target_model = model or MODEL_NAME
    backoff = initial_backoff
    curr_response_format = response_format

    for attempt in range(1, max_retries + 1):
        try:
            curr_max_tokens = max_tokens
            # gpt-oss models use reasoning tokens and fail when response_format is enforced
            if "gpt-oss" in target_model or "qwen3.6" in target_model:
                curr_response_format = None
                curr_max_tokens = max(max_tokens, 1000)

            kwargs = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": curr_max_tokens,
            }
            if curr_response_format is not None:
                kwargs["response_format"] = curr_response_format

            return client.chat.completions.create(**kwargs)

        except Exception as exc:
            err_str = str(exc).lower()

            if "json_validate_failed" in err_str:
                logger.warning(
                    f"[Groq Retry] json_validate_failed on {target_model}, retrying without response_format..."
                )
                print(f"[Groq Retry] json_validate_failed on {target_model}, retrying without response_format...")
                curr_response_format = None
                continue

            is_rate_limit = any(
                term in err_str
                for term in ["429", "rate limit", "tpm", "tokens per minute", "tpd", "tokens per day", "rpm", "413"]
            )

            if is_rate_limit and attempt < max_retries:
                match = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", str(exc), re.IGNORECASE)
                if match:
                    mins = float(match.group(1)) if match.group(1) else 0.0
                    secs = float(match.group(2))
                    wait_time = mins * 60 + secs + 2.0
                else:
                    wait_time = backoff * (2 ** (attempt - 1))

                # On the first retry, if wait time is short (up to 5s), wait for quota decay
                if attempt == 1 and wait_time <= 5.0:
                    logger.warning(
                        f"[Groq Retry] Rate limit on {target_model} (attempt {attempt}/{max_retries}). "
                        f"Sleeping {wait_time:.1f}s for quota reset..."
                    )
                    print(f"[Groq Retry] Rate limit on {target_model}. Sleeping {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # Otherwise, rotate to the next model in the chain
                models_chain = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
                try:
                    curr_idx = models_chain.index(target_model)
                    alt_model = models_chain[(curr_idx + 1) % len(models_chain)]
                except ValueError:
                    alt_model = models_chain[0]

                logger.warning(f"[Groq Fallback] Switching from {target_model} to {alt_model}...")
                print(f"[Groq Fallback] Switching from {target_model} to {alt_model}...")
                target_model = alt_model
                time.sleep(1.0)
                continue

            elif attempt < max_retries:
                time.sleep(1.0)
            else:
                logger.error(f"[Groq Error] Final failure after {attempt} attempts: {exc}")
                raise exc
