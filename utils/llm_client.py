import urllib.request
import urllib.error
import json
import time
import socket
from typing import Optional
from config import OLLAMA_API_URL, DEFAULT_MODEL, LLM_CALL_TIMEOUT
from utils.logger import logger


def _ollama_reachable(timeout: int = 5) -> bool:
    """Quick health check: returns True if Ollama's root endpoint responds."""
    try:
        req = urllib.request.Request(OLLAMA_API_URL, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: int = LLM_CALL_TIMEOUT,
) -> str:
    """
    Calls the local Ollama chat API. Retries with exponential backoff and
    increased timeout if a timeout occurs.

    Args:
        prompt (str): The user message prompt.
        system_prompt (str, optional): Instruction prompt for the system role.
        model (str): Ollama model to use.
        temperature (float): Controls response creativity.
        timeout (int): Initial socket timeout in seconds.

    Returns:
        str: Response content from the assistant.
    """
    url = f"{OLLAMA_API_URL}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    data = json.dumps(payload).encode("utf-8")
    current_timeout = timeout

    for attempt in range(1, 4):
        logger.debug(
            f"Calling Ollama API (attempt {attempt}/3): model={model}, "
            f"temp={temperature}, timeout={current_timeout}"
        )

        # Connection health check before each retry (skip on first attempt to save time)
        if attempt > 1:
            if not _ollama_reachable():
                wait = 2 ** attempt
                logger.warning(
                    f"Ollama not reachable at {OLLAMA_API_URL}. "
                    f"Waiting {wait}s before retry {attempt}/3..."
                )
                time.sleep(wait)
                continue

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=current_timeout) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                content = res_json.get("message", {}).get("content", "")
                if attempt > 1:
                    logger.info(f"Ollama API succeeded on retry attempt {attempt}.")
                logger.debug("Received successful response from Ollama API.")
                return content

        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            is_timeout = (
                isinstance(e, (TimeoutError, socket.timeout))
                or (isinstance(e, urllib.error.URLError) and isinstance(e.reason, socket.timeout))
                or (isinstance(e, urllib.error.URLError) and "timed out" in str(e.reason).lower())
            )

            if is_timeout:
                current_timeout += 60
                wait = 2 ** attempt  # exponential backoff: 2, 4, 8 seconds
                logger.warning(
                    f"Ollama API timed out (attempt {attempt}/3). "
                    f"Backing off {wait}s, new timeout={current_timeout}s..."
                )
                if attempt < 3:
                    time.sleep(wait)
                    continue

            logger.error(f"Ollama API request failed: {e}")
            raise ConnectionError(
                f"Could not connect to Ollama at {OLLAMA_API_URL}. Ensure Ollama is running."
            ) from e

        except json.JSONDecodeError as e:
            logger.error(f"Ollama API returned invalid JSON: {e}")
            raise ValueError("Invalid JSON response from Ollama API.") from e

        except Exception as e:
            logger.error(f"Unexpected error when calling Ollama API: {e}")
            raise

    raise RuntimeError("All LLM call retries timed out or failed.")
