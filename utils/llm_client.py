import urllib.request
import urllib.error
import json
import time
import socket
from config import OLLAMA_API_URL, DEFAULT_MODEL
from utils.logger import logger

def call_llm(prompt, system_prompt=None, model=DEFAULT_MODEL, temperature=0.2, timeout=120):
    """
    Calls the local Ollama chat API. Retries with backoff and increased timeout if a timeout occurs.
    
    Args:
        prompt (str): The user message prompt.
        system_prompt (str, optional): Instruction prompt for the system role.
        model (str): Ollama model to use.
        temperature (float): Controls response creativity.
        timeout (int): Socket timeout in seconds.
        
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
    
    data = json.dumps(payload).encode('utf-8')
    
    for attempt in range(1, 4):
        logger.debug(f"Calling Ollama API (attempt {attempt}/3): model={model}, temp={temperature}, timeout={timeout}")
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_body = response.read().decode('utf-8')
                res_json = json.loads(res_body)
                content = res_json.get("message", {}).get("content", "")
                logger.debug("Received successful response from Ollama API.")
                return content
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            # Check if this error was caused by a socket timeout
            is_timeout = False
            if isinstance(e, (TimeoutError, socket.timeout)):
                is_timeout = True
            elif isinstance(e, urllib.error.URLError) and isinstance(e.reason, socket.timeout):
                is_timeout = True
            elif isinstance(e, urllib.error.URLError) and "timed out" in str(e.reason).lower():
                is_timeout = True
                
            if is_timeout:
                logger.warning(f"Ollama API request timed out (attempt {attempt}/3). Increasing timeout and retrying...")
                timeout += 60
                if attempt < 3:
                    time.sleep(2)
                    continue
            
            logger.error(f"Ollama API request failed: {e}")
            raise ConnectionError(f"Could not connect to Ollama at {OLLAMA_API_URL}. Ensure Ollama is running.") from e
        except json.JSONDecodeError as e:
            logger.error(f"Ollama API returned invalid JSON: {e}")
            raise ValueError("Invalid JSON response from Ollama API.") from e
        except Exception as e:
            logger.error(f"Unexpected error when calling Ollama API: {e}")
            raise

    raise RuntimeError("All LLM call retries timed out or failed.")

