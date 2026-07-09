import re
from typing import Optional
from utils.llm_client import call_llm
from config import INTERPRETER_TEMP, DEFAULT_MODEL, MAX_PROMPT_TOKENS
from utils.logger import logger

# Whole-word keywords that indicate the prompt involves file/directory operations.
# Using regex word boundaries prevents false positives like 'move' inside 'movement'.
_FILE_KEYWORD_PATTERN = re.compile(
    r'\b(?:file|files|folder|directory|directories|path|log|logs|csv|txt|json|'
    r'read|write|parse|rename|delete|move|copy)\b',
    re.IGNORECASE
)

# Keywords that indicate the prompt requires physical hardware or heavy ML inference
_HARDWARE_KEYWORD_PATTERN = re.compile(
    r'\b(?:webcam|camera|microphone|mic|gpu|real-time|realtime|mediapipe|'
    r'tensorflow|torch|pytorch|yolo|face detection|object detection|inference)\b',
    re.IGNORECASE
)

# Keywords that indicate the prompt is about screen brightness control
_BRIGHTNESS_KEYWORD_PATTERN = re.compile(
    r'\b(?:brightness|screen brightness|display brightness)\b',
    re.IGNORECASE
)


def _prompt_uses_files(prompt: str) -> bool:
    """Return True if the prompt is about file or directory operations."""
    prompt_clean = prompt.lower()

    # Remove generic coding command prefixes (e.g. "write a python script", "create a function")
    prompt_clean = re.sub(
        r"\b(?:write|create|build|make|develop)\s+(?:a\s+)?(?:python\s+)?(?:script|function|program|code|app|application|sieve|generator)\b",
        "",
        prompt_clean
    )

    # Remove generic user/console input readings (e.g. "read from console", "get user input")
    prompt_clean = re.sub(
        r"\b(?:read|get|accept|ask\s+for)\s+(?:user\s+)?(?:input|console|keyboard|args|arguments)\b",
        "",
        prompt_clean
    )

    return bool(_FILE_KEYWORD_PATTERN.search(prompt_clean))



def _prompt_needs_hardware_note(prompt: str) -> bool:
    """Return True only if the prompt explicitly involves hardware or heavy ML models."""
    return bool(_HARDWARE_KEYWORD_PATTERN.search(prompt))


def _prompt_uses_brightness(prompt: str) -> bool:
    """Return True only if the prompt explicitly involves screen brightness control."""
    return bool(_BRIGHTNESS_KEYWORD_PATTERN.search(prompt))


def _check_prompt_length(prompt: str) -> Optional[str]:
    """
    Estimate token count and return a warning string if it exceeds MAX_PROMPT_TOKENS,
    or None if within limits. Uses the approximation: 1 token ≈ 4 characters.
    """
    approx_tokens = len(prompt) // 4
    if approx_tokens > MAX_PROMPT_TOKENS:
        return (
            f"[WARNING] Your prompt is approximately {approx_tokens} tokens, "
            f"which exceeds the recommended limit of {MAX_PROMPT_TOKENS} tokens "
            f"for the current model. The model may truncate or ignore parts of the request.\n"
            f"Consider shortening your prompt or switching to a larger model."
        )
    return None


def engineer_prompt(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Expands a vague user prompt into an ultra-detailed, technical prompt for code generation.
    Ambiguity checks and feasibility notes are injected only when relevant to avoid
    confusing small LLMs with unrelated rules.

    Args:
        prompt (str): The user's original request.
        model (str): The Ollama model to use.

    Returns:
        str: An enhanced technical prompt.
    """
    logger.info("Running Interpreter (Prompt Engineer) Agent...")

    # Prompt length guard
    length_warning = _check_prompt_length(prompt)
    if length_warning:
        logger.warning(length_warning)
        print(f"\n{length_warning}\n")

    # ── Conditional rule injection ────────────────────────────────────────────
    # Only add sys.argv rule when the prompt is about files/folders
    file_ops_rule = (
        "   - Directory/File Paths: The script must ALWAYS accept the target path from sys.argv[1]. "
        "If sys.argv[1] is missing at runtime, fallback to asking the user via input().\n"
    ) if _prompt_uses_files(prompt) else (
        "   - Arguments: Only add command-line argument handling if it is genuinely required by the task.\n"
    )

    # Only add feasibility note rule when the prompt actually involves hardware/heavy ML
    feasibility_rule = (
        "2. Feasibility Note: This task involves hardware (webcam/camera/microphone) or heavy ML models. "
        "Add a brief note at the top of the enhanced prompt describing what hardware or compute is needed "
        "so the user knows what to expect before running the script.\n"
    ) if _prompt_needs_hardware_note(prompt) else ""

    # Only add SBC rules when prompt explicitly mentions brightness — injecting these rules
    # for unrelated tasks causes small models to hallucinate SBC imports everywhere.
    sbc_rule = (
        "BRIGHTNESS CONTROL API RULES (this task involves screen brightness):\n"
        "   - Do NOT use a 'ScreenBrightnessControl' class — it does not exist.\n"
        "   - Correct API: 'import screen_brightness_control as sbc', "
        "then 'sbc.get_brightness()' and 'sbc.set_brightness(value)'.\n"
        "   - Always wrap brightness calls in try/except.\n"
    ) if _prompt_uses_brightness(prompt) else ""
    # ─────────────────────────────────────────────────────────────────────────

    system_prompt = (
        "You are a Senior Prompt Engineer specializing in Python code generation.\n"
        "Your ONLY task is to rewrite the user's request into an ultra-detailed, unambiguous technical prompt "
        "for a code-generating AI.\n\n"
        "Follow these rules strictly:\n"
        "0. Title: Your output MUST start with a clear, concise '**Title:**' line that summarizes the task in 5-8 words "
        "(e.g., '**Title:** Sieve of Eratosthenes Prime Generator'). This title will be used as the filename.\n"
        "1. Ambiguity Check: Identify ANY ambiguous terms in the request and state the assumption you are making. "
        "Example: 'swipe detection' is ambiguous — assume directional hand movement delta, NOT a pixel distance threshold.\n"
        + feasibility_rule +
        "3. Algorithm: If the user implies an algorithm (e.g., Sieve of Eratosthenes, A*, QuickSort), "
        "explicitly name it and describe it step by step.\n"
        "4. Data Structures: Specify exactly what data structures to use.\n"
        "5. Logic: Describe the core logic step-by-step.\n"
        "6. Constraints:\n"
        "   - Do NOT import from custom filenames or assume external .py files exist.\n"
        "   - Define all helper functions inside this single file.\n"
        "   - ONLY include imports for libraries that are explicitly required by the task. "
        "Do NOT invent or add any library not mentioned or clearly implied by the task.\n"
        "   - Every library used must be imported at the top of the file with its standard alias.\n"
        "7. Concrete Implementation Rules:\n"
        + file_ops_rule +
        "   - Error handling: Handle all edge cases gracefully without crash loops.\n"
        "8. Output: Define exactly how the final output should be formatted or printed.\n"
        "9. Edge Cases: Explicitly address boundaries, empty inputs, or limit constraints.\n"
        + sbc_rule +
        "\nOutput ONLY the enhanced technical prompt. No greetings, filler, or markdown code blocks."
    )

    user_message = (
        f"Please rewrite this request into a detailed technical prompt:\n\n"
        f"User Request: {prompt}\n\nEnhanced Technical Prompt:"
    )

    enhanced_prompt = call_llm(user_message, system_prompt=system_prompt, model=model, temperature=INTERPRETER_TEMP)
    logger.info("Interpreter (Prompt Engineer) Agent finished successfully.")
    logger.debug(f"Enhanced Technical Prompt:\n{enhanced_prompt}")
    return enhanced_prompt
