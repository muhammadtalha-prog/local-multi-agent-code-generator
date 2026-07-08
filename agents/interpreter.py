import re
from utils.llm_client import call_llm
from config import INTERPRETER_TEMP, DEFAULT_MODEL
from utils.logger import logger

# Whole-word keywords that indicate the prompt involves file/directory operations.
# Using regex word boundaries prevents false positives like 'move' inside 'movement'.
_FILE_KEYWORD_PATTERN = re.compile(
    r'\b(?:file|files|folder|directory|directories|path|log|logs|csv|txt|json|'
    r'read|write|parse|rename|delete|move|copy)\b',
    re.IGNORECASE
)

def _prompt_uses_files(prompt: str) -> bool:
    """Return True if the prompt is about file or directory operations."""
    return bool(_FILE_KEYWORD_PATTERN.search(prompt))


def engineer_prompt(prompt, model=DEFAULT_MODEL):
    """
    Expands a vague user prompt into an ultra-detailed, technical prompt for code generation.

    Args:
        prompt (str): The user's original request.
        model (str): The Ollama model to use.

    Returns:
        str: An enhanced technical prompt.
    """
    logger.info("Running Interpreter (Prompt Engineer) Agent...")

    # Issue 7 fix: only add sys.argv rule when the prompt is about files/folders
    file_ops_rule = (
        "   - Directory/File Paths: The script must ALWAYS accept the target path from sys.argv[1]. "
        "If sys.argv[1] is missing at runtime, fallback to asking the user via input().\n"
    ) if _prompt_uses_files(prompt) else (
        "   - Arguments: Only add command-line argument handling if it is genuinely required by the task.\n"
    )

    system_prompt = (
        "You are a Senior Prompt Engineer specializing in Python code generation.\n"
        "Your ONLY task is to rewrite the user's request into an ultra-detailed, unambiguous technical prompt for a code-generating AI.\n\n"
        "Follow these rules strictly:\n"
        "0. Title: Your output MUST start with a clear, concise '**Title:**' line that summarizes the task in 5-8 words "
        "(e.g., '**Title:** Multi-Threaded Log File Analyzer'). This title will be used as the filename, so make it descriptive and safe.\n"
        "1. Algorithm: If the user implies an algorithm (e.g., maze generation, sorting, pathfinding), explicitly name and detail "
        "the best standard algorithm (e.g., 'Recursive Backtracking', 'A* Search', 'QuickSort').\n"
        "2. Data Structures: Specify exactly what data structures to use (e.g., '2D list of characters', 'priority queue').\n"
        "3. Logic: Explicitly describe the core logic step-by-step.\n"
        "4. Constraints:\n"
        "   - Do NOT import from custom filenames or assume external .py files exist.\n"
        "   - Define all helper functions inside this single file.\n"
        "   - If the task requires external packages (e.g., cv2, mediapipe, numpy, requests), state them explicitly. "
        "Every library used must be imported at the top of the file with its standard alias "
        "(e.g., numpy -> import numpy as np, mediapipe -> import mediapipe as mp, opencv -> import cv2).\n"
        "5. Concrete Implementation Rules:\n"
        + file_ops_rule +
        "   - Imports: All required libraries must be imported explicitly at the top of the file.\n"
        "   - Error handling: Handle all edge cases gracefully without crash loops.\n"
        "6. Output: Define exactly how the final output should be formatted or printed.\n"
        "7. Edge Cases: Explicitly address boundaries, empty inputs, or limit constraints.\n"
        "8. screen_brightness_control (SBC) rules (apply ONLY if the task involves screen brightness):\n"
        "   - Do NOT use `ScreenBrightnessControl` class – it does not exist.\n"
        "   - Correct API: `import screen_brightness_control as sbc`, then `sbc.get_brightness()` and `sbc.set_brightness(value)`.\n"
        "   - Always wrap brightness calls in try/except.\n\n"
        "Output ONLY the enhanced technical prompt. No greetings, filler, or markdown code blocks."
    )

    user_message = (
        f"Please rewrite this request into a detailed technical prompt:\n\n"
        f"User Request: {prompt}\n\nEnhanced Technical Prompt:"
    )

    enhanced_prompt = call_llm(user_message, system_prompt=system_prompt, model=model, temperature=INTERPRETER_TEMP)
    logger.info("Interpreter (Prompt Engineer) Agent finished successfully.")
    logger.debug(f"Enhanced Technical Prompt:\n{enhanced_prompt}")
    return enhanced_prompt
