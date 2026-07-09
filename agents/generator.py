import re
from typing import Optional
from utils.llm_client import call_llm
from config import GENERATOR_TEMP, DEFAULT_MODEL
from utils.logger import logger


def extract_python_code(text: str) -> str:
    """
    Extracts python code from markdown blocks.

    Args:
        text (str): The raw text output from the LLM.

    Returns:
        str: Extracted python code.
    """
    # Look for ```python ... ```
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback to general ``` ... ```
    match_generic = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()

    # If no markdown block, return raw text trimmed
    return text.strip()


# Shared generator system prompt — kept minimal and generic so the small model
# does NOT hallucinate library-specific imports for unrelated tasks.
_GENERATOR_SYSTEM_PROMPT = (
    "You are an expert Python Developer. Write complete, syntactically correct, runnable Python code.\n"
    "STRICT RULES:\n"
    "  1. Import ONLY the libraries that are explicitly mentioned or clearly required by the plan/request. "
    "Do NOT add any library that is not referenced in the plan.\n"
    "  2. Every library you import must actually be used in the code.\n"
    "  3. Import every module you use at the top of the file. No implicit imports.\n"
    "  4. Do not include placeholders, '# TODO' comments, or incomplete logic.\n"
    "  5. Output the final code wrapped inside a single ```python ... ``` markdown block. "
    "Omit all explanations outside that block."
)


def generate(
    plan: str,
    model: str = DEFAULT_MODEL,
    error_message: Optional[str] = None,
    previous_code: Optional[str] = None,
) -> str:
    """
    Generates runnable Python code from an implementation plan, or refines code to fix errors.

    Args:
        plan (str): The implementation plan.
        model (str): The Ollama model to use.
        error_message (str, optional): The error message from the previous attempt.
        previous_code (str, optional): The code from the previous attempt that had the error.

    Returns:
        str: Raw Python code.
    """
    logger.info("Running Generator Agent...")

    if error_message and previous_code:
        logger.info("Generator Agent is performing correction...")
        user_message = (
            f"Here is the software implementation plan:\n\n{plan}\n\n"
            f"Your previous code attempt had an error. Here is the code you generated:\n"
            f"```python\n{previous_code}\n```\n\n"
            f"The error reported was:\n"
            f"{error_message}\n\n"
            f"Please write the complete, corrected version of the code that resolves this error. "
            f"Make sure to output the entire script, not just the fix."
        )
    else:
        user_message = (
            f"Please write complete, runnable Python code following this implementation plan:\n\n{plan}"
        )

    raw_response = call_llm(user_message, system_prompt=_GENERATOR_SYSTEM_PROMPT, model=model, temperature=GENERATOR_TEMP)
    code = extract_python_code(raw_response)

    logger.info("Generator Agent finished successfully.")
    logger.debug(f"Generated Python Code:\n{code}")
    return code


def generate_direct(
    prompt: str,
    model: str = DEFAULT_MODEL,
    error_message: Optional[str] = None,
    previous_code: Optional[str] = None,
) -> str:
    """
    Generates runnable Python code directly from a prompt in single-shot mode.

    Args:
        prompt (str): The original user prompt/request.
        model (str): The Ollama model to use.
        error_message (str, optional): The error message from the previous attempt.
        previous_code (str, optional): The code from the previous attempt that had the error.

    Returns:
        str: Raw Python code.
    """
    logger.info("Running Generator Agent (Single-Shot Mode)...")

    if error_message and previous_code:
        logger.info("Generator Agent is performing correction in Single-Shot Mode...")
        user_message = (
            f"Here is the software request:\n\n{prompt}\n\n"
            f"Your previous code attempt had an error. Here is the code you generated:\n"
            f"```python\n{previous_code}\n```\n\n"
            f"The error reported was:\n"
            f"{error_message}\n\n"
            f"Please write the complete, corrected version of the code that resolves this error. "
            f"Make sure to output the entire script, not just the fix."
        )
    else:
        user_message = (
            f"Please write a complete, runnable Python script for the following request:\n\n{prompt}"
        )

    raw_response = call_llm(user_message, system_prompt=_GENERATOR_SYSTEM_PROMPT, model=model, temperature=GENERATOR_TEMP)
    code = extract_python_code(raw_response)

    logger.info("Generator Agent finished successfully (Single-Shot Mode).")
    logger.debug(f"Generated Single-Shot Python Code:\n{code}")
    return code
