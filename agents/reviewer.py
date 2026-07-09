"""
Code Reviewer Agent
===================
Validates that generated code semantically matches the user's original request.
Catches cases where the LLM used the wrong library, wrong algorithm, or omitted
key requirements (e.g., asked for scipy but got numpy).
"""
import re
from typing import Tuple
from utils.llm_client import call_llm
from config import DEFAULT_MODEL, INTERPRETER_TEMP
from utils.logger import logger


def review_code(original_prompt: str, generated_code: str, model: str = DEFAULT_MODEL) -> Tuple[bool, str]:
    """
    Checks whether the generated code correctly fulfills the user's original request.

    Args:
        original_prompt (str): The user's original natural-language request.
        generated_code (str): The Python code produced by the Generator agent.
        model (str): The Ollama model to use for review.

    Returns:
        Tuple[bool, str]: (is_aligned, issues_description)
            - is_aligned: True if code matches the request, False if discrepancies found.
            - issues_description: Empty string if aligned; otherwise a description of problems.
    """
    logger.info("Running Code Reviewer Agent...")

    system_prompt = (
        "You are a strict Code Reviewer. Your job is to check whether generated Python code "
        "actually implements what the user asked for.\n\n"
        "Review the code against the original request and look for:\n"
        "1. Wrong libraries used (e.g., user asked for scipy but code uses numpy)\n"
        "2. Missing key features explicitly mentioned in the request\n"
        "3. Incorrect algorithms (e.g., user asked for A* search but code uses BFS)\n"
        "4. Logic that clearly cannot work (e.g., swipe detection using distance instead of movement)\n"
        "5. Placeholder or incomplete implementations (TODO comments, pass-only functions)\n\n"
        "IMPORTANT RULES:\n"
        "- Do NOT complain about code style, formatting, or minor preferences.\n"
        "- Do NOT flag missing error handling as a discrepancy unless explicitly requested.\n"
        "- If the code correctly satisfies the request (even imperfectly), respond with: ALIGNED\n"
        "- If there are real discrepancies, respond with: ISSUES\\n<bullet list of specific problems>\n"
        "- Be concise. Max 5 bullet points."
    )

    user_message = (
        f"Original User Request:\n{original_prompt}\n\n"
        f"Generated Code:\n```python\n{generated_code}\n```\n\n"
        "Does this code correctly implement the user's request? "
        "Reply with ALIGNED or ISSUES followed by a bullet list."
    )

    try:
        raw = call_llm(user_message, system_prompt=system_prompt, model=model, temperature=INTERPRETER_TEMP)
        raw = raw.strip()
        logger.debug(f"Reviewer raw response:\n{raw}")

        if raw.upper().startswith("ALIGNED"):
            logger.info("Code Reviewer Agent: code is aligned with the request.")
            return True, ""

        # Extract issues from the response
        issues_match = re.search(r"ISSUES\s*[\n:](.*)", raw, re.DOTALL | re.IGNORECASE)
        if issues_match:
            issues = issues_match.group(1).strip()
        else:
            # Fallback: treat entire response as issues if neither keyword matched
            issues = raw

        logger.warning(f"Code Reviewer Agent found discrepancies:\n{issues}")
        return False, issues

    except Exception as e:
        # Reviewer failure is non-fatal — log and treat as aligned to avoid blocking the pipeline
        logger.warning(f"Code Reviewer Agent failed (non-fatal): {e}. Treating as aligned.")
        return True, ""
