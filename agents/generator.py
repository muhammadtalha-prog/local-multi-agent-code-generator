import re
from utils.llm_client import call_llm
from config import GENERATOR_TEMP, DEFAULT_MODEL
from utils.logger import logger

def extract_python_code(text):
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

def generate(plan, model=DEFAULT_MODEL, error_message=None, previous_code=None):
    """
    Generates runnable Python code from an implementation plan, or refines code to fix syntax errors.
    
    Args:
        plan (str): The implementation plan.
        model (str): The Ollama model to use.
        error_message (str, optional): The syntax error message from the previous attempt.
        previous_code (str, optional): The code from the previous attempt that had the syntax error.
        
    Returns:
        str: Raw Python code.
    """
    logger.info("Running Generator Agent...")
    
    system_prompt = (
        "You are an expert Python Developer. Your goal is to write complete, syntactically correct, "
        "and clean Python code implementing the given plan. Use only standard libraries unless "
        "external modules are explicitly requested. Do not include placeholders, comments like '# TODO', "
        "or write incomplete implementations. Your entire code must be runnable.\n"
        "Crucially: Output the final code wrapped inside a single standard ```python ... ``` markdown block. "
        "Any explanations or discussions must go outside that markdown block, or be omitted entirely."
    )
    
    if error_message and previous_code:
        logger.info("Generator Agent is performing syntax correction...")
        user_message = (
            f"Here is the software implementation plan:\n\n{plan}\n\n"
            f"Your previous code attempt had a syntax error. Here is the code you generated:\n"
            f"```python\n{previous_code}\n```\n\n"
            f"The syntax checker reported the following error:\n"
            f"{error_message}\n\n"
            f"Please write the complete, corrected version of the code that resolves this error. "
            f"Make sure to output the entire script, not just the fix."
        )
    else:
        user_message = (
            f"Please write complete, runnable Python code following this implementation plan:\n\n{plan}"
        )
        
    raw_response = call_llm(user_message, system_prompt=system_prompt, model=model, temperature=GENERATOR_TEMP)
    code = extract_python_code(raw_response)
    
    logger.info("Generator Agent finished successfully.")
    logger.debug(f"Generated Python Code:\n{code}")
    return code

def generate_direct(prompt, model=DEFAULT_MODEL, error_message=None, previous_code=None):
    """
    Generates runnable Python code directly from a prompt in single-shot mode.
    
    Args:
        prompt (str): The original user prompt/request.
        model (str): The Ollama model to use.
        error_message (str, optional): The syntax error message from the previous attempt.
        previous_code (str, optional): The code from the previous attempt that had the syntax error.
        
    Returns:
        str: Raw Python code.
    """
    logger.info("Running Generator Agent (Single-Shot Mode)...")
    
    system_prompt = (
        "You are an expert Python Developer. Your goal is to write a complete, syntactically correct, "
        "and clean Python script that fulfills the user's request. Use only standard libraries unless "
        "external modules are explicitly requested. Do not include placeholders or incomplete logic.\n"
        "Crucially: Output the final code wrapped inside a single standard ```python ... ``` markdown block. "
        "Do not include any conversational explanation before or after the code block."
    )
    
    if error_message and previous_code:
        logger.info("Generator Agent is performing syntax correction in Single-Shot Mode...")
        user_message = (
            f"Here is the software request:\n\n{prompt}\n\n"
            f"Your previous code attempt had a syntax error. Here is the code you generated:\n"
            f"```python\n{previous_code}\n```\n\n"
            f"The syntax checker reported the following error:\n"
            f"{error_message}\n\n"
            f"Please write the complete, corrected version of the code that resolves this error. "
            f"Make sure to output the entire script, not just the fix."
        )
    else:
        user_message = (
            f"Please write a complete, runnable Python script for the following request:\n\n{prompt}"
        )
        
    raw_response = call_llm(user_message, system_prompt=system_prompt, model=model, temperature=GENERATOR_TEMP)
    code = extract_python_code(raw_response)
    
    logger.info("Generator Agent finished successfully (Single-Shot Mode).")
    logger.debug(f"Generated Single-Shot Python Code:\n{code}")
    return code

