from utils.llm_client import call_llm
from config import INTERPRETER_TEMP, DEFAULT_MODEL
from utils.logger import logger

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
    
    system_prompt = (
        "You are a Senior Prompt Engineer specializing in Python code generation.\n"
        "Your ONLY task is to rewrite the user's request into an ultra-detailed, unambiguous technical prompt for a code-generating AI.\n\n"
        "Follow these rules strictly:\n"
        "0. Title: Your output MUST start with a clear, concise '**Title:**' line that summarizes the task in 5-8 words (e.g., '**Title:** Multi-Threaded Log File Analyzer'). This title will be used as the filename, so make it descriptive and safe.\n"
        "1. Algorithm: If the user implies an algorithm (e.g., maze generation, sorting, pathfinding), explicitly name and detail the best standard algorithm (e.g., 'Recursive Backtracking', 'A* Search', 'QuickSort').\n"
        "2. Data Structures: Specify exactly what data structures to use (e.g., '2D list of characters', 'priority queue', 'caching dictionary').\n"
        "3. Logic: Explicitly describe the core logic step-by-step (e.g., how coordinates are traversed, walls carved, or states mutated). For maze generation, ALWAYS specify that '#' must be used for walls, ' ' (space) for paths, and to use standard Recursive Backtracking moving in 2-step increments.\n"
        "4. Constraints: State clearly: 'Use ONLY the Python standard library. No third-party packages are allowed. You MUST ABSOLUTELY NOT use from ... import statements targeting custom filenames (e.g., from log_parser import parse_log_file). Do not assume any external .py files exist. Define all helper functions directly inside this single file. You must only import built-in standard library modules (like os, sys, re, datetime, json, collections, concurrent.futures).'\n"
        "5. Concrete Implementation Rules:\n"
        "   - Directory Paths: The script must ALWAYS accept the target directory from sys.argv[1]. If sys.argv[1] is missing at runtime, it must fallback to asking the user for the directory path using input().\n"
        "   - Imports: All required libraries must be imported explicitly at the top of the file (e.g., if using thread pools, explicitly import ThreadPoolExecutor and as_completed from concurrent.futures).\n"
        "   - Log parsing format: The log parsing regex must follow a standard pattern without assuming brackets around the level or timestamp (e.g., 'YYYY-MM-DD HH:MM:SS LEVEL message'), parsing it cleanly.\n"
        "   - Error handling: Handle missing files or directories by logging an error message and prompting the user again or exiting gracefully without crash loops.\n"
        "6. Output: Define exactly how the final output should be formatted or printed (e.g., standard print logs, formatting styles).\n"
        "7. Edge Cases: Explicitly address boundaries, empty inputs, or limit constraints.\n\n"
        "Output ONLY the enhanced technical prompt. Do not add greetings, conversational filler, markdown formatting blocks, or explanations."
    )
    
    user_message = f"Please rewrite this request into a detailed technical prompt:\n\nUser Request: {prompt}\n\nEnhanced Technical Prompt:"
    
    enhanced_prompt = call_llm(user_message, system_prompt=system_prompt, model=model, temperature=INTERPRETER_TEMP)
    logger.info("Interpreter (Prompt Engineer) Agent finished successfully.")
    logger.debug(f"Enhanced Technical Prompt:\n{enhanced_prompt}")
    return enhanced_prompt
