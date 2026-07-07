from utils.llm_client import call_llm
from config import PLANNER_TEMP, DEFAULT_MODEL
from utils.logger import logger

def plan(enhanced_prompt, model=DEFAULT_MODEL):
    """
    Takes an enhanced technical prompt and generates a detailed step-by-step implementation plan.
    
    Args:
        enhanced_prompt (str): The highly detailed technical prompt.
        model (str): The Ollama model to use.
        
    Returns:
        str: A detailed implementation plan in Markdown.
    """
    logger.info("Running Planner Agent...")
    
    system_prompt = (
        "You are an expert Software Architect and Tech Lead. Based on the highly detailed technical prompt provided, "
        "produce a detailed, step-by-step implementation plan for writing the Python code.\n"
        "Your plan must cover:\n"
        "1. Module imports and structural layout\n"
        "2. Component layout: List functions, classes, their parameters, and logic\n"
        "3. Step-by-step logic flow detailing algorithmic execution\n\n"
        "Output the implementation plan in Markdown. Do not write the actual runnable Python code."
    )
    
    user_message = f"Please write a step-by-step implementation plan based on this technical prompt:\n\n{enhanced_prompt}"
    
    implementation_plan = call_llm(user_message, system_prompt=system_prompt, model=model, temperature=PLANNER_TEMP)
    logger.info("Planner Agent finished successfully.")
    logger.debug(f"Generated Plan:\n{implementation_plan}")
    return implementation_plan
