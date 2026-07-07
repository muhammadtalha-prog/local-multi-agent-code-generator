import argparse
import sys
import os
import time
import re
from datetime import datetime
from config import MAX_RETRIES, MAX_SYNTAX_RETRIES, DEFAULT_MODEL
from utils.logger import logger
from agents import engineer_prompt, plan, generate, generate_direct, check_syntax, check_imports

def run_agent_with_retry(agent_func, *args, **kwargs):
    """
    Executes an agent function, retrying up to MAX_RETRIES if it encounters an exception.
    """
    retries = kwargs.pop("max_retries", MAX_RETRIES)
    for attempt in range(retries):
        try:
            return agent_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Component '{agent_func.__name__}' failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                logger.error(f"Component '{agent_func.__name__}' failed permanently after {retries} attempts.")
                raise
            time.sleep(1)
    return None

def smoke_test(code):
    """
    Runs the generated code in a subprocess with a dummy argument to catch common runtime issues
    (like NameError, AttributeError, ModuleNotFoundError) on startup.
    """
    import tempfile
    import subprocess
    
    logger.info("Running Runtime Smoke Test...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name
        
    try:
        # Run standard python executable on the script with a short timeout and a dummy argument
        result = subprocess.run(
            [sys.executable, temp_path, "."],
            capture_output=True,
            timeout=3,
            text=True
        )
        
        # Analyze standard error output
        stderr_lower = result.stderr.lower()
        if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower:
            err = f"Runtime Import Error:\n{result.stderr.strip()}"
            logger.warning(err)
            return False, err
        if "nameerror" in stderr_lower:
            err = f"Runtime Name Error (undefined variable/function):\n{result.stderr.strip()}"
            logger.warning(err)
            return False, err
        if "attributeerror" in stderr_lower:
            err = f"Runtime Attribute Error:\n{result.stderr.strip()}"
            logger.warning(err)
            return False, err
            
        logger.info("Runtime Smoke Test passed successfully.")
        return True, None
    except subprocess.TimeoutExpired:
        # Timeout means the script launched and is likely blocking for user input or executing correctly
        logger.info("Runtime Smoke Test passed (timeout expired without crash).")
        return True, None
    except Exception as e:
        logger.error(f"Unexpected error during smoke test: {e}")
        return False, f"Unexpected error during smoke test: {e}"
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

def slugify(text):
    """
    Convert a title into a safe filename.
    Example: "Generate Random Grid-Based Maze" -> "generate_random_grid_based_maze"
    """
    # Remove special characters, keep letters, numbers, spaces, underscores
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    # Replace spaces and hyphens with underscores
    text = re.sub(r'[-\s]+', '_', text)
    # Remove extra underscores and limit length
    text = re.sub(r'_+', '_', text)
    return text[:50]  # Prevent excessively long filenames

class Supervisor:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = model

    def run(self, prompt, output_path=None, fast=False):
        """
        Orchestrates the multi-agent code generation pipeline.
        """
        logger.info(f"Supervisor initiated. Target Model: {self.model}")
        logger.info(f"User Prompt: '{prompt}' (Fast Mode: {fast})")
        
        start_time = time.time()
        
        try:
            if fast:
                # Single-shot Code Generation -> Generate code directly
                logger.info("=== SINGLE-SHOT CODE GENERATION ===")
                code = run_agent_with_retry(generate_direct, prompt, model=self.model)
                if not code:
                    raise RuntimeError("Generator agent failed to generate code in single-shot mode.")
            else:
                # Step 1: Prompt Engineering
                logger.info("=== STEP 1: PROMPT ENGINEERING ===")
                enhanced_prompt = run_agent_with_retry(engineer_prompt, prompt, model=self.model)
                if not enhanced_prompt:
                    raise RuntimeError("Interpreter agent failed to engineer a prompt.")
                print("\n[ENHANCED PROMPT] Engineered Prompt:\n" + "="*40)
                print(enhanced_prompt)
                print("="*40 + "\n")
                
                # Step 2: Create Implementation Plan -> Step-by-Step Code Structure
                logger.info("=== STEP 2: CREATING IMPLEMENTATION PLAN ===")
                code_plan = run_agent_with_retry(plan, enhanced_prompt, model=self.model)
                if not code_plan:
                    raise RuntimeError("Planner agent failed to generate an implementation plan.")
                print("\n[PLAN] Generated Implementation Plan:\n" + "="*40)
                print(code_plan)
                print("="*40 + "\n")
                
                # Step 3: Code Generation -> Generate initial script
                logger.info("=== STEP 3: GENERATING CODE ===")
                code = run_agent_with_retry(generate, code_plan, model=self.model)
                if not code:
                    raise RuntimeError("Generator agent failed to generate code.")
            
            # Auto-patch missing sys import if sys.argv is present in code (Upgrade 3)
            if "sys.argv" in code and "import sys" not in code:
                logger.info("Auto-patching missing 'sys' import...")
                code = "import sys\n" + code
            
            # Step 4: Syntax Check Loop -> Validate and correct code
            logger.info("=== STEP 4: VALIDATING & CORRECTING CODE ===")
            syntax_error = None
            for iteration in range(1, MAX_SYNTAX_RETRIES + 1):
                logger.info(f"AST check iteration {iteration}/{MAX_SYNTAX_RETRIES}")
                syntax_error, validated_code = check_syntax(code)
                
                if syntax_error is None:
                    # Run static import check (Upgrade 1)
                    import_ok, import_error = check_imports(code)
                    if not import_ok:
                        syntax_error = import_error
                    else:
                        # Run runtime smoke test (Upgrade 2)
                        smoke_ok, smoke_error = smoke_test(code)
                        if smoke_ok:
                            code = validated_code
                            logger.info("Code syntax, static imports, and smoke test are valid.")
                            break
                        else:
                            syntax_error = smoke_error
                
                logger.warning(f"Syntax error found during compilation (attempt {iteration}):\n{syntax_error}")
                if iteration == MAX_SYNTAX_RETRIES:
                    logger.error("Maximum syntax correction retries reached.")
                    break
                
                # Feedback loop: Regenerate code with the error details
                logger.info("Re-running Generator agent with syntax error feedback...")
                if fast:
                    code = run_agent_with_retry(
                        generate_direct, 
                        prompt, 
                        model=self.model, 
                        error_message=syntax_error, 
                        previous_code=code
                    )
                else:
                    code = run_agent_with_retry(
                        generate, 
                        code_plan, 
                        model=self.model, 
                        error_message=syntax_error, 
                        previous_code=code
                    )
                
            if syntax_error:
                logger.error("Failed to generate syntactically correct code.")
                print("\n[ERROR] The supervisor failed to produce syntactically correct code.")
                print(f"Details:\n{syntax_error}")
                print("\nHere is the generated code with syntax errors:\n")
                print(code)
                sys.exit(1)
                
            elapsed_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds.")
            
            print("\n[SUCCESS] Final Python Code:\n" + "="*40)
            print(code)
            print("="*40 + "\n")
            
            # Save the code
            try:
                if output_path:
                    final_path = output_path
                else:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    workspace_dir = os.path.join(base_dir, "workspace")
                    os.makedirs(workspace_dir, exist_ok=True)
                    
                    # Try to extract the title from the enhanced prompt if not fast mode
                    base_name = None
                    if not fast and 'enhanced_prompt' in locals():
                        title_match = re.search(r'\*\*Title:\*\*\s*(.*?)(?:\n|$)', enhanced_prompt, re.IGNORECASE)
                        if title_match:
                            base_name = slugify(title_match.group(1).strip())
                    
                    if not base_name:
                        # Fallback: use the first 50 chars of the user's original prompt
                        raw_prompt = prompt[:50].strip()
                        base_name = slugify(raw_prompt) if raw_prompt else "generated_code"
                        
                    if len(base_name) < 3:
                        base_name = "generated_script"
                        
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{base_name}_{timestamp}.py"
                    final_path = os.path.join(workspace_dir, filename)
                
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(code)
                logger.info(f"Saved generated code to '{final_path}'")
                print(f"Code saved to [{os.path.basename(final_path)}](file:///{os.path.abspath(final_path).replace(chr(92), '/')})")
            except Exception as file_err:
                logger.error(f"Failed to write generated code: {file_err}")
                print(f"Warning: Could not save code to file: {file_err}")
                
        except Exception as e:
            logger.critical(f"Supervisor encountered a critical exception: {e}")
            print(f"\n[CRITICAL FAILURE] {e}")
            print("Please check agent_system.log for details.")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Local Python Code Generator Supervisor CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="The software requirements description (e.g., 'a prime number generator')."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"The Ollama model to use. Defaults to '{DEFAULT_MODEL}'."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Optional path to output the generated python script. Defaults to 'generated_code.py'."
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Use fast single-shot code generation mode, skipping specifications and plans."
    )
    
    args = parser.parse_args()
    
    # If prompt not provided in arguments, prompt interactivly
    if not args.prompt:
        try:
            print("--- Local Multi-Agent Code Generator CLI ---")
            args.prompt = input("Enter your prompt: ").strip()
            if not args.prompt:
                print("Prompt cannot be empty. Exiting.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)
            
    supervisor = Supervisor(model=args.model)
    supervisor.run(args.prompt, args.output, args.fast)

if __name__ == "__main__":
    main()
