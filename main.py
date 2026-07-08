import argparse
import sys
import os
import time
import re
import subprocess
import tempfile
from datetime import datetime
from config import MAX_RETRIES, MAX_SYNTAX_RETRIES, DEFAULT_MODEL, IMPORT_TO_PACKAGE
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
    Runs the generated code in a subprocess with a short timeout to catch early
    import/attribute/name errors before the script blocks (e.g. on webcam or user input).

    Returns: (is_ok, error_message_or_None)
    """
    logger.info("Running Runtime Smoke Test...")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name

    proc = None
    try:
        # Issue 4 fix: Use Popen so we can forcibly kill child processes on Windows
        proc = subprocess.Popen(
            [sys.executable, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            _, stderr_output = proc.communicate(timeout=2)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            # Script is still running after 2s – likely blocking on camera/input – treat as pass
            proc.kill()
            proc.communicate()  # drain pipes to avoid zombie
            logger.info("Runtime Smoke Test passed (timeout – script started without immediate crash).")
            return True, None

        if returncode != 0:
            err = stderr_output.strip()
            err_lower = err.lower()

            # Issue 5 fix: explicit checks in priority order
            if "modulenotfounderror" in err_lower or "importerror" in err_lower:
                logger.info("Smoke test: missing dependency – user will install later.")
                return True, None  # not fatal, deps reported separately

            if "nameerror" in err_lower:
                return False, f"Runtime Name Error (undefined variable/function):\n{err}"

            if "attributeerror" in err_lower:
                return False, f"Runtime Attribute Error:\n{err}"

            if "typeerror" in err_lower:
                return False, f"Runtime Type Error:\n{err}"

            if "error" in err_lower or "exception" in err_lower:
                return False, f"Startup error:\n{err}"

            return False, f"Process exited with non-zero code {returncode}:\n{err}"
        else:
            logger.info("Runtime Smoke Test passed successfully.")
            return True, None

    except Exception as e:
        logger.error(f"Unexpected error during smoke test: {e}")
        return False, f"Unexpected error during smoke test: {e}"
    finally:
        if proc and proc.poll() is None:
            proc.kill()
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def slugify(text):
    """
    Convert a title into a safe filename.
    Example: "Generate Random Grid-Based Maze" -> "generate_random_grid_based_maze"
    """
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[-\s]+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text[:50]


def install_packages(mapped_missing):
    """Runs pip install for the given list of PyPI package names. Returns True on success."""
    logger.info(f"Installing packages: {mapped_missing}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + mapped_missing,
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("[SUCCESS] Installation successful!")
        logger.info("pip install succeeded.")
        return True
    else:
        print("[ERROR] Installation failed. Please run the commands above manually.")
        print(result.stderr)
        logger.error(f"pip install failed:\n{result.stderr}")
        return False


class Supervisor:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = model

    def run(self, prompt, output_path=None, fast=False, auto_install=False):
        """
        Orchestrates the multi-agent code generation pipeline.

        Args:
            prompt (str): User's natural language request.
            output_path (str): Optional explicit output file path.
            fast (bool): If True, skip the planning stage.
            auto_install (bool): If True, install missing packages without prompting.
        """
        logger.info(f"Supervisor initiated. Target Model: {self.model}")
        logger.info(f"User Prompt: '{prompt}' (Fast Mode: {fast}, Auto-Install: {auto_install})")

        start_time = time.time()
        failed = False  # Issue 1 fix: use flag instead of sys.exit inside try/except

        # Issue 2 fix: compute shared workspace_dir and timestamp ONCE up front
        base_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.join(base_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            code_plan = None  # keep in scope for non-fast mode

            if fast:
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

                # Step 2: Create Implementation Plan
                logger.info("=== STEP 2: CREATING IMPLEMENTATION PLAN ===")
                code_plan = run_agent_with_retry(plan, enhanced_prompt, model=self.model)
                if not code_plan:
                    raise RuntimeError("Planner agent failed to generate an implementation plan.")
                print("\n[PLAN] Generated Implementation Plan:\n" + "="*40)
                print(code_plan)
                print("="*40 + "\n")

                # Step 3: Code Generation
                logger.info("=== STEP 3: GENERATING CODE ===")
                code = run_agent_with_retry(generate, code_plan, model=self.model)
                if not code:
                    raise RuntimeError("Generator agent failed to generate code.")

            # Auto-patch missing sys import
            if "sys.argv" in code and "import sys" not in code:
                logger.info("Auto-patching missing 'sys' import...")
                code = "import sys\n" + code

            # Step 4: Syntax & Smoke Test Loop
            logger.info("=== STEP 4: VALIDATING & CORRECTING CODE ===")
            last_error = None
            # Issue 3 fix: track the last fully-validated code explicitly
            final_code = None

            for iteration in range(1, MAX_SYNTAX_RETRIES + 1):
                logger.info(f"Validation iteration {iteration}/{MAX_SYNTAX_RETRIES}")
                syntax_error, validated_code = check_syntax(code)

                if syntax_error is not None:
                    last_error = syntax_error
                    logger.warning(f"Syntax error (iter {iteration}): {syntax_error}")
                else:
                    # Static import scan
                    import_ok, import_error, external_packages = check_imports(code)
                    if not import_ok:
                        last_error = import_error
                        logger.warning(f"Import check failed (iter {iteration}): {import_error}")
                    else:
                        # Detect missing packages
                        missing = []
                        for pkg in external_packages:
                            try:
                                __import__(pkg)
                            except (ImportError, ModuleNotFoundError):
                                missing.append(pkg)
                            except Exception as ie:
                                # Issue 5 fix: catch other import-time crashes too
                                logger.warning(f"Importing '{pkg}' raised unexpected error: {ie}")
                                missing.append(pkg)

                        if missing:
                            mapped_missing = [IMPORT_TO_PACKAGE.get(p, p) for p in missing]

                            req_path = os.path.join(workspace_dir, f"requirements_{run_timestamp}.txt")
                            try:
                                with open(req_path, "w", encoding="utf-8") as rf:
                                    rf.write("\n".join(mapped_missing) + "\n")
                            except Exception as file_err:
                                logger.error(f"Failed to write requirements file: {file_err}")

                            req_cmd  = f'pip install -r "{req_path}"'
                            direct_cmd = "pip install " + " ".join(mapped_missing)

                            print("\n" + "="*60)
                            print("[WARNING] Missing External Dependencies Detected")
                            print("="*60)
                            print(f"The generated code uses these packages: {', '.join(missing)}")
                            print("\nTo install them, run ONE of these commands:\n")
                            print(f"  {req_cmd}")
                            print("  # OR")
                            print(f"  {direct_cmd}")
                            print("\n" + "="*60)

                            # Issue 8 fix: honour --install flag or prompt interactively
                            if auto_install:
                                install_packages(mapped_missing)
                                missing = []
                            else:
                                try:
                                    answer = input("\nDo you want me to install them for you now? (y/n): ").strip().lower()
                                    if answer == 'y':
                                        if install_packages(mapped_missing):
                                            missing = []
                                    else:
                                        print("\nOK. You can install later using the commands above.")
                                except Exception as e:
                                    print(f"\nCould not prompt for installation: {e}. Please install manually.")

                        # Smoke test (handles missing deps gracefully)
                        smoke_ok, smoke_error = smoke_test(code)
                        if smoke_ok:
                            final_code = validated_code  # Issue 3 fix: only set on confirmed pass
                            logger.info("Code passed all validation checks.")
                            break
                        else:
                            last_error = smoke_error
                            logger.warning(f"Smoke test failed (iter {iteration}): {smoke_error}")

                if iteration < MAX_SYNTAX_RETRIES:
                    logger.info("Re-running Generator with error feedback...")
                    if fast:
                        code = run_agent_with_retry(
                            generate_direct, prompt, model=self.model,
                            error_message=last_error, previous_code=code
                        )
                    else:
                        code = run_agent_with_retry(
                            generate, code_plan, model=self.model,
                            error_message=last_error, previous_code=code
                        )
                else:
                    logger.error("Maximum validation retries reached.")

            # Issue 1 fix: no sys.exit inside try; use flag
            if final_code is None:
                print("\n[ERROR] The supervisor failed to produce validated code after all retries.")
                if last_error:
                    print(f"Last error:\n{last_error}")
                if code:
                    print("\nHere is the last generated code:\n")
                    print(code)
                failed = True
            else:
                elapsed_time = time.time() - start_time
                logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds.")

                print("\n[SUCCESS] Final Python Code:\n" + "="*40)
                print(final_code)
                print("="*40 + "\n")

                # Save the code
                try:
                    if output_path:
                        final_path = output_path
                    else:
                        base_name = None
                        if not fast and 'enhanced_prompt' in locals():
                            title_match = re.search(r'\*\*Title:\*\*\s*(.*?)(?:\n|$)', enhanced_prompt, re.IGNORECASE)
                            if title_match:
                                base_name = slugify(title_match.group(1).strip())

                        if not base_name:
                            raw_prompt = prompt[:50].strip()
                            base_name = slugify(raw_prompt) if raw_prompt else "generated_code"

                        if len(base_name) < 3:
                            base_name = "generated_script"

                        filename = f"{base_name}_{run_timestamp}.py"
                        final_path = os.path.join(workspace_dir, filename)

                    with open(final_path, "w", encoding="utf-8") as f:
                        f.write(final_code)
                    logger.info(f"Saved generated code to '{final_path}'")
                    print(f"Code saved to [{os.path.basename(final_path)}](file:///{os.path.abspath(final_path).replace(chr(92), '/')})")
                except Exception as file_err:
                    logger.error(f"Failed to write generated code: {file_err}")
                    print(f"Warning: Could not save code to file: {file_err}")

        except Exception as e:
            logger.critical(f"Supervisor encountered a critical exception: {e}")
            print(f"\n[CRITICAL FAILURE] {e}")
            print("Please check agent_system.log for details.")
            failed = True

        # Issue 1 fix: exit AFTER except block is fully done
        if failed:
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
        help="Optional path to output the generated python script."
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Use fast single-shot code generation mode, skipping specifications and plans."
    )
    # Issue 8 fix: --install flag
    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="Automatically install missing external packages without prompting."
    )

    args = parser.parse_args()

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
    supervisor.run(args.prompt, args.output, args.fast, auto_install=args.install)


if __name__ == "__main__":
    main()
