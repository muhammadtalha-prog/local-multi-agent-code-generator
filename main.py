import argparse
import sys
import os
import time
import re
import importlib.util
from datetime import datetime
from typing import Optional, Set

# ── UTF-8 stdout: prevents Windows-1252 charmap crashes on emoji/unicode output ──
try:
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
except Exception:
    pass  # Silently skip if stdout is not a real file (e.g. redirected/piped)

from config import (
    MAX_RETRIES, MAX_SYNTAX_RETRIES, DEFAULT_MODEL,
    IMPORT_TO_PACKAGE, MAX_PROMPT_TOKENS,
)
from utils.logger import logger
from utils.session import save_session, load_last_session, print_history
from agents import engineer_prompt, plan, generate, generate_direct, check_syntax, check_imports, review_code


# ---------------------------------------------------------------------------
# Prompt input validation
# ---------------------------------------------------------------------------

_SHELL_METACHAR_RE = re.compile(r"[;&|`$<>]")


def validate_prompt(prompt: str) -> Optional[str]:
    """
    Returns an error string if the prompt is invalid, or None if it's OK.

    Checks:
    - Not empty / whitespace-only
    - Does not exceed MAX_PROMPT_TOKENS (approx)
    - Does not contain shell metacharacters that could indicate injection
    """
    if not prompt or not prompt.strip():
        return "Prompt cannot be empty."

    approx_tokens = len(prompt) // 4
    if approx_tokens > MAX_PROMPT_TOKENS:
        return (
            f"Prompt is too long (~{approx_tokens} tokens). "
            f"Please keep it under {MAX_PROMPT_TOKENS} tokens (~{MAX_PROMPT_TOKENS * 4} characters) "
            f"for the current model."
        )

    bad = _SHELL_METACHAR_RE.findall(prompt)
    if bad:
        return (
            f"Prompt contains potentially unsafe characters: {set(bad)}. "
            "Please rephrase without shell metacharacters (;, &, |, `, $, <, >)."
        )

    return None


# ---------------------------------------------------------------------------
# Dependency pre-validation
# ---------------------------------------------------------------------------

# Simple keyword → package hints for fast pre-check (before spending LLM compute)
_PROMPT_TO_PACKAGES: dict = {
    "numpy": "numpy", "np": "numpy",
    "pandas": "pandas", "dataframe": "pandas",
    "matplotlib": "matplotlib", "plot": "matplotlib",
    "opencv": "opencv-python", "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "tensorflow": "tensorflow", "keras": "tensorflow",
    "torch": "torch", "pytorch": "torch",
    "sklearn": "scikit-learn", "scikit": "scikit-learn",
    "scipy": "scipy",
    "requests": "requests",
    "flask": "flask",
    "fastapi": "fastapi",
    "pygame": "pygame",
    "pillow": "pillow", "pil": "pillow",
    "screen_brightness": "screen-brightness-control",
    "brightness": "screen-brightness-control",
}


def precheck_dependencies(prompt: str) -> None:
    """
    Scans the prompt for library keywords and warns about missing packages
    BEFORE running any LLM calls, so the user can install early.
    """
    prompt_lower = prompt.lower()
    likely_missing: Set[str] = set()

    for keyword, pypi_name in _PROMPT_TO_PACKAGES.items():
        if keyword in prompt_lower:
            import_name = keyword if keyword != "np" else "numpy"
            # Use importlib to avoid actually importing heavy packages
            if importlib.util.find_spec(import_name.replace("-", "_")) is None:
                likely_missing.add(pypi_name)

    if likely_missing:
        print("\n" + "=" * 60)
        print("[PRE-CHECK] Likely missing packages detected from your prompt:")
        for pkg in sorted(likely_missing):
            print(f"  - {pkg}")
        print("\nYou may want to install them now before generation starts:")
        print(f"  pip install {' '.join(sorted(likely_missing))}")
        print("=" * 60 + "\n")
        logger.info(f"Pre-check detected likely missing packages: {likely_missing}")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """
    Convert a title into a safe filename.
    Example: "Generate Random Grid-Based Maze" -> "generate_random_grid_based_maze"
    """
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text[:50]


def install_packages(mapped_missing: list) -> bool:
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


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def run(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        fast: bool = False,
        auto_install: bool = False,
        refine: bool = False,
    ) -> None:
        """
        Orchestrates the multi-agent code generation pipeline.

        Args:
            prompt (str): User's natural language request.
            output_path (str): Optional explicit output file path.
            fast (bool): If True, skip the planning stage.
            auto_install (bool): If True, install missing packages without prompting.
            refine (bool): If True, load the last session and improve it instead of starting fresh.
        """
        logger.info(f"Supervisor initiated. Target Model: {self.model}")
        logger.info(f"User Prompt: '{prompt}' (Fast: {fast}, AutoInstall: {auto_install}, Refine: {refine})")

        start_time = time.time()
        failed = False

        base_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.join(base_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Refine mode: inject last session context into prompt ---
        previous_session = None
        if refine:
            previous_session = load_last_session()
            if previous_session:
                prev_prompt = previous_session.get("prompt", "")
                prev_code = previous_session.get("code", "")
                print(f"\n[REFINE] Improving last generation: '{prev_prompt[:60]}'\n")
                prompt = (
                    f"Improve and fix the following Python script based on this feedback: {prompt}\n\n"
                    f"Original request: {prev_prompt}\n\n"
                    f"Previous code:\n```python\n{prev_code}\n```"
                )
            else:
                print("[REFINE] No previous session found. Starting fresh generation.")

        # --- Dependency pre-check (before spending LLM compute) ---
        precheck_dependencies(prompt)

        try:
            code_plan = None
            enhanced_prompt = None

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
                print("\n[ENHANCED PROMPT] Engineered Prompt:\n" + "=" * 40)
                print(enhanced_prompt)
                print("=" * 40 + "\n")

                # Step 2: Create Implementation Plan
                logger.info("=== STEP 2: CREATING IMPLEMENTATION PLAN ===")
                code_plan = run_agent_with_retry(plan, enhanced_prompt, model=self.model)
                if not code_plan:
                    raise RuntimeError("Planner agent failed to generate an implementation plan.")
                print("\n[PLAN] Generated Implementation Plan:\n" + "=" * 40)
                print(code_plan)
                print("=" * 40 + "\n")

                # Step 3: Code Generation
                logger.info("=== STEP 3: GENERATING CODE ===")
                code = run_agent_with_retry(generate, code_plan, model=self.model)
                if not code:
                    raise RuntimeError("Generator agent failed to generate code.")

                # Step 3b: Code Review (semantic alignment check)
                logger.info("=== STEP 3b: REVIEWING CODE ===")
                is_aligned, review_issues = review_code(prompt, code, model=self.model)
                if not is_aligned:
                    print("\n[REVIEWER] Code does not fully match your request. Feeding issues back to generator...")
                    print(f"Issues found:\n{review_issues}\n")
                    logger.info("Re-generating code with reviewer feedback...")
                    code = run_agent_with_retry(
                        generate, code_plan, model=self.model,
                        error_message=f"Code Review Issues:\n{review_issues}",
                        previous_code=code,
                    )
                    if not code:
                        raise RuntimeError("Generator agent failed to regenerate code after review.")
                else:
                    logger.info("Code Reviewer: code is aligned with request.")

            # Auto-patch missing sys import
            if "sys.argv" in code and "import sys" not in code:
                logger.info("Auto-patching missing 'sys' import...")
                code = "import sys\n" + code

            # Step 4: Syntax & Dependency Processing (No runtime execution checks)
            logger.info("=== STEP 4: VALIDATING CODE ===")
            last_error = None
            final_code = None

            for iteration in range(1, MAX_SYNTAX_RETRIES + 1):
                logger.info(f"Validation iteration {iteration}/{MAX_SYNTAX_RETRIES}")
                syntax_error, validated_code = check_syntax(code)

                if syntax_error is not None:
                    last_error = syntax_error
                    logger.warning(f"Syntax error (iter {iteration}): {syntax_error}")
                else:
                    import_ok, import_error, external_packages = check_imports(validated_code)
                    if not import_ok:
                        last_error = import_error
                        logger.warning(f"Import check failed (iter {iteration}): {import_error}")
                    else:
                        # Document external dependencies in the file headers
                        if external_packages:
                            mapped_missing = [IMPORT_TO_PACKAGE.get(p, p) for p in external_packages]

                            # Prepend dependency install commands as header comments
                            dep_comment = (
                                "# ==========================================================================\n"
                                "# REQUIRED EXTERNAL DEPENDENCIES\n"
                                "# To run this script, please install the following packages first:\n"
                                f"#   pip install {' '.join(mapped_missing)}\n"
                                "# ==========================================================================\n\n"
                            )
                            validated_code = dep_comment + validated_code

                            print("\n" + "=" * 60)
                            print("[INFO] External Dependencies Detected")
                            print("=" * 60)
                            print(f"The generated code uses these packages: {', '.join(external_packages)}")
                            print("\nTo install them, run:\n")
                            print(f"  pip install {' '.join(mapped_missing)}")
                            print("\nNote: Dependencies have been documented in the file header comments.")
                            print("=" * 60 + "\n")

                            if auto_install:
                                # Run background installation if requested via CLI flag
                                try:
                                    import subprocess
                                    logger.info(f"Auto-installing: {mapped_missing}")
                                    subprocess.run([sys.executable, "-m", "pip", "install"] + mapped_missing, capture_output=True)
                                except Exception as install_err:
                                    logger.warning(f"Background auto-installation failed: {install_err}")

                        final_code = validated_code
                        logger.info("Code passed validation checks.")
                        break

                if iteration < MAX_SYNTAX_RETRIES:
                    logger.info("Re-running Generator with syntax error feedback...")
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

                print("\n[SUCCESS] Final Python Code:\n" + "=" * 40)
                print(final_code)
                print("=" * 40 + "\n")

                # Save the code
                final_path = output_path or ""
                try:
                    if not final_path:
                        base_name = None
                        if not fast and enhanced_prompt:
                            title_match = re.search(r"\*\*Title:\*\*\s*(.*?)(?:\n|$)", enhanced_prompt, re.IGNORECASE)
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

                    # Persist session
                    save_session(
                        prompt=prompt,
                        enhanced_prompt=enhanced_prompt,
                        plan=code_plan,
                        code=final_code,
                        output_path=final_path,
                    )

                except Exception as file_err:
                    logger.error(f"Failed to write generated code: {file_err}")
                    print(f"Warning: Could not save code to file: {file_err}")

        except Exception as e:
            logger.critical(f"Supervisor encountered a critical exception: {e}")
            print(f"\n[CRITICAL FAILURE] {e}")
            print("Please check agent_system.log for details.")
            failed = True

        if failed:
            sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
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
    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="Automatically install missing external packages without prompting."
    )
    parser.add_argument(
        "--refine", "-r",
        type=str,
        metavar="FEEDBACK",
        help="Improve the last generated code. Provide feedback as a string (e.g. 'make it faster')."
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show a list of all previous code generation sessions and exit."
    )

    args = parser.parse_args()

    # --history: just print and exit
    if args.history:
        print_history()
        sys.exit(0)

    # --refine: use feedback as prompt, set refine=True
    if args.refine:
        prompt = args.refine
        refine = True
    elif args.prompt:
        prompt = args.prompt
        refine = False
    else:
        try:
            print("--- Local Multi-Agent Code Generator CLI ---")
            prompt = input("Enter your prompt: ").strip()
            if not prompt:
                print("Prompt cannot be empty. Exiting.")
                sys.exit(1)
            refine = False
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    # Input validation
    validation_error = validate_prompt(prompt)
    if validation_error:
        print(f"[INPUT ERROR] {validation_error}")
        sys.exit(1)

    supervisor = Supervisor(model=args.model)
    supervisor.run(prompt, args.output, args.fast, auto_install=args.install, refine=refine)


if __name__ == "__main__":
    main()
