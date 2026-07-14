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


# ---------------------------------------------------------------------------
# Conversation Context — interactive agent dialogue tracker
# ---------------------------------------------------------------------------

class ConversationContext:
    """
    Tracks the evolving dialogue between agents during a pipeline run.

    Each agent announces its work and findings via speak(), which prints a
    styled, colour-coded dialogue box to the terminal so the pipeline feels
    like a team actively collaborating.  The accumulated history is also
    available as plain text for injecting back into the LLM on correction
    passes, giving the Generator full conversational context.
    """

    # Agent display style: emoji + ANSI colour code
    _STYLES: dict = {
        "Supervisor":  ("\U0001f9e0", "\033[1;36m"),   # bold cyan
        "Interpreter": ("\u270f\ufe0f ", "\033[1;34m"),  # bold blue
        "Planner":     ("\U0001f4cb", "\033[1;33m"),   # bold yellow
        "Generator":   ("\u2699\ufe0f ", "\033[1;32m"),  # bold green
        "Reviewer":    ("\U0001f50d", "\033[1;35m"),   # bold magenta
        "Checker":     ("\u2705", "\033[1;37m"),        # bold white
    }
    _RESET    = "\033[0m"
    _BOLD     = "\033[1m"
    _BOX_W    = 62

    def __init__(self) -> None:
        self.history: list = []   # list of (agent_name, message) tuples

    def speak(self, agent: str, message: str) -> None:
        """Print a styled dialogue box for the agent and record the message."""
        import textwrap

        self.history.append((agent, message))
        emoji, color = self._STYLES.get(agent, ("\U0001f916", "\033[1m"))
        w = self._BOX_W

        header      = f" {emoji} {agent} "
        pad_left    = (w - len(header)) // 2
        pad_right   = w - len(header) - pad_left

        top    = f"{color}\u2554{'\u2550' * w}\u2557{self._RESET}"
        title  = (f"{color}\u2551{'\u2500' * pad_left}"
                  f"{self._BOLD}{header}{self._RESET}"
                  f"{color}{'\u2500' * pad_right}\u2551{self._RESET}")
        sep    = f"{color}\u2560{'\u2550' * w}\u2563{self._RESET}"
        bottom = f"{color}\u255a{'\u2550' * w}\u255d{self._RESET}"

        lines: list = []
        for raw in message.splitlines():
            wrapped = textwrap.wrap(raw, width=w - 4) or [""]
            lines.extend(wrapped)

        body = [
            f"{color}\u2551{self._RESET}  {ln:<{w - 2}}{color}\u2551{self._RESET}"
            for ln in lines
        ]

        print(f"\n{top}")
        print(title)
        print(sep)
        for bl in body:
            print(bl)
        print(bottom)

    def format_history(self) -> str:
        """Return the full dialogue as plain text for LLM prompt injection."""
        return "\n".join(
            f"[{agent}]: {msg}" for agent, msg in self.history
        )


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
    ) -> None:
        """
        Orchestrates the multi-agent code generation pipeline.

        Args:
            prompt (str): User's natural language request.
            output_path (str): Optional explicit output file path.
            fast (bool): If True, skip the planning stage.
            auto_install (bool): If True, install missing packages.
        """
        logger.info(f"Supervisor initiated. Target Model: {self.model}")
        logger.info(f"User Prompt: '{prompt}' (Fast: {fast}, AutoInstall: {auto_install})")

        start_time = time.time()
        failed = False

        # Initialise shared conversation context for this pipeline run
        ctx = ConversationContext()
        ctx.speak(
            "Supervisor",
            f"Pipeline starting. Model: {self.model} | Fast: {fast} | AutoInstall: {auto_install}\n"
            f"Task: {prompt[:120]}{'...' if len(prompt) > 120 else ''}",
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.join(base_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Dependency pre-check (before spending LLM compute) ---
        precheck_dependencies(prompt)

        try:
            code_plan = None
            enhanced_prompt = None

            if fast:
                logger.info("=== SINGLE-SHOT CODE GENERATION ===")
                ctx.speak("Generator", "Fast mode enabled — skipping Interpreter and Planner. Generating code directly from your prompt...")
                code = run_agent_with_retry(generate_direct, prompt, model=self.model)
                if not code:
                    raise RuntimeError("Generator agent failed to generate code in single-shot mode.")
            else:
                # Step 1: Prompt Engineering
                logger.info("=== STEP 1: PROMPT ENGINEERING ===")
                ctx.speak("Interpreter", "Analysing your request. Expanding it into a detailed, unambiguous technical prompt for the Planner and Generator...")
                enhanced_prompt = run_agent_with_retry(engineer_prompt, prompt, model=self.model)
                if not enhanced_prompt:
                    raise RuntimeError("Interpreter agent failed to engineer a prompt.")
                ctx.speak("Interpreter", "Prompt engineering complete. Handing detailed specification to Planner.")
                print("\n[ENHANCED PROMPT] Engineered Prompt:\n" + "=" * 40)
                print(enhanced_prompt)
                print("=" * 40 + "\n")

                # Step 2: Create Implementation Plan
                logger.info("=== STEP 2: CREATING IMPLEMENTATION PLAN ===")
                ctx.speak("Planner", "Received the technical specification from Interpreter. Designing a step-by-step implementation plan for the Generator...")
                code_plan = run_agent_with_retry(plan, enhanced_prompt, model=self.model)
                if not code_plan:
                    raise RuntimeError("Planner agent failed to generate an implementation plan.")
                ctx.speak("Planner", "Implementation plan ready. Forwarding to Generator.")
                print("\n[PLAN] Generated Implementation Plan:\n" + "=" * 40)
                print(code_plan)
                print("=" * 40 + "\n")

                # Step 3: Code Generation
                logger.info("=== STEP 3: GENERATING CODE ===")
                ctx.speak("Generator", "Received the implementation plan from Planner. Writing Python code...")
                code = run_agent_with_retry(generate, code_plan, model=self.model)
                if not code:
                    raise RuntimeError("Generator agent failed to generate code.")
                ctx.speak("Generator", "First draft complete. Sending code to Reviewer for alignment check.")

                # Step 3b: Code Review — multi-turn agent dialogue loop
                logger.info("=== STEP 3b: REVIEWING CODE ===")
                _MAX_REVIEW_ROUNDS = MAX_RETRIES
                for _review_round in range(1, _MAX_REVIEW_ROUNDS + 1):
                    ctx.speak(
                        "Reviewer",
                        f"[Round {_review_round}/{_MAX_REVIEW_ROUNDS}] Checking whether the code correctly "
                        f"implements the original request...",
                    )
                    is_aligned, review_issues = review_code(prompt, code, model=self.model)
                    if is_aligned:
                        ctx.speak(
                            "Reviewer",
                            "Code is fully aligned with the original request. Approving — "
                            "forwarding to Syntax Checker.",
                        )
                        logger.info("Code Reviewer: code is aligned with request.")
                        break
                    else:
                        ctx.speak(
                            "Reviewer",
                            f"Issues found in Round {_review_round}:\n{review_issues}",
                        )
                        logger.info(f"Review round {_review_round}: issues found, feeding back to Generator...")
                        if _review_round == _MAX_REVIEW_ROUNDS:
                            ctx.speak(
                                "Reviewer",
                                "Maximum review rounds reached. Forwarding best available attempt to Syntax Checker.",
                            )
                            logger.warning("Max review rounds reached without full alignment.")
                            break
                        ctx.speak(
                            "Generator",
                            f"Acknowledged Reviewer feedback (Round {_review_round}). "
                            f"Revising code to address all raised issues...",
                        )
                        code = run_agent_with_retry(
                            generate, code_plan, model=self.model,
                            error_message=f"Code Review Issues:\n{review_issues}",
                            previous_code=code,
                            conversation_history=ctx.format_history(),
                        )
                        if not code:
                            raise RuntimeError("Generator agent failed to regenerate code after review.")

            # Auto-patch missing common standard library imports
            for lib in ["sys", "unittest", "time", "math"]:
                if f"{lib}." in code and f"import {lib}" not in code and f"from {lib}" not in code:
                    logger.info(f"Auto-patching missing '{lib}' import...")
                    code = f"import {lib}\n" + code

            # Step 4: Syntax & Dependency Processing
            logger.info("=== STEP 4: VALIDATING CODE ===")
            last_error = None
            final_code = None

            for iteration in range(1, MAX_SYNTAX_RETRIES + 1):
                logger.info(f"Validation iteration {iteration}/{MAX_SYNTAX_RETRIES}")
                syntax_error, validated_code = check_syntax(code)

                if syntax_error is not None:
                    last_error = syntax_error
                    logger.warning(f"Syntax error (iter {iteration}): {syntax_error}")
                    ctx.speak("Checker", f"Syntax error detected (iteration {iteration}/{MAX_SYNTAX_RETRIES}):\n{syntax_error}")
                else:
                    import_ok, import_error, external_packages = check_imports(validated_code)
                    if not import_ok:
                        last_error = import_error
                        logger.warning(f"Import check failed (iter {iteration}): {import_error}")
                        ctx.speak("Checker", f"Import check failed (iteration {iteration}/{MAX_SYNTAX_RETRIES}):\n{import_error}")
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

                            ctx.speak(
                                "Checker",
                                f"External dependencies detected: {', '.join(external_packages)}\n"
                                f"Run: pip install {' '.join(mapped_missing)}\n"
                                f"(Install commands also documented in the generated file's header.)",
                            )

                            if auto_install:
                                try:
                                    import subprocess
                                    logger.info(f"Auto-installing: {mapped_missing}")
                                    subprocess.run([sys.executable, "-m", "pip", "install"] + mapped_missing, capture_output=True)
                                except Exception as install_err:
                                    logger.warning(f"Background auto-installation failed: {install_err}")

                        final_code = validated_code
                        logger.info("Code passed validation checks.")
                        ctx.speak("Checker", "All validation checks passed. Code is syntactically correct and imports are verified. Handing off to Supervisor.")
                        break

                if iteration < MAX_SYNTAX_RETRIES:
                    logger.info("Re-running Generator with syntax error feedback...")
                    ctx.speak(
                        "Generator",
                        f"Acknowledged error report from Checker (iteration {iteration}). "
                        f"Rewriting code to fix the issue...",
                    )
                    if fast:
                        code = run_agent_with_retry(
                            generate_direct, prompt, model=self.model,
                            error_message=last_error, previous_code=code,
                            conversation_history=ctx.format_history(),
                        )
                    else:
                        code = run_agent_with_retry(
                            generate, code_plan, model=self.model,
                            error_message=last_error, previous_code=code,
                            conversation_history=ctx.format_history(),
                        )
                else:
                    ctx.speak("Checker", "Maximum validation retries exhausted. Unable to produce clean code.")
                    logger.error("Maximum validation retries reached.")

            if final_code is None:
                ctx.speak("Supervisor", "Pipeline failed to produce validated code after all retries.")
                if last_error:
                    print(f"\nLast error:\n{last_error}")
                if code:
                    print("\nHere is the last generated code:\n")
                    print(code)
                failed = True
            else:
                elapsed_time = time.time() - start_time
                logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds.")

                ctx.speak(
                    "Supervisor",
                    f"All agents signed off. Pipeline complete in {elapsed_time:.2f}s. Saving output file...",
                )
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

    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    else:
        try:
            print("--- Local Multi-Agent Code Generator CLI ---")
            prompt = input("Enter your prompt: ").strip()
            if not prompt:
                print("Prompt cannot be empty. Exiting.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    # Input validation
    validation_error = validate_prompt(prompt)
    if validation_error:
        print(f"[INPUT ERROR] {validation_error}")
        sys.exit(1)

    supervisor = Supervisor(model=args.model)
    supervisor.run(prompt, args.output, args.fast, auto_install=args.install)


if __name__ == "__main__":
    main()
