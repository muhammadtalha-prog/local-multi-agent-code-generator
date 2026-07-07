import ast
from utils.logger import logger

def check_syntax(code):
    """
    Parses Python code using the standard AST library to check for syntax errors.
    
    Args:
        code (str): The Python code to check.
        
    Returns:
        tuple: (error_message, code) where error_message is None if valid,
               or a detailed string description of the SyntaxError if invalid.
    """
    logger.info("Running Syntax Checker Agent...")
    if not code:
        logger.warning("Empty code passed to Syntax Checker.")
        return "The generated code is empty.", None
        
    try:
        ast.parse(code)
        logger.info("Syntax check passed. No errors found.")
        return None, code
    except SyntaxError as e:
        # Extract detailed error details
        error_msg = e.msg or "syntax error"
        line_no = e.lineno
        offset = e.offset
        text = e.text or ""
        
        # Build a helpful pointer illustration if we have line text and offset
        pointer = ""
        if text and offset is not None:
            # text might contain trailing newlines, clean it up
            text_cleaned = text.rstrip('\r\n')
            # build spacing up to offset
            spacing = " " * max(0, offset - 1)
            pointer = f"\nCode Line:\n{text_cleaned}\n{spacing}^"
            
        full_error = (
            f"SyntaxError: {error_msg} (at line {line_no}, offset {offset}){pointer}"
        )
        
        logger.warning(f"Syntax check failed: {full_error}")
        return full_error, None
    except Exception as e:
        logger.error(f"Unexpected error in AST parsing: {e}")
        return f"Unexpected AST validation error: {str(e)}", None
