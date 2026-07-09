"""
Tests for the _prompt_uses_files() helper in the Interpreter agent.
Converted from a standalone script to pytest format.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.interpreter import _prompt_uses_files


@pytest.mark.parametrize("prompt, expected", [
    ("real time hand movement detection using camera", False),
    ("read all csv files in a folder and plot data", True),
    ("control screen brightness with hand gestures", False),
    ("parse log files and generate a report", True),
    ("generate a prime number sieve", False),
    ("rename all files in directory", True),
    # Edge cases
    ("movement tracking app", False),   # 'move' inside 'movement' should NOT match
    ("write a sorting algorithm", True), # 'write' is a file keyword
    ("delete duplicate entries", True),  # 'delete' is a file keyword
])
def test_prompt_uses_files(prompt: str, expected: bool) -> None:
    """Verify _prompt_uses_files correctly identifies file-operation prompts."""
    result = _prompt_uses_files(prompt)
    assert result == expected, (
        f"_prompt_uses_files({prompt!r}) returned {result!r}, expected {expected!r}"
    )
