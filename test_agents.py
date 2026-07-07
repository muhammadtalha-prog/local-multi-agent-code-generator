import sys
import unittest
from config import DEFAULT_MODEL
from utils.logger import logger
from utils.llm_client import call_llm
from agents.syntax_checker import check_syntax

class TestCodeGeneratorAgents(unittest.TestCase):
    
    def test_syntax_checker_valid(self):
        """Test syntax checker with syntactically valid code."""
        valid_code = (
            "def hello_world():\n"
            "    print('Hello, world!')\n"
            "\n"
            "hello_world()\n"
        )
        error_msg, code = check_syntax(valid_code)
        self.assertIsNone(error_msg)
        self.assertEqual(code, valid_code)
        
    def test_syntax_checker_invalid(self):
        """Test syntax checker with syntactically invalid code."""
        invalid_code = (
            "def hello_world():\n"
            "    print('Hello, world!'\n"  # missing closing paren
        )
        error_msg, code = check_syntax(invalid_code)
        self.assertIsNotNone(error_msg)
        self.assertIn("SyntaxError", error_msg)
        self.assertIsNone(code)
        
    def test_ollama_connectivity(self):
        """Test that we can connect to Ollama and make a fast query."""
        print(f"\n[Test] Testing Ollama connectivity using model: {DEFAULT_MODEL}...")
        try:
            response = call_llm("Respond with exactly 'Ollama OK' and nothing else.", temperature=0.1)
            response_clean = response.strip()
            print(f"[Test] Ollama response: '{response_clean}'")
            self.assertTrue(len(response_clean) > 0)
        except Exception as e:
            self.fail(f"Could not reach Ollama: {e}. Make sure 'ollama serve' is running!")

if __name__ == "__main__":
    unittest.main()
