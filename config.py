# Configuration settings for the Multi-Agent Python Code Generator

# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434"

# Lightweight default model for low RAM systems (e.g., qwen2.5-coder:1.5b, deepseek-coder:1.3b, phi:2.7b)
DEFAULT_MODEL = "qwen2.5-coder:1.5b"

# Agent Temperatures (lower means more deterministic, higher means more creative)
INTERPRETER_TEMP = 0.3
PLANNER_TEMP = 0.2
GENERATOR_TEMP = 0.1

# Retry and loop configurations
MAX_RETRIES = 3
MAX_SYNTAX_RETRIES = 3

# Timeout settings (seconds)
LLM_CALL_TIMEOUT = 120        # Initial LLM API call timeout; increases on each retry
SMOKE_TEST_TIMEOUT = 2        # Max seconds for the subprocess smoke test

# Prompt length guard (approximate tokens — 1 token ≈ 4 characters)
# qwen2.5-coder:1.5b has a ~4K context window; leave headroom for system prompt + output
MAX_PROMPT_TOKENS = 3000

# Logging configuration
LOG_FILE = "agent_system.log"

# Mapping of Python import names to their actual PyPI package names.
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "requests": "requests",
    "mediapipe": "mediapipe",
    "pygame": "pygame",
    "tensorflow": "tensorflow",
    "torch": "torch",
    "screen_brightness_control": "screen-brightness-control",
    "scipy": "scipy",
    "flask": "flask",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "sqlalchemy": "sqlalchemy",
    "aiohttp": "aiohttp",
}
