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

# Logging configuration
LOG_FILE = "agent_system.log"
