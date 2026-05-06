import os
from dotenv import load_dotenv

# Load environment variables (like API keys) from the .env file
load_dotenv()

# --- SECRETS (Loaded from .env) ---
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")

# --- APPLICATION CONFIGURATION ---
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "gemma-3-4b"
LLM_TEMPERATURE = 0.1

MAX_CRITIC_ITERATIONS = 3
