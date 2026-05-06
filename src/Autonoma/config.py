import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "gemma-3-4b"
LLM_TEMPERATURE = 0.1

MAX_CRITIC_ITERATIONS = 3

import sys
VERBOSE_MODE = "--verbose" in sys.argv or os.getenv("VERBOSE_MODE", "false").lower() == "true"

def debug_print(*args, **kwargs):
    if VERBOSE_MODE:
        print(*args, **kwargs)
