import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "mistralai/mistral-7b-instruct-v0.3")

# ── Singleton client ──────────────────────────────────────────────────
_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY not set in .env\n"
                "Get free key at: https://build.nvidia.com"
            )
        _client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
        )
        logger.success(f"NVIDIA NIM client ready — model: {NVIDIA_MODEL}")
    return _client


def call_nvidia(
    messages:     list,
    temperature:  float = 0.3,
    max_tokens:   int   = 256,
) -> str:
    """
    Call NVIDIA NIM API and return the response text.
    Handles rate limits with retry.
    """
    client = get_client()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate" in str(e).lower() and attempt < 2:
                wait = (attempt + 1) * 10
                logger.warning(f"Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"NVIDIA API error: {e}")
                return ""
    return ""


def test_connection():
    """Quick test to verify NVIDIA API works."""
    logger.info("Testing NVIDIA NIM connection...")
    try:
        response = call_nvidia([
            {"role": "user", "content": "Say 'OK' in one word."}
        ])
        if response:
            logger.success(f"NVIDIA NIM connected — test response: '{response}'")
            return True
        else:
            logger.error("NVIDIA NIM connection failed")
            return False
    except Exception as e:
        logger.error(f"NVIDIA NIM connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
