"""Shared Azure OpenAI client and helpers used by all agents."""

import json
import os
from openai import AzureOpenAI

def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


ENDPOINT = _get_required_env("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
SUBSCRIPTION_KEY = _get_required_env("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)


def call_extraction_agent(system_prompt: str, user_prompt: str, ocr_json_str: str) -> str:
    """Send OCR JSON to an extraction agent and return raw response text."""
    try:
        completion = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + ocr_json_str},
            ],
            temperature=1.0,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        return json.dumps(
            {"error": "extraction_failed", "message": str(e)},
            ensure_ascii=False,
        )


def maybe_parse_json(text: str) -> object:
    try:
        return json.loads(text)
    except Exception:
        return text
