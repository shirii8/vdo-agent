"""
Single place that creates the chat model used by the summarizer, the
extractors and the RAG engine.

Uses Google Gemini through the free Google AI Studio API key
(GOOGLE_API_KEY in .env). Change the models with GEMINI_MODEL and
GEMINI_FALLBACK_MODELS (comma-separated) in .env.
"""

import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# The google-genai SDK logs a harmless "Direct use of automatic function
# calling (AFC)..." warning on every call; LangChain doesn't use AFC, so hide it.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# "-latest" aliases always point at Google's current Flash / Flash-Lite models.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# Tried in order when the model before it fails. The free tier limits each
# model separately (the main Flash model allows only ~20 requests a day), so
# a chain of models keeps the app working after one of them runs out.
GEMINI_FALLBACK_MODELS = [
    name.strip()
    for name in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash,gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
    ).split(",")
    if name.strip()
]


def _gemini(model: str, temperature: float, retries: int):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,  # lower = more factual, less creative
        max_retries=retries,
    )


def get_llm(temperature: float = 0.3):
    # A failed call (503 "high demand", 429 quota exhausted) is retried with the
    # same prompt on the next model in the chain. No model retries on its own,
    # so a busy or exhausted one hands over immediately instead of waiting
    # through backoff delays; only the last one retries, as the final attempt.
    last = len(GEMINI_FALLBACK_MODELS) - 1
    return _gemini(GEMINI_MODEL, temperature, retries=0).with_fallbacks(
        [
            _gemini(name, temperature, retries=2 if i == last else 0)
            for i, name in enumerate(GEMINI_FALLBACK_MODELS)
        ]
    )
