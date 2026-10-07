"""
Single place that creates the chat model used by the summarizer, the
extractors and the RAG engine.

Uses Google Gemini through the free Google AI Studio API key
(GOOGLE_API_KEY in .env). Change the models with GEMINI_MODEL and
GEMINI_FALLBACK_MODEL in .env.
"""

import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# The google-genai SDK logs a harmless "Direct use of automatic function
# calling (AFC)..." warning on every call; LangChain doesn't use AFC, so hide it.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# "-latest" aliases always point at Google's current Flash / Flash-Lite models.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")


def _gemini(model: str, temperature: float, retries: int):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,  # lower = more factual, less creative
        max_retries=retries,
    )


def get_llm(temperature: float = 0.3):
    # The free tier often answers 503 "high demand" on the main model;
    # if a call fails, LangChain retries the same prompt on the fallback model.
    # The main model gets no retries so a busy model hands over immediately
    # instead of waiting through backoff delays.
    return _gemini(GEMINI_MODEL, temperature, retries=0).with_fallbacks(
        [_gemini(GEMINI_FALLBACK_MODEL, temperature, retries=2)]
    )
