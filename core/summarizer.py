"""
Summarisation and title generation with Gemini via LangChain LCEL.

`summarize` sends the whole transcript in one call when it fits in one chunk
(about an hour of speech). Longer transcripts use map-reduce:
  1. map    - split into chunks, summarise all chunks in parallel
  2. reduce - merge the partial summaries into one bullet-point summary
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm


def split_transcript(transcript: str) -> list:
    # Gemini reads ~1M tokens, so chunks can be large: 60k chars is roughly an
    # hour of speech. 500-char overlap keeps boundary sentences in both chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=60000, chunk_overlap=500)

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    chunks = split_transcript(transcript)

    # ── Short transcript: one call straight to the final summary ──
    if len(chunks) <= 1:
        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert meeting summarizer. Write a professional "
                    "meeting summary of this transcript in bullet points.",
                ),
                ("human", "{text}"),
            ]
        )
        return (final_prompt | llm | StrOutputParser()).invoke({"text": transcript})

    # ── Map step: summarise every chunk, in parallel ──
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    # .batch runs the calls concurrently instead of one after another.
    chunk_summaries = map_chain.batch([{"text": chunk} for chunk in chunks])

    combined = "\n\n".join(chunk_summaries)

    # ── Reduce step: merge the partial summaries ──
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    # The opening ~2000 chars are enough to name the meeting and keep the call cheap.
    return title_chain.invoke(transcript[:2000])
