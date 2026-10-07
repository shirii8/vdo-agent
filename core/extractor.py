"""
Extracts structured information from a meeting transcript:
action items, key decisions, and open questions.

Each extractor is the same LCEL chain (prompt -> Gemini -> string) with a
different system prompt, built by `build_chain`.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm


def build_chain(system_prompt: str):
    """Build a chain: raw transcript string -> {"text": ...} -> prompt -> LLM -> str."""
    # Low temperature (0.2): extraction should be factual, not creative.
    llm = get_llm(temperature=0.2)
    return (
        RunnablePassthrough()
        # Wrap the plain string into the dict the prompt template expects.
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        # Turn the AIMessage into a plain string.
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """Return a numbered list of tasks with owner and deadline."""
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    """Return a numbered list of decisions made in the meeting."""
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    """Return a numbered list of unresolved questions / follow-ups."""
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)
