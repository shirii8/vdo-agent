"""
RAG (Retrieval-Augmented Generation) chat over the meeting transcript.

Flow of one question:
  question -> hybrid retriever (keyword + vector search, plus neighbouring
              chunks; see core/retriever.py)
           -> format_docs (chunks labelled with their [mm:ss] timestamps)
           -> prompt (answer ONLY from context, cite timestamps, else abstain)
           -> Gemini -> {"answer": str, "sources": [Document, ...]}

The sources are returned alongside the answer so the UI can show exactly
which transcript snippets the answer was grounded in.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from core.llm import get_llm
from core.vector_store import build_vector_store
from core.retriever import HybridRetriever, docs_from_store

# Exact sentence the model must use when the transcript doesn't contain the answer.
NOT_FOUND = "I could not find this information in the video transcript."

RAG_SYSTEM_PROMPT = (
    "You are a meeting assistant. Answer the user's question using ONLY the "
    "transcript excerpts below. Do not use outside knowledge, even if you know "
    "the answer.\n\n"
    "Rules:\n"
    "- Cite the timestamp of every excerpt you used, in square brackets exactly as "
    "written, e.g. [03:12]. Put the citation right after the sentence it supports.\n"
    "- The excerpts are consecutive pieces of the transcript, so an explanation "
    "may be spread over several of them: read them together and answer from "
    "whatever they do say about the topic, even if it is only part of the picture.\n"
    f'- Only when none of the excerpts is about the question, reply exactly: "{NOT_FOUND}" '
    "and nothing else.\n"
    "- Be concise. If quoting someone, say so.\n\n"
    "Transcript excerpts:\n{context}"
)


def format_timestamp(seconds: float) -> str:
    """75.4 -> '01:15';  3725 -> '1:02:05'."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_docs(docs):
    """Label each retrieved chunk with its start timestamp so the model can cite it."""
    return "\n\n".join(
        f"[{format_timestamp(doc.metadata.get('start', 0))}] {doc.page_content}" for doc in docs
    )


def _make_chain(retriever):
    """
    Question string -> {"answer": str, "sources": [Document]}.
    `retriever` is anything that maps a question to Documents: a function such
    as HybridRetriever.search, or a LangChain retriever.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    answer_chain = (
        # Turn the retrieved Documents into the labelled context string.
        RunnablePassthrough.assign(context=lambda x: format_docs(x["sources"]))
        | prompt
        | get_llm(temperature=0.1)  # low temperature: stick to the excerpts
        | StrOutputParser()
    )

    # Step 1: retrieve once and keep the docs ("sources") next to the question.
    # Step 2: add "answer", generated from those same docs.
    return RunnableParallel(
        sources=retriever, question=RunnablePassthrough()
    ) | RunnablePassthrough.assign(answer=answer_chain)


def build_rag_chain(segments: list, k: int = 4):
    """Index a new transcript's timestamped segments and return a chain to query it."""
    vector_store = build_vector_store(segments)
    # The keyword index needs the chunk texts; read them back from the store so
    # both searches work on exactly the same chunks.
    retriever = HybridRetriever(vector_store, docs_from_store(vector_store), k=k)
    return _make_chain(retriever.search)


def ask_question(rag_chain, question: str) -> dict:
    """Return {"answer": str, "sources": [Document]} for one question."""
    print(f"Question : {question}")
    result = rag_chain.invoke(question)
    print(f"answer :{result['answer']}")
    return {"answer": result["answer"], "sources": result["sources"]}
