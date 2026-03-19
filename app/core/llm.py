"""
LLM chain construction for Diya.
Builds context-aware prompts that inject memory, detected language,
and optional web-search results.
"""

from __future__ import annotations

import datetime
import logging

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from app.core import memory as mem_module
from app.utils.config import (
    ASSISTANT_NAME,
    LANG_CODE_TO_NAME,
    LLM_MODEL,
)

logger = logging.getLogger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(lang_code: str, mem: dict) -> str:
    """
    Construct the system prompt dynamically per turn so it always reflects:
    - current detected language
    - latest memory state
    - today's date
    """
    lang_name  = LANG_CODE_TO_NAME.get(lang_code, "English")
    mem_ctx    = mem_module.build_context(mem)
    user_name  = mem.get("user_name") or "there"
    today      = datetime.datetime.now().strftime("%B %d, %Y")

    memory_block = (
        f"\nWhat you remember about the user:\n{mem_ctx}"
        if mem_ctx else ""
    )

    return (
        f"You are {ASSISTANT_NAME}, a warm, intelligent and helpful Indian female voice assistant.\n"
        f"Your personality: friendly, concise, culturally aware, respectful.\n\n"
        f"IMPORTANT RULES:\n"
        f"- Respond ONLY in {lang_name}. If the user switches language, match them immediately.\n"
        f"- Keep every response to 2-3 short sentences — this is a voice interface.\n"
        f"- Do NOT use markdown, bullet points, asterisks, or special characters in your response.\n"
        f"- When greeting the user for the first time, introduce yourself as {ASSISTANT_NAME}.\n"
        f"- Address the user as {user_name}.\n"
        f"- If asked who you are, say you are {ASSISTANT_NAME}, a voice assistant.\n"
        f"{memory_block}\n"
        f"Today is {today}."
    )


# ── Chain factory ─────────────────────────────────────────────────────────────

def get_llm(api_key: str) -> ChatGroq:
    return ChatGroq(model=LLM_MODEL, groq_api_key=api_key)


def build_chain(
    llm: ChatGroq,
    lang_code: str,
    mem: dict,
    search_context: str | None = None,
):
    """
    Return a LangChain LCEL chain for one turn.

    Parameters
    ----------
    llm : ChatGroq
        Instantiated LLM.
    lang_code : str
        Detected language code for this turn.
    mem : dict
        Current memory snapshot.
    search_context : str | None
        Tavily search results to inject, or None.
    """
    system = build_system_prompt(lang_code, mem)
    if search_context:
        system += f"\n\nLive web search results for context:\n{search_context}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def invoke(
    llm: ChatGroq,
    question: str,
    chat_history: list[BaseMessage],
    lang_code: str,
    mem: dict,
    search_context: str | None = None,
) -> str:
    """
    Convenience wrapper: build the right chain and call it.
    Returns the assistant's response string.
    """
    chain    = build_chain(llm, lang_code, mem, search_context)
    response = chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })
    logger.debug("LLM response (%s): %s", lang_code, response[:120])
    return response
