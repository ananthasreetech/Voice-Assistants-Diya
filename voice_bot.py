"""
Diya - Indian Voice Assistant (English)
Run: streamlit run voice_bot.py
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import datetime
import json
import logging
import os
import pathlib
import tempfile

import edge_tts
import groq as groq_sdk
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────

ASSISTANT_NAME  = "Diya"
ASSISTANT_ICON  = "🪔"
LLM_MODEL       = "llama-3.3-70b-versatile"
WHISPER_MODEL   = "whisper-large-v3-turbo"
TTS_VOICE       = "en-IN-NeerjaNeural"          # Indian English female
MEMORY_FILE     = str(pathlib.Path(__file__).parent / "diya_memory.json")

SEARCH_KEYWORDS: list[str] = [
    "latest", "recent", "today", "news", "current", "now", "trending",
    "update", "released", "live", "price", "weather", "score",
    "who is", "when did", "how much", "where is", "located", "place",
    "area", "city", "address", "near", "directions", "location",
    "neighbourhood", "street", "road", "locality", "pincode",
]

STATE_UI: dict[str, tuple[str, str]] = {
    "ready":    ("🟢", "Ready — tap the mic to speak"),
    "thinking": ("🟡", "Thinking..."),
    "speaking": ("🔵", "Diya is speaking — tap Stop to interrupt"),
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Diya — Voice Assistant", page_icon=ASSISTANT_ICON)
st.title(f"{ASSISTANT_ICON} {ASSISTANT_NAME}")
st.caption("Your Indian voice assistant — speak to me in English!")

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "chat_history":  [],
    "pending_tts":   None,
    "api_key":       "",
    "tavily_key":    "",
    "recorder_key":  0,
    "diya_state":    "ready",
    "memory":        None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Memory ────────────────────────────────────────────────────────────────────

def _default_memory() -> dict:
    return {
        "user_name":          None,
        "preferences":        {},
        "relationships":      {},
        "past_topics":        [],
        "conversation_count": 0,
        "last_seen":          None,
    }

def load_memory() -> dict:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = _default_memory()
        base.update(data)
        return base
    except FileNotFoundError:
        return _default_memory()
    except Exception:
        return _default_memory()

def save_memory(mem: dict) -> None:
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logging.warning("Could not save memory: %s", exc)

def _safe_json(obj) -> str:
    """Serialise to JSON with curly braces replaced so they never break prompts."""
    return json.dumps(obj, ensure_ascii=False).replace("{", "(").replace("}", ")")

def build_memory_context(mem: dict) -> str:
    lines: list[str] = []
    if mem.get("user_name"):
        lines.append(f"User name: {mem['user_name']}")
    if mem.get("preferences"):
        lines.append(f"Known preferences: {_safe_json(mem['preferences'])}")
    if mem.get("relationships"):
        lines.append(f"Known relationships: {_safe_json(mem['relationships'])}")
    recent = mem.get("past_topics", [])[-8:]
    if recent:
        lines.append(f"Topics discussed recently: {', '.join(recent)}")
    if mem.get("last_seen"):
        lines.append(f"Last conversation: {mem['last_seen']}")
    return "\n".join(lines)

def update_memory(mem: dict, user_query: str, response: str, llm) -> dict:
    """Ask LLM to extract new personal info and merge it into memory."""
    prompt = (
        "Extract any NEW personal information from this conversation to remember about the user.\n"
        "Already known: " + _safe_json(mem) + "\n"
        "User said: " + user_query + "\n"
        "Assistant said: " + response + "\n\n"
        "Return ONLY a JSON object with keys: "
        "user_name (string), preferences (dict), relationships (dict), past_topics (list).\n"
        "For user_name: extract ONLY if the user explicitly states their name "
        "(e.g. 'my name is ...', 'call me ...'). Preserve EXACT spelling.\n"
        "If nothing new, return exactly: {}"
    )
    try:
        result  = llm.invoke(prompt)
        raw     = result.content.strip().strip("```json").strip("```").strip()
        updates = json.loads(raw)
        if updates.get("user_name"):
            mem["user_name"] = updates["user_name"]
        if updates.get("preferences"):
            mem.setdefault("preferences", {}).update(updates["preferences"])
        if updates.get("relationships"):
            mem.setdefault("relationships", {}).update(updates["relationships"])
        if updates.get("past_topics"):
            existing = set(mem.get("past_topics", []))
            for t in updates["past_topics"]:
                if t not in existing:
                    mem.setdefault("past_topics", []).append(t)
        mem["conversation_count"] = mem.get("conversation_count", 0) + 1
        mem["last_seen"] = datetime.datetime.now().strftime("%B %d, %Y %H:%M")
        save_memory(mem)
    except Exception as exc:
        logging.debug("Memory extraction skipped: %s", exc)
    return mem

if st.session_state.memory is None:
    st.session_state.memory = load_memory()

# ── API key resolution ────────────────────────────────────────────────────────

def _resolve_keys() -> None:
    for state_key, env_name in [
        ("api_key",    "GROQ_API_KEY"),
        ("tavily_key", "TAVILY_API_KEY"),
    ]:
        if st.session_state[state_key]:
            continue
        try:
            v = st.secrets[env_name]
            if v:
                st.session_state[state_key] = v.strip()
                continue
        except Exception:
            pass
        load_dotenv()
        v = os.getenv(env_name, "").strip()
        if v:
            st.session_state[state_key] = v

_resolve_keys()

if not st.session_state.api_key or not st.session_state.tavily_key:
    st.markdown("### 🔑 Enter your API keys to meet Diya")
    st.markdown(
        "Free Groq key → [console.groq.com](https://console.groq.com) · "
        "Free Tavily key → [app.tavily.com](https://app.tavily.com)"
    )
    c1, c2 = st.columns(2)
    with c1:
        g = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    with c2:
        t = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")
    if st.button("Save & Start", use_container_width=True):
        if g.strip() and t.strip():
            st.session_state.api_key    = g.strip()
            st.session_state.tavily_key = t.strip()
            st.rerun()
        else:
            st.error("Both keys are required.")
    st.stop()

api_key    = st.session_state.api_key
tavily_key = st.session_state.tavily_key
os.environ["TAVILY_API_KEY"] = tavily_key

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatGroq(model=LLM_MODEL, groq_api_key=api_key)

def build_system_prompt() -> str:
    mem       = st.session_state.memory
    mem_ctx   = build_memory_context(mem)
    user_name = mem.get("user_name") or "there"
    today     = datetime.datetime.now().strftime("%B %d, %Y")
    mem_block = f"\nWhat you remember about the user:\n{mem_ctx}" if mem_ctx else ""

    return (
        f"You are {ASSISTANT_NAME}, a warm, intelligent and helpful Indian female voice assistant.\n"
        f"Your personality: friendly, concise, culturally aware, respectful.\n\n"
        f"RULES:\n"
        f"1. Always respond in clear, natural English.\n"
        f"2. Keep every response to 2-3 short sentences. This is a voice interface — be concise.\n"
        f"3. No markdown, no bullet points, no asterisks, no special characters.\n"
        f"4. ADDRESSING — READ THIS CAREFULLY:\n"
        f"   - The primary user is {user_name}. Use their name when speaking TO or ABOUT them.\n"
        f"   - When the user asks you to greet, wish, or speak TO a DIFFERENT person by name,\n"
        f"     your ENTIRE response must be directed at THAT person. Do NOT end with {user_name}.\n"
        f"   - BAD:  'Good luck Khevanch, I hope you do well, Sreeni.'\n"
        f"   - GOOD: 'Good luck Khevanch, I hope you do brilliantly in your exams!'\n"
        f"   - The name {user_name} must NEVER appear in a message addressed to someone else.\n"
        f"5. You are {ASSISTANT_NAME}. Introduce yourself only on the very first greeting.\n"
        f"6. If you don't know something, say so honestly rather than guessing.\n"
        f"{mem_block}\n"
        f"Today is {today}."
    )

def get_llm_response(user_query: str, search_context: str | None) -> str:
    """
    Invoke the LLM using a direct message list — no ChatPromptTemplate —
    so curly braces in memory data can never crash the prompt.
    """
    system = build_system_prompt()
    if search_context:
        system += f"\n\nLive web search results (use these to answer accurately):\n{search_context}"

    messages = [SystemMessage(content=system)]
    messages.extend(st.session_state.chat_history)
    messages.append(HumanMessage(content=user_query))

    result = llm.invoke(messages)
    return result.content.strip()

# ── Transcription ─────────────────────────────────────────────────────────────

# Whisper hallucinates these phrases when it receives silence or background noise.
# Any transcription that exactly matches or starts with one of these is rejected.
_WHISPER_HALLUCINATIONS: set[str] = {
    "you", "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thank you.", "thanks.", "bye", "bye.", "goodbye", "goodbye.",
    "please subscribe", "like and subscribe", "see you next time",
    "uh", "um", "hmm", "hm", "ah", "oh",
    "order of p1", "order of pi", "i",
}

def _is_hallucination(text: str) -> bool:
    """
    Return True if the transcription is a known Whisper silence-hallucination.
    Checks:
      1. Exact match against known hallucinated phrases (case-insensitive).
      2. Very short result (1-2 words, under 10 chars) with no real content.
      3. Repetitive filler like "..." or "......"
    """
    cleaned = text.strip().lower().rstrip(".")
    # Exact match
    if cleaned in _WHISPER_HALLUCINATIONS:
        return True
    # Single character or empty
    if len(cleaned) <= 2:
        return True
    # Pure punctuation / dots
    if all(c in "., !?-_" for c in cleaned):
        return True
    return False


def transcribe(audio_bytes: bytes) -> str:
    """
    Transcribe audio with explicit English hint.
    Returns empty string if Whisper returns silence hallucination.
    """
    client = groq_sdk.Groq(api_key=api_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                response_format="text",
                language="en",
            )
        text = (result or "").strip()
        if _is_hallucination(text):
            logging.debug("Whisper hallucination rejected: %s", text)
            return ""
        return text
    finally:
        os.unlink(tmp_path)

# ── TTS ───────────────────────────────────────────────────────────────────────

async def _tts_async(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)

def synthesize(text: str) -> bytes:
    def _run():
        return asyncio.run(_tts_async(text))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()

def autoplay_audio(audio_bytes: bytes) -> None:
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        '<audio id="diya-audio" autoplay controls style="width:100%">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        "</audio>",
        unsafe_allow_html=True,
    )

STOP_AUDIO_JS = (
    "<script>var a=document.getElementById('diya-audio');"
    "if(a){a.pause();a.currentTime=0;}</script>"
)

# ── Web search ────────────────────────────────────────────────────────────────

def should_search(q: str) -> bool:
    return any(kw in q.lower() for kw in SEARCH_KEYWORDS)

def web_search(query: str) -> str:
    try:
        tool    = TavilySearchResults(max_results=3)
        results = tool.invoke({"query": query})
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Summary: {r.get('content', '').strip()}"
            )
        return "\n\n".join(parts)
    except Exception as exc:
        logging.warning("Web search failed: %s", exc)
        return ""

# ── State indicator ───────────────────────────────────────────────────────────

icon, label = STATE_UI[st.session_state.diya_state]
user_name   = st.session_state.memory.get("user_name")
name_badge  = f" · Hi, {user_name}!" if user_name else ""

st.markdown(
    f'<div style="display:flex;align-items:center;gap:10px;padding:6px 14px;'
    f'border-radius:20px;border:1px solid var(--color-border-tertiary);'
    f'width:fit-content;margin:8px 0">'
    f'<span style="font-size:12px">{icon}</span>'
    f'<span style="font-size:13px;color:var(--color-text-secondary)">{label}</span>'
    f'<span style="font-size:11px;color:var(--color-text-tertiary)">{name_badge}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"🎤 {msg.content}")
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(f"{ASSISTANT_ICON} {msg.content}")

# ── Play pending TTS (saved before last rerun, played now) ────────────────────

if st.session_state.pending_tts and st.session_state.diya_state == "speaking":
    autoplay_audio(st.session_state.pending_tts)
    st.session_state.pending_tts = None

    # Barge-in: stop audio and open mic immediately
    if st.button("🛑 Stop & Speak", use_container_width=True, type="primary"):
        st.markdown(STOP_AUDIO_JS, unsafe_allow_html=True)
        st.session_state.diya_state    = "ready"
        st.session_state.recorder_key += 1
        st.rerun()

    st.session_state.diya_state = "ready"

# ── Mic input ─────────────────────────────────────────────────────────────────

st.divider()
st.markdown("#### 🎙️ Tap the mic and speak")

audio_bytes = audio_recorder(
    text="",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_size="2x",
    pause_threshold=2.0,
    key=f"recorder_{st.session_state.recorder_key}",
)

if audio_bytes:
    st.session_state.diya_state = "thinking"

    # 1. Transcribe
    with st.spinner("Transcribing..."):
        try:
            user_query = transcribe(audio_bytes)
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            st.session_state.diya_state    = "ready"
            st.session_state.recorder_key += 1
            st.stop()

    if not user_query:
        st.warning("No speech detected — please try again.")
        st.session_state.diya_state    = "ready"
        st.session_state.recorder_key += 1
        st.rerun()

    # 2. Web search if needed
    search_context: str | None = None
    if should_search(user_query):
        with st.spinner("🔍 Searching the web..."):
            result = web_search(user_query)
            search_context = result or None

    # 3. LLM response
    with st.spinner(f"{ASSISTANT_NAME} is thinking..."):
        try:
            response = get_llm_response(user_query, search_context)
        except Exception as exc:
            st.error(f"LLM error: {exc}")
            st.session_state.diya_state    = "ready"
            st.session_state.recorder_key += 1
            st.stop()

    # 4. TTS — save to session state, played after rerun
    with st.spinner("Generating voice..."):
        try:
            st.session_state.pending_tts = synthesize(response)
        except Exception as exc:
            st.warning(f"Voice unavailable: {exc}")
            st.session_state.pending_tts = None

    # 5. Update memory (best-effort)
    try:
        st.session_state.memory = update_memory(
            st.session_state.memory, user_query, response, llm
        )
    except Exception:
        pass

    # 6. Append to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    st.session_state.diya_state    = "speaking"
    st.session_state.recorder_key += 1
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
if st.button("🗑️ Clear chat", use_container_width=True):
    st.session_state.chat_history  = []
    st.session_state.pending_tts   = None
    st.session_state.diya_state    = "ready"
    st.session_state.recorder_key += 1
    st.rerun()
