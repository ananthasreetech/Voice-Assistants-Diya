"""
Diya - Indian Voice Assistant (English only)
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
import time

import edge_tts
import groq as groq_sdk
import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────

ASSISTANT_NAME       = "Diya"
ASSISTANT_ICON       = "🪔"
LLM_MODEL            = "llama-3.3-70b-versatile"
WHISPER_MODEL        = "whisper-large-v3-turbo"
TTS_VOICE            = "en-IN-NeerjaNeural"          # Indian English female
MEMORY_FILE          = str(pathlib.Path(__file__).parent / "diya_memory.json")
INACTIVITY_TIMEOUT   = 180                           # seconds before recorder reset

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
    "speaking": ("🔵", "Diya is speaking"),
}

_WHISPER_HALLUCINATIONS: set[str] = {
    "you", "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thank you.", "thanks.", "bye", "bye.", "goodbye", "goodbye.",
    "please subscribe", "like and subscribe", "see you next time",
    "uh", "um", "hmm", "hm", "ah", "oh", "order of p1", "order of pi", "i",
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Diya — Voice Assistant", page_icon=ASSISTANT_ICON)
st.title(f"{ASSISTANT_ICON} {ASSISTANT_NAME}")
st.caption("Your Indian voice assistant — speak to me in English!")

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "chat_history":   [],
    "pending_tts":    None,
    "api_key":        "",
    "tavily_key":     "",
    "recorder_key":   0,
    "diya_state":     "ready",
    "memory":         None,
    "continuous":     False,
    "last_activity":  time.time(),   # tracks last user speech for inactivity reset
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

def update_memory_bg(mem: dict, user_query: str, response: str, llm) -> None:
    """Run memory update — called in a background thread."""
    prompt = (
        "Extract any NEW personal information from this conversation to remember about the user.\n"
        "Already known: " + _safe_json(mem) + "\n"
        "User said: " + user_query + "\n"
        "Assistant said: " + response + "\n\n"
        "Return ONLY a JSON object with keys: "
        "user_name (string), preferences (dict), relationships (dict), past_topics (list).\n"
        "For user_name: extract ONLY if the user explicitly states their name. "
        "Preserve EXACT spelling.\n"
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
        st.session_state.memory = mem
    except Exception as exc:
        logging.debug("Memory extraction skipped: %s", exc)

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
llm = ChatGroq(model=LLM_MODEL, groq_api_key=api_key)

# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    mem       = st.session_state.memory
    mem_ctx   = build_memory_context(mem)
    user_name = mem.get("user_name") or "there"
    now       = datetime.datetime.now()
    today     = now.strftime("%B %d, %Y")
    hour      = now.hour
    if 5 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "night"
    current_time = now.strftime("%I:%M %p")
    mem_block = f"\nWhat you remember about the user:\n{mem_ctx}" if mem_ctx else ""

    return (
        f"You are {ASSISTANT_NAME}, a warm, intelligent and helpful Indian female voice assistant.\n"
        f"Your personality: friendly, empathetic, concise, culturally aware, respectful.\n\n"

        f"RULES:\n"
        f"1. LANGUAGE — CRITICAL: You ONLY speak and respond in English. This is absolute.\n"
        f"   You CANNOT speak Tamil, Hindi, Telugu, Kannada or any other language.\n"
        f"   Your voice output is English-only (en-IN). If asked to speak another language:\n"
        f"   - Do NOT attempt it. Do NOT pretend you tried.\n"
        f"   - Politely explain: 'I currently only support English, but I can understand\n"
        f"     questions about other languages and answer them in English.'\n\n"

        f"2. LENGTH: 2-3 SHORT sentences max. Voice interface — be concise.\n"
        f"3. FORMAT: No markdown, no bullet points, no asterisks, no special characters.\n"

        f"4. ADDRESSING:\n"
        f"   - Primary user is {user_name}. Use their name occasionally, not every sentence.\n"
        f"   - When speaking TO someone else, address ONLY that person — never append {user_name}.\n"
        f"   - BAD: 'Good luck Khevanch, Sreeni.' GOOD: 'Good luck Khevanch!'\n\n"

        f"5. CONVERSATION:\n"
        f"   - This is an ongoing conversation — NEVER treat any message as a fresh start.\n"
        f"   - Respond naturally to small talk, greetings, and emotions.\n"
        f"   - TIME: It is currently {current_time} ({time_of_day}).\n"
        f"     Use correct greeting: morning before 12pm, afternoon 12-5pm,\n"
        f"     evening after 5pm, good night after 9pm.\n"
        f"   - If called wrong name, correct ONCE briefly and continue — never reset context.\n"
        f"   - Mid-conversation: NEVER say 'How can I assist you today?' as if starting fresh.\n\n"

        f"6. IDENTITY: You are {ASSISTANT_NAME}. Introduce yourself ONLY on the first message.\n"
        f"7. HONESTY: Say you don't know rather than guessing.\n"
        f"{mem_block}\n"
        f"Today is {today}, {current_time}."
    )

def get_llm_response(user_query: str, search_context: str | None) -> str:
    """Single clean LLM call — no streaming complexity."""
    system = build_system_prompt()
    if search_context:
        system += f"\n\nLive web search results:\n{search_context}"

    messages = [SystemMessage(content=system)]
    messages.extend(st.session_state.chat_history)
    messages.append(HumanMessage(content=user_query))

    result = llm.invoke(messages)
    return result.content.strip()

def summarize_conversation() -> str:
    if not st.session_state.chat_history:
        return "No conversation to summarize yet."
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Diya'}: {m.content}"
        for m in st.session_state.chat_history
    )
    result = llm.invoke(
        "Summarize this voice conversation in 3-5 clear sentences. "
        "Capture the main topics, key points, and any important information.\n\n"
        f"Conversation:\n{history_text}"
    )
    return result.content.strip()

# ── Transcription ─────────────────────────────────────────────────────────────

def _is_hallucination(text: str) -> bool:
    cleaned = text.strip().lower().rstrip(".")
    if cleaned in _WHISPER_HALLUCINATIONS:
        return True
    if len(cleaned) <= 2:
        return True
    if all(c in "., !?-_" for c in cleaned):
        return True
    return False

def transcribe(audio_bytes: bytes) -> str:
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
            logging.debug("Hallucination rejected: %s", text)
            return ""
        return text
    finally:
        os.unlink(tmp_path)

# ── TTS — single clean call, no streaming complexity ──────────────────────────

async def _tts_async(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)

def synthesize(text: str) -> bytes:
    """Single TTS call in a dedicated thread to avoid asyncio conflicts."""
    def _run():
        return asyncio.run(_tts_async(text))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=30)

# ── Audio player via components.html (scripts execute here, not in markdown) ──

def autoplay_audio(audio_bytes: bytes, auto_restart: bool = False) -> None:
    """
    Render audio player using st.components.v1.html so <script> tags execute.
    st.markdown strips scripts for security — this was why auto-restart never worked.
    When auto_restart=True the onended event clicks the recorder mic button.
    """
    b64 = base64.b64encode(audio_bytes).decode()

    on_ended = ""
    if auto_restart:
        on_ended = """
        audio.onended = function() {
            setTimeout(function() {
                try {
                    var frames = window.parent.document.querySelectorAll('iframe');
                    for (var i = 0; i < frames.length; i++) {
                        var f = frames[i];
                        if (f === window.frameElement) continue;
                        try {
                            var doc = f.contentDocument || f.contentWindow.document;
                            if (!doc) continue;
                            var btn = doc.querySelector('button');
                            if (btn) { btn.click(); break; }
                        } catch(e) {}
                    }
                } catch(e) {}
            }, 600);
        };
        """

    html = f"""
    <audio id="diya-audio" autoplay controls
           style="width:100%;border-radius:8px;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
    (function() {{
        var audio = document.getElementById('diya-audio');
        if (!audio) return;
        {on_ended}
    }})();
    </script>
    """
    components.html(html, height=70)

def stop_audio_js() -> None:
    components.html(
        """<script>
        try {
            var frames = window.parent.document.querySelectorAll('iframe');
            frames.forEach(function(f) {
                try {
                    var doc = f.contentDocument || f.contentWindow.document;
                    var a = doc && doc.getElementById('diya-audio');
                    if (a) { a.pause(); a.currentTime = 0; }
                } catch(e) {}
            });
        } catch(e) {}
        </script>""",
        height=0,
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

# ── Inactivity check ──────────────────────────────────────────────────────────
# FIX 3: After INACTIVITY_TIMEOUT seconds of silence, reset the recorder widget
# by bumping recorder_key. This remounts a fresh recorder so it is responsive.
# A banner tells the user the mic was reset and they can tap to continue.

def check_inactivity() -> None:
    if st.session_state.diya_state != "ready":
        return
    elapsed = time.time() - st.session_state.last_activity
    if elapsed >= INACTIVITY_TIMEOUT:
        st.session_state.recorder_key += 1
        st.session_state.last_activity = time.time()
        st.info(
            f"🎙️ Mic refreshed after {int(elapsed // 60)} min of silence — tap the green button to continue.",
            icon="🔄",
        )

# ── Top bar ───────────────────────────────────────────────────────────────────

check_inactivity()

top_col1, top_col2 = st.columns([3, 1])

with top_col1:
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

with top_col2:
    cont_label = "🔁 ON" if st.session_state.continuous else "🔁 Continuous"
    if st.button(cont_label, use_container_width=True,
                 type="primary" if st.session_state.continuous else "secondary"):
        st.session_state.continuous = not st.session_state.continuous
        st.rerun()

if st.session_state.continuous:
    st.caption("🔁 Continuous mode — Diya listens automatically after each response.")

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"🎤 {msg.content}")
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(f"{ASSISTANT_ICON} {msg.content}")

# ── Play pending TTS ──────────────────────────────────────────────────────────

if st.session_state.pending_tts and st.session_state.diya_state == "speaking":
    autoplay_audio(
        st.session_state.pending_tts,
        auto_restart=st.session_state.continuous,
    )
    st.session_state.pending_tts = None

    if st.button("🛑 Stop & Speak", use_container_width=True, type="primary"):
        stop_audio_js()
        st.session_state.diya_state    = "ready"
        st.session_state.recorder_key += 1
        st.rerun()

    st.session_state.diya_state = "ready"

# ── Mic — only mounted in ready state so it never picks up Diya's own voice ───

st.divider()

if st.session_state.diya_state == "ready":
    if st.session_state.continuous:
        st.markdown("#### 🎙️ Listening — speak when ready")
    else:
        st.markdown("#### 🎙️ Tap the mic and speak")

    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#2ecc71",
        icon_size="2x",
        pause_threshold=2.0,
        key=f"recorder_{st.session_state.recorder_key}",
    )
else:
    audio_bytes = None
    st.markdown(
        f"#### {'🟡 Thinking...' if st.session_state.diya_state == 'thinking' else '🔵 Speaking...'}"
    )

# ── Process recording ─────────────────────────────────────────────────────────

if audio_bytes:
    # Update activity timestamp
    st.session_state.last_activity = time.time()
    st.session_state.diya_state    = "thinking"

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
        with st.spinner("🔍 Searching..."):
            result = web_search(user_query)
            search_context = result or None

    # 3. LLM response — single clean call
    with st.spinner(f"{ASSISTANT_NAME} is thinking..."):
        try:
            response = get_llm_response(user_query, search_context)
        except Exception as exc:
            st.error(f"LLM error: {exc}")
            st.session_state.diya_state    = "ready"
            st.session_state.recorder_key += 1
            st.stop()

    # 4. TTS — single clean call (removed streaming complexity that caused 180s)
    with st.spinner("Generating voice..."):
        try:
            st.session_state.pending_tts = synthesize(response)
        except Exception as exc:
            st.warning(f"Voice unavailable: {exc}")
            st.session_state.pending_tts = None

    # 5. Memory update in background — never blocks the rerun
    mem_snapshot = dict(st.session_state.memory)
    executor     = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(update_memory_bg, mem_snapshot, user_query, response, llm)
    executor.shutdown(wait=False)

    # 6. Append to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    st.session_state.diya_state    = "speaking"
    st.session_state.recorder_key += 1
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history  = []
        st.session_state.pending_tts   = None
        st.session_state.diya_state    = "ready"
        st.session_state.recorder_key += 1
        st.session_state.last_activity = time.time()
        st.rerun()

with col2:
    if st.button("📋 Summarize", use_container_width=True):
        if st.session_state.chat_history:
            with st.spinner("Summarizing..."):
                summary = summarize_conversation()
            st.info(summary)
        else:
            st.warning("No conversation to summarize yet.")
