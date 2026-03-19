"""
API key resolution.
Priority: Streamlit Cloud secrets → .env file → already in session state.
Isolates all key-related logic from the main app.
"""

from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv


def _try_secret(name: str) -> str:
    """Read a single key from st.secrets; return '' on any failure."""
    try:
        value = st.secrets[name]
        return value.strip() if value else ""
    except Exception:
        return ""


def _try_env(name: str) -> str:
    """Read a single key from environment / .env file."""
    load_dotenv()
    return os.getenv(name, "").strip()


def resolve_keys() -> None:
    """
    Populate st.session_state.api_key and st.session_state.tavily_key
    from the highest-priority available source.
    Idempotent — already-set keys are never overwritten.
    """
    for state_key, env_name in [
        ("api_key",    "GROQ_API_KEY"),
        ("tavily_key", "TAVILY_API_KEY"),
    ]:
        if st.session_state.get(state_key):
            continue  # already resolved

        value = _try_secret(env_name) or _try_env(env_name)
        if value:
            st.session_state[state_key] = value


def keys_ready() -> bool:
    """Return True only when both required keys are present."""
    return bool(
        st.session_state.get("api_key") and st.session_state.get("tavily_key")
    )


def render_key_entry_screen() -> None:
    """
    Render the key-entry form on the main page (mobile-friendly — no sidebar).
    Calls st.stop() after rendering so the rest of the app is blocked.
    """
    st.markdown("### 🔑 Enter your API keys to meet Diya")
    st.markdown(
        "Free Groq key → [console.groq.com](https://console.groq.com) · "
        "Free Tavily key → [app.tavily.com](https://app.tavily.com)"
    )

    col1, col2 = st.columns(2)
    with col1:
        groq_input = st.text_input(
            "Groq API Key", type="password", placeholder="gsk_..."
        )
    with col2:
        tavily_input = st.text_input(
            "Tavily API Key", type="password", placeholder="tvly-..."
        )

    if st.button("✅ Save & Start", use_container_width=True):
        if groq_input.strip() and tavily_input.strip():
            st.session_state.api_key    = groq_input.strip()
            st.session_state.tavily_key = tavily_input.strip()
            st.rerun()
        else:
            st.error("Both keys are required.")

    st.stop()
