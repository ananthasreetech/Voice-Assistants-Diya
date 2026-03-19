"""
Centralised configuration for Diya.
All tuneable constants live here — no magic strings scattered across the app.
"""

from __future__ import annotations

# ── Identity ─────────────────────────────────────────────────────────────────
ASSISTANT_NAME = "Diya"
ASSISTANT_ICON = "🪔"

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL          = "llama-3.3-70b-versatile"
MEMORY_EXTRACT_MODEL = "llama-3.3-70b-versatile"

# ── Transcription ─────────────────────────────────────────────────────────────
WHISPER_MODEL = "whisper-large-v3-turbo"   # faster + cheaper than large-v3

# ── Language / TTS ────────────────────────────────────────────────────────────
# Edge-TTS voice IDs — Indian female, one per supported language
LANG_VOICES: dict[str, str] = {
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "en": "en-IN-NeerjaNeural",
}
DEFAULT_LANG  = "en"
DEFAULT_VOICE = LANG_VOICES[DEFAULT_LANG]

# Whisper returns full language names; map to ISO 639-1 codes
WHISPER_LANG_TO_CODE: dict[str, str] = {
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "english": "en",
}
LANG_CODE_TO_NAME: dict[str, str] = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "en": "English",
}

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_FILE          = "/tmp/diya_memory.json"
MEMORY_TOPIC_HISTORY = 8      # how many past topics to include in context

# ── Audio recorder ────────────────────────────────────────────────────────────
RECORDER_PAUSE_THRESHOLD = 2.0   # seconds of silence before auto-stop

# ── Web search ────────────────────────────────────────────────────────────────
SEARCH_MAX_RESULTS = 3
SEARCH_TRIGGER_KEYWORDS: list[str] = [
    "latest", "recent", "today", "news", "current", "now", "price",
    "weather", "score", "who is", "when did", "how much", "trending",
    "update", "released", "just", "live",
]

# ── UI ────────────────────────────────────────────────────────────────────────
STATE_UI: dict[str, tuple[str, str]] = {
    "ready":    ("🟢", "Ready — tap the mic to speak"),
    "thinking": ("🟡", "Thinking..."),
    "speaking": ("🔵", "Diya is speaking — tap Stop to interrupt"),
}
