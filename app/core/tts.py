"""
Text-to-speech via Microsoft Edge-TTS.
Selects the appropriate Indian female voice for the detected language.
Runs the async Edge-TTS call safely from a synchronous Streamlit context.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import logging

import edge_tts

from app.utils.config import LANG_VOICES, DEFAULT_VOICE

logger = logging.getLogger(__name__)


# ── Async helper ──────────────────────────────────────────────────────────────

async def _synthesize(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def _run_async(coro) -> bytes:
    """Execute an async coroutine from synchronous code without event-loop conflicts."""
    def _target():
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_target).result()


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(text: str, lang_code: str = "en") -> bytes:
    """
    Convert *text* to MP3 audio bytes using Edge-TTS.

    Parameters
    ----------
    text : str
        Text to speak.
    lang_code : str
        ISO 639-1 code used to select the correct Indian female voice.

    Returns
    -------
    bytes
        Raw MP3 audio data ready to embed as a data-URI.
    """
    voice = LANG_VOICES.get(lang_code, DEFAULT_VOICE)
    logger.debug("TTS voice=%s for lang=%s", voice, lang_code)
    return _run_async(_synthesize(text, voice))


def audio_to_html(audio_bytes: bytes) -> str:
    """Return an autoplay <audio> HTML tag embedding the MP3 bytes as base64."""
    b64 = base64.b64encode(audio_bytes).decode()
    return (
        '<audio id="diya-audio" autoplay controls style="width:100%">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        "</audio>"
    )


STOP_AUDIO_JS = (
    "<script>"
    "var a=document.getElementById('diya-audio');"
    "if(a){a.pause();a.currentTime=0;}"
    "</script>"
)
