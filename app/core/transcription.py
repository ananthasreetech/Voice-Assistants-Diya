"""
Audio transcription via Groq Whisper.
Returns both the transcript text and the auto-detected language code.
"""

from __future__ import annotations

import os
import tempfile
import logging

import groq as groq_sdk

from app.utils.config import WHISPER_MODEL, WHISPER_LANG_TO_CODE, DEFAULT_LANG, LANG_VOICES

logger = logging.getLogger(__name__)


def transcribe(audio_bytes: bytes, api_key: str) -> tuple[str, str]:
    """
    Transcribe raw audio bytes using Groq Whisper.

    Returns
    -------
    text : str
        The transcribed text.
    lang_code : str
        ISO 639-1 language code detected by Whisper (e.g. 'hi', 'en').
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
                response_format="verbose_json",   # gives us language field
            )

        raw_lang  = (getattr(result, "language", "") or "").lower().strip()
        lang_code = WHISPER_LANG_TO_CODE.get(raw_lang, raw_lang[:2] if raw_lang else DEFAULT_LANG)

        # Fall back to English if we got a code we have no voice for
        if lang_code not in LANG_VOICES:
            lang_code = DEFAULT_LANG

        text = (result.text or "").strip()
        logger.debug("Transcribed (%s): %s", lang_code, text[:80])
        return text, lang_code

    finally:
        os.unlink(tmp_path)
