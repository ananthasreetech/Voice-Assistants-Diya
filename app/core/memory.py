"""
Persistent memory for Diya.
Stores user name, preferences, relationships and past topics across sessions.
Memory is saved as JSON to MEMORY_FILE after every exchange.
"""

from __future__ import annotations

import datetime
import json
import logging

from app.utils.config import MEMORY_FILE, MEMORY_TOPIC_HISTORY

logger = logging.getLogger(__name__)

# ── Default schema ────────────────────────────────────────────────────────────

def _default() -> dict:
    return {
        "user_name":         None,
        "preferences":       {},
        "relationships":     {},
        "past_topics":       [],
        "conversation_count": 0,
        "last_seen":         None,
    }


# ── Load / save ───────────────────────────────────────────────────────────────

def load() -> dict:
    """Load memory from disk; return defaults if file missing or corrupt."""
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Backfill any keys added since the file was last written
        merged = _default()
        merged.update(data)
        return merged
    except FileNotFoundError:
        return _default()
    except Exception as exc:
        logger.warning("Could not load memory: %s", exc)
        return _default()


def save(mem: dict) -> None:
    """Persist memory to disk silently — never raise."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Could not save memory: %s", exc)


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(mem: dict) -> str:
    """Return a human-readable memory summary for injection into the system prompt."""
    lines: list[str] = []
    if mem.get("user_name"):
        lines.append(f"User's name: {mem['user_name']}")
    if mem.get("preferences"):
        lines.append(
            f"Known preferences: {json.dumps(mem['preferences'], ensure_ascii=False)}"
        )
    if mem.get("relationships"):
        lines.append(
            f"Known relationships: {json.dumps(mem['relationships'], ensure_ascii=False)}"
        )
    recent_topics = mem.get("past_topics", [])[-MEMORY_TOPIC_HISTORY:]
    if recent_topics:
        lines.append(f"Topics discussed recently: {', '.join(recent_topics)}")
    if mem.get("last_seen"):
        lines.append(f"Last conversation: {mem['last_seen']}")
    return "\n".join(lines)


# ── LLM-assisted extraction ───────────────────────────────────────────────────

def update_from_exchange(mem: dict, user_query: str, response: str, llm) -> dict:
    """
    Ask the LLM to extract any NEW personal information revealed in this
    exchange and merge it into the memory dict.  Returns the updated dict.
    """
    extract_prompt = (
        "Extract any NEW personal information from this conversation exchange "
        "to remember about the user.\n"
        f"Already known: {json.dumps(mem, ensure_ascii=False)}\n"
        f"User said: {user_query}\n"
        f"Assistant said: {response}\n\n"
        "Return ONLY a compact JSON object with keys from: "
        "user_name (string), preferences (dict), relationships (dict), "
        "past_topics (list of short strings).\n"
        "Include ONLY fields where genuinely new info was found. "
        "If nothing new, return exactly: {}"
    )
    try:
        result = llm.invoke(extract_prompt)
        raw    = result.content.strip().strip("```json").strip("```").strip()
        updates: dict = json.loads(raw)

        if updates.get("user_name"):
            mem["user_name"] = updates["user_name"]
        if updates.get("preferences"):
            mem.setdefault("preferences", {}).update(updates["preferences"])
        if updates.get("relationships"):
            mem.setdefault("relationships", {}).update(updates["relationships"])
        if updates.get("past_topics"):
            existing = set(mem.get("past_topics", []))
            for topic in updates["past_topics"]:
                if topic not in existing:
                    mem.setdefault("past_topics", []).append(topic)

        mem["conversation_count"] = mem.get("conversation_count", 0) + 1
        mem["last_seen"] = datetime.datetime.now().strftime("%B %d, %Y %H:%M")

        save(mem)
    except Exception as exc:
        logger.debug("Memory extraction skipped: %s", exc)

    return mem
