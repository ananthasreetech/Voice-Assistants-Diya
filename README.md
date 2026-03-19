# 🪔 Diya — Indian Voice Assistant

Diya is an English-language Indian female voice assistant built with Streamlit, Groq Whisper, LangChain, Tavily web search, and Microsoft Edge-TTS. She remembers your name and past conversations across sessions.

---

## Features

| Feature | Details |
|---|---|
| Voice name | Diya — introduces herself, remembers your name permanently |
| Indian female voice | Edge-TTS `en-IN-NeerjaNeural` — natural Indian English |
| Barge-in | Stop Diya mid-speech and speak immediately |
| Turn-taking | Visual state: 🟢 Ready → 🟡 Thinking → 🔵 Speaking |
| Low latency | `whisper-large-v3-turbo` for fast transcription |
| Persistent memory | Remembers name, preferences, relationships, past topics across sessions |
| Web search | Tavily — triggered automatically for factual and location queries |

---

## Project structure

```
diya/
├── main.py              # Full application (single file, easy to deploy)
├── requirements.txt
├── .env.example         # Copy to .env for local development
├── .gitignore
└── .streamlit/
    └── config.toml      # Streamlit server settings
```

---

## Quickstart (local)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/diya.git
cd diya

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Edit .env — fill in GROQ_API_KEY and TAVILY_API_KEY

# 5. Run
streamlit run main.py
```

---

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo → main file: `main.py`.
3. In the deployed app: **⋮ → Settings → Secrets**, add:

```toml
GROQ_API_KEY   = "gsk_..."
TAVILY_API_KEY = "tvly-..."
```

4. Save — app restarts and is ready on any browser including mobile Chrome.

> **Mobile note:** Chrome on Android requires HTTPS for microphone access. Streamlit Cloud serves over HTTPS automatically.

---

## API keys (both free)

| Service | Purpose | Get key |
|---|---|---|
| [Groq](https://console.groq.com) | LLM + Whisper transcription | console.groq.com |
| [Tavily](https://app.tavily.com) | Web search (1000 free/month) | app.tavily.com |

---

## Memory

Diya saves memory to `diya_memory.json` in the project folder. This file is excluded from git via `.gitignore`. On Streamlit Cloud it persists until redeployment. To reset memory, delete the file.

---

## Customisation

All constants are at the top of `main.py`:

- `LLM_MODEL` — swap to any Groq-supported model
- `WHISPER_MODEL` — change transcription speed/accuracy
- `TTS_VOICE` — change to any Edge-TTS voice
- `SEARCH_KEYWORDS` — add triggers for web search
- The `pause_threshold` in `audio_recorder()` controls silence detection

---

## License

MIT
