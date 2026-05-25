# mohit-tool-use-demo

An autonomous agent that chains multiple tools to answer complex queries — built with Groq/Llama and Google Gemini, with multi-turn conversation memory and structured JSON logging.

Ask it *"What is 15% of Stripe's CEO's annual salary?"* and it will search the web, extract the number, calculate, and answer — automatically, in one shot.

## Architecture

```
User Query → LLM (Groq/Llama) → Tool Router → [search_web | calculate | get_weather]
                ↑                                          ↓
        Conversation History ←←←←←←←←←←← Tool Result + JSONL Log
```

## Stack

| | |
|---|---|
| LLM (primary) | Groq — llama-3.3-70b-versatile |
| LLM (secondary) | Google Gemini 2.5 Flash |
| Web search | DuckDuckGo (ddgs) |
| Logging | JSONL — one structured event per tool call |
| Language | Python 3.10+ |

## How to Run

```bash
git clone https://github.com/mohit746/mohit-tool-use-demo.git && cd mohit-tool-use-demo
pip install -r requirements.txt
GROQ_API_KEY=your-key python main.py
```

For Gemini: `GEMINI_API_KEY=your-key python gemini_agent.py`

## How It Works

Two memory layers run simultaneously:

**Intra-turn loop** — within one question, the agent keeps calling tools until it has enough to answer. LLM calls `search_web` → result appended to messages → LLM calls `calculate` → result appended → LLM returns final answer.

**Inter-turn memory** — `conversation_history` lives outside the function and is passed on every API call. Turn 2 can reference Turn 1 because the model sees the full prior exchange.

```
Turn 1: "Search for Stripe"     → history grows to 4 messages
Turn 2: "What did you find?"    → model sees all 4, answers without re-searching
```

## Key Learning: Groq System Prompt Quirk

Groq/Llama breaks tool call formatting when the system prompt contains ordering or prescriptive instructions. The model generates malformed `<function=tool_name=args>` syntax that the API rejects with a 400 error.

Fix: keep the system prompt minimal — `"You are a helpful assistant. Use tools to answer questions accurately."` — and put any task-specific framing in the user message instead.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Groq agent — multi-turn CLI chat with tool use and JSONL logging |
| `gemini_agent.py` | Same pattern on Google Gemini API (different message format) |
| `analyze_logs.py` | Parse `agent_logs.jsonl` for per-tool success/failure stats |
