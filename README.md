# mohit-tool-use-demo

A learning project demonstrating **agent loops with tool use** across multiple LLM APIs (Groq, Anthropic).

## What This Teaches

### Stage 1 — Single-turn agent loop
An **agent** is a loop where an LLM decides which tools to use, you execute them, and feed results back to the LLM:
1. Define tools with descriptions (get_weather, calculate, search_web)
2. Send user question + tools to LLM
3. LLM decides: "I need tool X"
4. You execute tool → get result
5. Feed result back in conversation history
6. Loop until LLM says "I have the answer"

The key insight: **messages list grows with each iteration**, so the LLM always sees the full context and can chain multiple tools.

### Stage 2 — Multi-turn conversation memory
A single `conversation_history` list lives outside the agent function and persists across turns within a session:

- **Intra-turn memory**: the tool call loop inside one question (search → calculate → answer)
- **Inter-turn memory**: `conversation_history` passed to every API call so the model can reference what was said earlier

The model answers "What did you find?" correctly because the full prior exchange is included in every request. This memory is **in-process only** — it resets when the Python process exits.

## How to Run

```bash
pip install groq python-dotenv ddgs
export GROQ_API_KEY="your-key-here"
python main.py
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Groq agent with tool use + conversation history |
| `gemini_agent.py` | Same pattern implemented with Google Gemini API |

## Architecture

```
User input
    ↓
conversation_history.append(user message)
    ↓
┌─────────────────────────────────┐
│  while True (intra-turn loop)   │
│    API call with full history   │
│    ↓                            │
│    tool_calls? → execute tool   │
│               → append result   │
│               → loop again      │
│    no tool_calls? → return ans  │
└─────────────────────────────────┘
    ↓
conversation_history.append(answer)
    ↓
Next user input sees full history
```

## Interview Talking Point

When asked "How do agents work?": The agent loop — send query with tools → LLM decides tool → execute → feed result back → loop. The LLM sees full history so context never breaks. For multi-turn chat, a persistent `conversation_history` list is passed on every API call — the model has no memory of its own, it only knows what you send it.
