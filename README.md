# mohit-tool-use-demo

A learning project demonstrating **agent loops with tool use** across multiple LLM APIs (Groq, Anthropic).

## What This Teaches

An **agent** is a loop where an LLM decides which tools to use, you execute them, and feed results back to the LLM. This project shows the complete flow:
1. Define tools with descriptions (get_weather, calculate, search_web)
2. Send user question + tools to LLM
3. LLM decides: "I need tool X"
4. You execute tool → get result
5. Feed result back in conversation history
6. Loop until LLM says "I have the answer"

The key insight: **messages list grows with each iteration**, so the LLM always sees the full context and can chain multiple tools.

## How to Run

```bash
pip install groq python-dotenv ddgs
export GROQ_API_KEY="your-key-here"
python main.py
```

## What Happens

- **Test 1**: Weather query → LLM calls get_weather → returns answer
- **Test 2**: Multi-step: "15% of engineer salary?" → search → calculate → answer

This pattern works the same across Anthropic Claude, Groq, OpenAI — only API details change.

## Interview Talking Point

When asked "How do agents work?": The agent loop - send query with tools → LLM decides tool → execute → feed result back → loop. The LLM sees full history, so context never breaks.
