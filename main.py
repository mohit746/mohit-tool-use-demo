from groq import Groq
from ddgs import DDGS
from dotenv import load_dotenv
import json
from datetime import datetime
from collections import defaultdict

load_dotenv()

client = Groq()

def log_tool_call(user_question, tool_name, tool_inputs, result, status="success", error=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_question": user_question,
        "tool_name": tool_name,
        "tool_inputs": tool_inputs,
        "result": result,
        "status": status,
        "error": error
    }
    with open("agent_logs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

def print_log_summary():
    try:
        with open("agent_logs.jsonl", "r") as f:
            logs = [json.loads(line) for line in f]
        stats = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
        for log in logs:
            t = log["tool_name"]
            stats[t]["total"] += 1
            stats[t]["success" if log["status"] == "success" else "failure"] += 1
        print("\n📊 Tool call summary:")
        for name, s in stats.items():
            print(f"  {name}: {s['success']}/{s['total']} succeeded")
    except FileNotFoundError:
        print("No logs found yet.")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression e.g. '150000 * 0.15'"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"]
            }
        }
    }
]

def execute_tool(name, inputs):
    try:
        if name == "get_weather":
            return f"Weather in {inputs.get('city', 'Unknown')}: 28°C, sunny"

        elif name == "calculate":
            return str(eval(inputs.get("expression", "")))

        elif name == "search_web":
            results = DDGS().text(inputs.get("query", ""), max_results=3)
            if not results:
                return "No results found."
            return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

        else:
            return f"Unknown tool: '{name}'"

    except Exception as e:
        return f"{name} failed: {type(e).__name__}: {str(e)}"

# Persists across turns for the lifetime of the process
conversation_history = []

def chat(user_input):
    conversation_history.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=conversation_history,
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            # Append SDK object directly — it carries tool_calls metadata the API needs on the next turn
            conversation_history.append(msg)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = execute_tool(name, args)

                log_tool_call(user_input, name, args, result)
                print(f"  [{name}] {args} → {str(result)[:120]}")

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            answer = msg.content
            conversation_history.append({"role": "assistant", "content": answer})
            return answer

if __name__ == "__main__":
    print("Agent ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print_log_summary()
            break
        if not user_input:
            continue
        answer = chat(user_input)
        print(f"Agent: {answer}\n")
