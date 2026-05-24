from google import genai
from google.genai import types
from ddgs import DDGS
from dotenv import load_dotenv
import json
from datetime import datetime
from collections import defaultdict
import os

load_dotenv()

# Initialize Gemini API client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
conversation_history = []  # To keep track of the conversation history for multi-turn interactions

# Step 0: Logging utility for tool calls
def log_tool_call(user_question, tool_name, tool_inputs, result, status="success", error=None):
    """Log every tool call to gemini_logs.jsonl for debugging and analysis"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_question": user_question,
        "tool_name": tool_name,
        "tool_inputs": tool_inputs,
        "result": result,
        "status": status,  # "success" or "failure"
        "error": error     # Error details if failed
    }

    # Append to JSONL file (one JSON per line)
    with open("gemini_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def print_log_summary():
    """Utility to print a summary of tool calls from the log file"""
    try:
        with open("gemini_logs.jsonl", "r") as f:
            logs = [json.loads(line) for line in f]

        tool_stats = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
        for log in logs:
            tool_name = log['tool_name']
            tool_stats[tool_name]["total"] += 1
            if log['status'] == "success":
                tool_stats[tool_name]["success"] += 1
            else:
                tool_stats[tool_name]["failure"] += 1

        print("\n📊 TOOL CALL SUMMARY:")
        for tool_name in tool_stats:
            print(f"  - {tool_name} : {tool_stats[tool_name]['total']} calls, {tool_stats[tool_name]['success']} successes")

    except FileNotFoundError:
        print("No logs found yet.")

# Step 1: Define your tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression e.g. '150000 * 0.15'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

# Step 2: Real tool executors
def execute_tool(name, inputs):
    try:
        if name == "calculate":
            try:
                expression = inputs.get('expression', '')
                result = eval(expression)
                return str(result)
            except Exception as e:
                error_msg = f"calculate failed: Invalid expression '{inputs.get('expression', '')}' - {type(e).__name__}: {str(e)}"
                print(f"    ⚠️  {error_msg}")
                return error_msg
        
        elif name == "search_web":
            try:
                query = inputs.get('query', '')
                results = DDGS().text(query, max_results=3)
                if not results:
                    return f"search_web: No results found for '{query}'"
                return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            except Exception as e:
                error_msg = f"search_web failed: DuckDuckGo error - {type(e).__name__}: {str(e)}"
                print(f"    ⚠️  {error_msg}")
                return error_msg

        elif name == "get_weather":
            try:
                city = inputs.get('city', 'Unknown')
                return f"Weather in {city}: 28°C, sunny"
            except Exception as e:
                error_msg = f"get_weather failed: {type(e).__name__}: {str(e)}"
                print(f"    ⚠️  {error_msg}")
                return error_msg

        else:
            # Unknown tool
            error_msg = f"Unknown tool: '{name}'. Available tools: get_weather, calculate, search_web"
            print(f"    ❌ {error_msg}")
            return error_msg

    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Unexpected error in execute_tool: {type(e).__name__}: {str(e)}"
        print(f"    ❌ {error_msg}")
        return error_msg

# Step 3: The agent loop
def run_agent(user_message):
    retry_count = 0
    system_prompt = "You are a helpful assistant. Use tools to answer questions accurately."
    original_user_question = user_message
    conversation_history.append({"role": "user", "content": user_message})

    # Build config once — tools live inside config, not as a top-level arg
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[types.Tool(function_declarations=tools)]
    )

    # Gemini uses types.Content objects for multi-turn history
    messages = [
        types.Content(role="user", parts=[types.Part(text=user_message)])
    ]

    while True:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
                config=config
            )

            # Check for tool calls
            has_tool_calls = False
            model_parts = response.candidates[0].content.parts if response.candidates else []

            for part in model_parts:
                if part.function_call:
                    has_tool_calls = True
                    tool_name = part.function_call.name
                    tool_inputs = dict(part.function_call.args)

                    print(f"\n🔧 Tool called: {tool_name} | Input: {tool_inputs}")

                    # Execute the tool
                    execution_status = "success"
                    execution_error = None
                    try:
                        result = execute_tool(tool_name, tool_inputs)
                        print(f"📤 Result: {result}")
                    except Exception as exec_error:
                        execution_status = "failure"
                        execution_error = str(exec_error)
                        result = f"Error executing {tool_name}: {str(exec_error)}"
                        print(f"❌ Tool Execution Error: {result}")

                    # Log the tool call
                    log_tool_call(
                        user_question=original_user_question,
                        tool_name=tool_name,
                        tool_inputs=tool_inputs,
                        result=result,
                        status=execution_status,
                        error=execution_error
                    )

                    # Add model's tool-call turn and the tool result to history
                    messages.append(
                        types.Content(role="model", parts=[part])
                    )
                    messages.append(
                        types.Content(
                            role="user",
                            parts=[types.Part(
                                function_response=types.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result}
                                )
                            )]
                        )
                    )
                conversation_history.append({"role": "tool", "content": result})
            if not has_tool_calls:
                text_response = "".join(p.text for p in model_parts if hasattr(p, "text"))
                if text_response:
                    print(f"\n✅ Final answer: {text_response}")
                    conversation_history.append({"role": "assistant", "content": text_response})
                break

        except Exception as e:
            retry_count += 1
            if "429" in str(e):
                if retry_count > 3:
                    print("❌ Max retries exceeded.")
                    break
                else:
                    wait = min(2 * retry_count, 60)
                    print(f"Rate limit hit. Waiting {wait}s before retrying...")
                    import time; time.sleep(wait)
                    continue
            else:
                print(f"\n❌ API Error: {type(e).__name__}: {str(e)}")
                break

# Test function to visualize the agent flow WITHOUT needing the API
def test_multi_step_chain():
    """Visualize how messages grow during a multi-step chain"""
    print("\n" + "=" * 70)
    print("SIMULATING: 'What is 15% of average engineer salary?'")
    print("=" * 70)

    # STEP 1: User asks question
    messages = [{"role": "user", "content": "What is 15% of average engineer salary at a top Indian startup?"}]
    print("\n📋 STEP 1 - User asks:")
    print(f"  messages = {messages}")

    # STEP 2: Gemini says "I need to search"
    print("\n🤖 STEP 2 - Gemini's response (tool_use):")
    print("  Gemini says: 'I need to search for this data'")
    messages.append({
        "role": "assistant",
        "content": "search_web tool will be called with query='average software engineer salary India'"
    })
    print(f"  messages now has {len(messages)} items")

    # STEP 3: We execute search, get result
    print("\n⚙️  STEP 3 - We execute search_web:")
    search_result = "Average salary at top Indian startups: ₹15-20 lakhs per year"
    print(f"  Result: {search_result}")
    messages.append({
        "role": "user",
        "content": f"Tool result: {search_result}"
    })
    print(f"  messages now has {len(messages)} items")

    # STEP 4: Gemini sees result, says "Now I'll calculate"
    print("\n🤖 STEP 4 - Gemini's response (tool_use again):")
    print("  Gemini says: 'Now I need to calculate 15% of 18 lakhs'")
    messages.append({
        "role": "assistant",
        "content": "calculate tool will be called with expression='1800000 * 0.15'"
    })
    print(f"  messages now has {len(messages)} items")

    # STEP 5: We execute calculate
    print("\n⚙️  STEP 5 - We execute calculate:")
    calc_result = str(1800000 * 0.15)
    print(f"  Result: ₹{calc_result}")
    messages.append({
        "role": "user",
        "content": f"Tool result: {calc_result}"
    })
    print(f"  messages now has {len(messages)} items")

    # STEP 6: Gemini has final answer
    print("\n🤖 STEP 6 - Gemini's response (end_turn):")
    final_answer = "15% of the average software engineer salary at a top Indian startup (₹18 lakhs) is ₹2.7 lakhs per year."
    print(f"  ✅ Final answer: {final_answer}")
    messages.append({
        "role": "assistant",
        "content": final_answer
    })
    print(f"  messages now has {len(messages)} items")

    print("\n" + "=" * 70)
    print("📊 FINAL CONVERSATION HISTORY:")
    print("=" * 70)
    for i, msg in enumerate(messages, 1):
        role = "USER" if msg["role"] == "user" else "GEMINI"
        content = msg.get('content', '')
        print(f"\n{i}. [{role}]")
        print(f"   {content[:80]}..." if len(content) > 80 else f"   {content}")


# Run the visualization test + real agent
if __name__ == "__main__":
    # test_multi_step_chain()

    # print("\n" + "=" * 90)
    # print("Test 1: RUNNING REAL AGENT with Gemini")
    # print("=" * 90)
    # run_agent("What's the weather in Udaipur, Rajasthan?")
    
    print("Agent ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("What's on your mind? ").strip()
        if user_input.lower() in ["quit", "q", "exit", "bye"]:
            print("Goodbye!")
            break
        elif not user_input:
            print("Please enter a message (or type 'quit' to exit)")
            continue
        else:
            run_agent(user_input)

    # print("\n" + "=" * 90)
    # print("🤖 INFINITE CHAT MODE - Type 'quit' to exit, 'help' for commands")
    # print("=" * 90)
    
    # conversation_count = 0
    # while True:
    #     try:
    #         # Get user input with validation
    #         user_input = input("\n💬 You: ").strip()

    #         # Handle special commands
    #         if not user_input:
    #             print("  ⚠️  Please enter a message (or type 'quit' to exit)")
    #             continue

    #         if user_input.lower() in ["quit", "exit", "bye"]:
    #             print("\n👋 Goodbye! Here's your final session summary:")
    #             print_log_summary()
    #             break

    #         if user_input.lower() == "help":
    #             print("\n📋 Available commands:")
    #             print("  • Type any question to chat with the agent")
    #             print("  • 'stats' - Show tool call statistics")
    #             print("  • 'quit' or 'exit' - End the chat session")
    #             print("  • 'help' - Show this help message")
    #             continue

    #         if user_input.lower() == "stats":
    #             print_log_summary()
    #             continue

    #         # Run the agent
    #         conversation_count += 1
    #         print(f"\n🔄 Running conversation #{conversation_count}...")
    #         run_agent(user_input)

    #     except KeyboardInterrupt:
    #         # Handle Ctrl+C gracefully
    #         print("\n\n⏸️  Session interrupted. Final statistics:")
    #         print_log_summary()
    #         break
    #     except Exception as e:
    #         print(f"\n❌ Unexpected error: {type(e).__name__}: {str(e)}")
    #         print("  Type 'quit' to exit or try another question")
