from groq import Groq
from ddgs import DDGS  # Updated: use ddgs (duckduckgo_search is deprecated)
from dotenv import load_dotenv

load_dotenv()

client = Groq()  # uses GROQ_API_KEY from .env

# Step 1: Define your tools (Groq format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
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
                "properties": {
                    "expression": {"type": "string", "description": "Math expression e.g. '150000 * 0.15'"}
                },
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
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

# Step 2: Real tool executors
def execute_tool(name, inputs):
    try:
        if name == "get_weather":
            try:
                city = inputs.get('city', 'Unknown')
                return f"Weather in {city}: 28°C, sunny"
            except Exception as e:
                error_msg = f"get_weather failed: {type(e).__name__}: {str(e)}"
                print(f"    ⚠️  {error_msg}")
                return error_msg

        elif name == "calculate":
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

# Step 3: The agent loop — THIS is what you need to understand cold
def run_agent(user_message):
    # IMPORTANT: System prompt MUST be first message in Groq API
    # This tells Groq HOW to think before it sees the user's question
    # Keep system prompts simple - Groq can be sensitive to complex prompts with tools
    # system_prompt = "You are a helpful assistant. Use tools to answer questions accurately."
    system_prompt = "You are a helpful assistant. Use tools to answer questions accurately."

    # messages = conversation history as a list of message objects
    # ORDER MATTERS:
    #   1. System prompt (how to behave)
    #   2. User question
    #   3. Assistant responses + tool results
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    while True:
        try:
            # CLIENT.CHAT.COMPLETIONS.CREATE() - Call Groq API
            # Think of this as: "Groq, here's the conversation so far. What should we do next?"
            response = client.chat.completions.create(
                # model: Which model to use
                #   "llama-3.3-70b-versatile" = Groq's latest model
                model="llama-3.3-70b-versatile",

                # max_tokens: Max length of response
                #   1024 tokens ≈ 4000 characters
                #   Limits cost and prevents very long outputs
                max_tokens=1024,

                # tools: Array of tool definitions Groq can choose from
                #   Groq READS this list and decides: "Do I need a tool for this?"
                #   If yes → returns tool_calls with tool name + inputs
                #   If no → returns just text answer
                tools=tools,

                # messages: The ENTIRE conversation history
                #   Groq sees this full history to maintain context
                #   Starts: [{"role": "user", "content": "What's weather in Mumbai?"}]
                #   After 1st loop: adds assistant response + tool result
                #   After 2nd loop: sees search result, can now answer
                #   This is how Groq "remembers" what happened before
                messages=messages,

            )

            # RESPONSE object contains:
            #   response.choices[0].message.tool_calls = list of tool calls (if any)
            #   response.choices[0].message.content = text response
            #   response.choices[0].finish_reason = "tool_calls" | "stop"

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

        except Exception as e:
            print(f"\n❌ API Error: {str(e)}")
            print("Retrying with simpler request...")
            continue

        # Did Groq want to call a tool?
        if finish_reason == "tool_calls" and message.tool_calls:
            # Groq said: "I need to use a tool"
            # message.tool_calls will have the tools to call

            for tool_call in message.tool_calls:
                try:
                    tool_name = tool_call.function.name

                    # Parse JSON arguments - wrap in try/except
                    try:
                        tool_inputs = eval(tool_call.function.arguments)  # Parse JSON string to dict
                    except Exception as parse_error:
                        print(f"❌ JSON Parse Error: {parse_error}")
                        tool_inputs = {}

                    print(f"\n🔧 Tool called: {tool_name} | Input: {tool_inputs}")

                    # Execute the tool - wrap in try/except
                    try:
                        result = execute_tool(tool_name, tool_inputs)
                        print(f"📤 Result: {result}")
                    except Exception as exec_error:
                        result = f"Error executing {tool_name}: {str(exec_error)}"
                        print(f"❌ Tool Execution Error: {result}")

                    # THIS IS THE KEY LOOP PART:
                    # Add Groq's response (with tool call) to messages
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    })

                    # Add the tool RESULT back to messages
                    # Groq format: tool results go in a simple text message that references the tool call
                    # The LLM can see from the content what tool was called and what the result was
                    # Groq will see this and either ask for another tool OR give final answer
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })

                except Exception as tool_error:
                    print(f"❌ Error processing tool call: {str(tool_error)}")
                    continue
            # Loop continues → calls chat.completions.create() AGAIN with full history

        else:
            # Groq said: "I have my final answer"
            # finish_reason == "stop"
            # message.content has the actual answer text
            print(f"\n✅ Final answer: {message.content}")
            break  # Exit the loop

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
    
    # STEP 2: Claude says "I need to search"
    print("\n🤖 STEP 2 - Claude's response (stop_reason='tool_use'):")
    print("  Claude says: 'I need to search for this data'")
    # Simulate Claude's tool_use response
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
    
    # STEP 4: Claude sees result, says "Now I'll calculate"
    print("\n🤖 STEP 4 - Claude's response (stop_reason='tool_use' again):")
    print("  Claude says: 'Now I need to calculate 15% of 18 lakhs'")
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
    
    # STEP 6: Claude has final answer
    print("\n🤖 STEP 6 - Claude's response (stop_reason='end_turn'):")
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
        role = "USER" if msg["role"] == "user" else "CLAUDE"
        print(f"\n{i}. [{role}]")
        print(f"   {msg['content'][:80]}..." if len(msg['content']) > 80 else f"   {msg['content']}")


# Run the visualization test + real agent
if __name__ == "__main__":
    # test_multi_step_chain()

    print("\n" + "=" * 90)
    print("Test 1: RUNNING REAL AGENT with Groq")
    print("=" * 90)
    # run_agent("What's the weather?")
    run_agent("What's the weather in Udaipur, Rajasthan?")
    
    print("\n" + "=" * 90)
    print("TEST 2: Multi-step chain")
    print("=" * 90)
    run_agent("What is 15% of the average software engineer salary at a top Indian startup?")
