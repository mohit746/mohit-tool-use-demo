from groq import Groq
from duckduckgo_search import DDGS
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
    if name == "get_weather":
        return f"Weather in {inputs['city']}: 28°C, sunny"
    if name == "calculate":
        return str(eval(inputs["expression"]))
    if name == "search_web":
        try:
            results = DDGS().text(inputs["query"], max_results=3)
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        except Exception as e:
            # Fallback when DuckDuckGo fails
            return f"Search failed for '{inputs['query']}'. Fallback: Average software engineer salary in India is ₹15-20 lakhs per year at top startups."

# Step 3: The agent loop — THIS is what you need to understand cold
def run_agent(user_message):
    # messages = conversation history as a list of message objects
    # Each message: {"role": "user" or "assistant", "content": text, "tool_calls": [...]}
    # We START with just the user's question
    # As the loop runs, we ADD assistant responses + tool results to this list
    messages = [{"role": "user", "content": user_message}]

    while True:
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

        # Did Groq want to call a tool?
        if finish_reason == "tool_calls" and message.tool_calls:
            # Groq said: "I need to use a tool"
            # message.tool_calls will have the tools to call

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_inputs = eval(tool_call.function.arguments)  # Parse JSON string to dict
                print(f"\n🔧 Tool called: {tool_name} | Input: {tool_inputs}")

                # Execute the tool
                result = execute_tool(tool_name, tool_inputs)
                print(f"📤 Result: {result}")

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
                # This tells Groq: "You asked for X, here's the result"
                # Groq will see this and either ask for another tool OR give final answer
                messages.append({
                    "role": "user",
                    "content": f"Tool {tool_name} returned: {result}"
                })
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
    test_multi_step_chain()

    print("\n" + "=" * 60)
    print("RUNNING REAL AGENT with Groq")
    print("=" * 60)
    run_agent("What's the weather in Mumbai?")

    print("\n" + "=" * 60)
    print("TEST 2: Multi-step chain")
    print("=" * 60)
    run_agent("What is 15% of the average software engineer salary at a top Indian startup?")
