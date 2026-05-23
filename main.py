import anthropic
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from .env

# Step 1: Define your tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
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
        "input_schema": {
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
        "input_schema": {
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
    if name == "get_weather":
        return f"Weather in {inputs['city']}: 28°C, sunny"
    if name == "calculate":
        return str(eval(inputs["expression"]))
    if name == "search_web":
        results = DDGS().text(inputs["query"], max_results=3)
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])

# Step 3: The agent loop — THIS is what you need to understand cold
def run_agent(user_message):
    # messages = conversation history as a list of message objects
    # Each message: {"role": "user" or "assistant", "content": text or list of blocks}
    # We START with just the user's question
    # As the loop runs, we ADD assistant responses + tool results to this list
    messages = [{"role": "user", "content": user_message}]

    while True:
        # CLIENT.MESSAGES.CREATE() - Call Claude API
        # Think of this as: "Claude, here's the conversation so far. What should we do next?"
        response = client.messages.create(
            # model: Which Claude to use
            #   "claude-opus-4-7" = latest, smartest model
            #   "claude-sonnet-4-6" = faster, cheaper
            #   Choose based on speed vs quality tradeoff
            model="claude-sonnet-4-6",

            # max_tokens: Max length of Claude's response
            #   1024 tokens ≈ 4000 characters
            #   If response might be long, increase this
            #   Limits cost and prevents very long outputs
            max_tokens=1024,

            # tools: Array of tool definitions Claude can choose from
            #   Claude READS this list and decides: "Do I need a tool for this?"
            #   If yes → returns stop_reason="tool_use" with tool name + inputs
            #   If no → returns stop_reason="end_turn" with text answer
            tools=tools,

            # messages: The ENTIRE conversation history
            #   Claude sees this full history to maintain context
            #   Starts: [{"role": "user", "content": "What's weather in Mumbai?"}]
            #   After 1st loop: adds assistant response + tool result
            #   After 2nd loop: sees search result, can now answer
            #   This is how Claude "remembers" what happened before
            messages=messages,

            # system: INSTRUCTION to guide Claude's thinking
            #   Tells Claude HOW to approach the problem
            #   "Always search first", "Think step-by-step", etc.
            #   This shapes Claude's decision about which tools to use
            system="When asked about calculations involving data you don't have: 1) Search for the data using search_web 2) Use calculate tool with the data 3) Provide the answer"
        )
        
        # RESPONSE object contains:
        #   response.stop_reason = "tool_use" | "end_turn" (why Claude stopped)
        #   response.content = list of blocks (TextBlock, ToolUseBlock, etc.)

        # Did Claude want to call a tool?
        if response.stop_reason == "tool_use":
            # Claude said: "I need to use a tool"
            # response.content will have a ToolUseBlock with tool name + inputs

            # Extract the ToolUseBlock from response.content list
            # (response.content might have multiple blocks, we find the tool_use one)
            tool_use = next(b for b in response.content if b.type == "tool_use")
            print(f"\n🔧 Tool called: {tool_use.name} | Input: {tool_use.input}")

            # Execute the tool (call our fake function)
            result = execute_tool(tool_use.name, tool_use.input)
            print(f"📤 Result: {result}")

            # THIS IS THE KEY LOOP PART:
            # Add Claude's response (with tool call) to messages
            messages.append({"role": "assistant", "content": response.content})

            # Add the tool RESULT back to messages
            # This tells Claude: "You asked for weather in Mumbai, here it is: 28°C"
            # Claude will see this and either ask for another tool OR give final answer
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": result}]
            })
            # Loop continues → calls messages.create() AGAIN with full history

        else:
            # Claude said: "I have my final answer"
            # response.stop_reason == "end_turn"
            # response.content[0] is a TextBlock with the actual answer text
            print(f"\n✅ Final answer: {response.content[0].text}")
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


# Run the visualization test
if __name__ == "__main__":
    test_multi_step_chain()

    # Uncomment below when you have valid ANTHROPIC_API_KEY:
    # print("\n" + "=" * 60)
    # print("RUNNING REAL AGENT (requires API key)")
    # print("=" * 60)
    # run_agent("What's the weather in Mumbai?")
    # run_agent("What is 15% of the average software engineer salary at a top Indian startup?")
