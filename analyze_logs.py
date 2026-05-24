"""Analyze agent_logs.jsonl to understand agent behavior"""
import json
from collections import defaultdict

def analyze_agent_logs():
    """Parse and analyze tool call logs"""
    
    tools_used = defaultdict(int)
    tools_succeeded = defaultdict(int)
    tools_failed = defaultdict(int)
    
    print("\n" + "=" * 70)
    print("AGENT LOGS ANALYSIS")
    print("=" * 70)
    
    try:
        with open("agent_logs.jsonl", "r") as f:
            total_calls = 0
            for line in f:
                log_entry = json.loads(line)
                total_calls += 1
                
                tool_name = log_entry["tool_name"]
                status = log_entry["status"]
                timestamp = log_entry["timestamp"]
                
                tools_used[tool_name] += 1
                
                if status == "success":
                    tools_succeeded[tool_name] += 1
                else:
                    tools_failed[tool_name] += 1
                
                # Print each log entry
                print(f"\n[{timestamp}] Tool: {tool_name}")
                print(f"  Question: {log_entry['user_question']}")
                print(f"  Inputs: {log_entry['tool_inputs']}")
                print(f"  Status: {status}")
                if status == "failure":
                    print(f"  Error: {log_entry['error']}")
        
        # Summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Total tool calls: {total_calls}")
        print(f"Total unique tools: {len(tools_used)}")
        
        print("\nTool Success Rate:")
        for tool in sorted(tools_used.keys()):
            total = tools_used[tool]
            success = tools_succeeded[tool]
            failed = tools_failed[tool]
            rate = (success / total * 100) if total > 0 else 0
            print(f"  {tool}: {success}/{total} ({rate:.0f}%)")
            if failed > 0:
                print(f"    ❌ {failed} failures")
        
        print("\n" + "=" * 70)
        
    except FileNotFoundError:
        print("❌ agent_logs.jsonl not found. Run the agent first.")

if __name__ == "__main__":
    analyze_agent_logs()
