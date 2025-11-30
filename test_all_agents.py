"""
Test that all agents accept tools correctly.
"""
from rusted_chain import GeminiModel, OpenAIModel, ClaudeModel, tool
# Here I am using the tool decorator like in LangChain
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
# Try creating all agents
try:
    agent = GeminiModel(tools=[add, multiply])
    print("GeminiModel created successfully with tools")
except Exception as e:
    print(f"GeminiModel failed: {e}")

try:
    agent = OpenAIModel(tools=[add, multiply])
    print("OpenAIModel created successfully with tools")
except Exception as e:
    print(f"OpenAIModel failed: {e}")

try:
    agent = ClaudeModel(tools=[add, multiply])
    print("ClaudeModel created successfully with tools")
except Exception as e:
    print(f"ClaudeModel failed: {e}")
