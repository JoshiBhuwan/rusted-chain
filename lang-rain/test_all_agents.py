"""
Test that all agents accept tools correctly.
"""
from lang_rain import GeminiModel, OpenAIModel, ClaudeModel, tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

print("=== Testing GeminiModel ===")
try:
    agent = GeminiModel(tools=[add, multiply])
    print("✅ GeminiModel created successfully with tools")
except Exception as e:
    print(f"❌ GeminiModel failed: {e}")

print("\n=== Testing OpenAIModel ===")
try:
    agent = OpenAIModel(tools=[add, multiply])
    print("✅ OpenAIModel created successfully with tools")
except Exception as e:
    print(f"❌ OpenAIModel failed: {e}")

print("\n=== Testing ClaudeModel ===")
try:
    agent = ClaudeModel(tools=[add, multiply])
    print("✅ ClaudeModel created successfully with tools")
except Exception as e:
    print(f"❌ ClaudeModel failed: {e}")
