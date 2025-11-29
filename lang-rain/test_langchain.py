"""
Test LangChain tool integration.
"""
from langchain_core.tools import tool
from lang_rain import GeminiModel

# Define a LangChain tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

print("=== Test LangChain Tool Integration ===")
print(f"Tool Name: {multiply.name}")
print(f"Tool Description: {multiply.description}")

# Create agent with LangChain tool
agent = GeminiModel("gemini-2.5-flash", tools=[multiply])
print("\nAgent created with LangChain tool!")

# Test invoke() - single shot, returns AgentResponse
print("\nTesting invoke() - single shot...")
response = agent.invoke("Multiply 5 by 3")
print(f"Response: {response}")
print(f"Is tool call: {response.is_tool_call}")

# Test run() - auto-executes tools
print("\nTesting run() - auto-execute tools...")
result = agent.run("Multiply 5 by 3")
print(f"Result: {result}")
