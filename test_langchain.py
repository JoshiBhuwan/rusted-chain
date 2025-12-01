"""
Test LangChain tool integration.
"""
from langchain_core.tools import tool
from rusted_chain import GeminiModel
from dotenv import load_dotenv

load_dotenv()

# Define a LangChain tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

print("LangChain Tool Integration")
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

# Test invoke() with tools - auto-executes tools (was run())
print("\nTesting invoke() with tools - auto-execute tools...")
response = agent.invoke("Multiply 5 by 3")
print(f"Response type: {type(response)}")
if response.is_text:
    print(f"Result: {response.text}")
else:
    print(f"Unexpected response type: {response}")
