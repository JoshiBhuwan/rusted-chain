"""
Test the simplified LangChain-style API - pass functions directly!
"""

from lang_rain import GeminiModel, tool

# Option 1: Just pass the function directly
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

print("=== Test 1: Pass functions directly ===")
agent = GeminiModel("gemini-2.5-flash", tools=[get_word_length, add_numbers])
print("Agent created with tools!")
print()

# Option 2: Use the @tool decorator for more control
@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

print("=== Test 2: Using @tool decorator ===")
agent2 = GeminiModel("gemini-2.5-flash", tools=[multiply])
print("Agent created with decorated tool!")
print()
print("=== Test 3: invoke() - single shot ===")
response = agent.invoke("What is 2 + 2?")
print(f"Response type: {'text' if response.is_text else 'tool_call'}")
print()

result = agent.run("How many letters in 'hello'?")
print(f"Result: {result}")
