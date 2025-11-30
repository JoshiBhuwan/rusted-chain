# Rusted Chain

Rusted Chain is a Rust-powered LLM agent framework for Python. It combines the performance of Rust with the usability of Python, providing a LangChain-compatible API for building AI agents.

## Features

*   **Rust Core**: Backend implemented in Rust using `pyo3`.
*   **Multi-Model Support**: Supports Gemini, OpenAI, and Claude.
*   **LangChain Compatible**: Accepts standard LangChain tools and `@tool` decorators.
*   **Auto-Execution Loop**: Handles tool calling and execution automatically.
*   **Simple API**: Straightforward interface for creating agents and tools.

## Installation

```bash
pip install rusted-chain
```

## Quick Start

### Basic Usage

```python
from rusted_chain import GeminiModel, OpenAIModel, ClaudeModel

# Initialize an agent
agent = GeminiModel(api_key="your-api-key")

# Simple text generation
response = agent.invoke("Tell me a joke about Rust.")
print(response.text)
```

### Using Tools

You can pass Python functions directly. `rusted-chain` handles schema generation and execution.

```python
from rusted_chain import GeminiModel

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 25°C."

# Create agent with tools
agent = GeminiModel(tools=[get_weather])

# Run the agent
result = agent.run("What's the weather like in Tokyo?")
print(result)
# Output: "The weather in Tokyo is sunny and 25°C."
```

### Using LangChain Tools

You can also use existing LangChain tools or the `@tool` decorator.

```python
from langchain_core.tools import tool
from rusted_chain import OpenAIModel

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

agent = OpenAIModel(tools=[multiply])

result = agent.run("What is 123 * 456?")
print(result)
```

## Supported Models

| Class | Provider | Env Variable |
|-------|----------|--------------|
| `GeminiModel` | Google Gemini | `GEMINI_API_KEY` |
| `OpenAIModel` | OpenAI (GPT-4, etc.) | `OPENAI_API_KEY` |
| `ClaudeModel` | Anthropic Claude | `ANTHROPIC_API_KEY` |

## Advanced Usage

### Manual Tool Control

Use `invoke()` to get the raw response (text or tool call) without auto-execution.

```python
response = agent.invoke("Call my_tool")

if response.is_tool_call:
    print(f"Tool: {response.tool_call.name}")
    print(f"Args: {response.tool_call.args}")
else:
    print(response.text)
```

## License

MIT
