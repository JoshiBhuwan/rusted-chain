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
result = agent.invoke("What's the weather like in Tokyo?")
print(result.text)
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

result = agent.invoke("What is 123 * 456?")
print(result.text)
```

## Supported Models

| Class | Provider | Env Variable |
|-------|----------|--------------|
| `GeminiModel` | Google Gemini | `GOOGLE_API_KEY` |
| `OpenAIModel` | OpenAI (GPT-4, etc.) | `OPENAI_API_KEY` |
| `ClaudeModel` | Anthropic Claude | `ANTHROPIC_API_KEY` |

## Advanced Usage

### Single-Shot vs Auto-Execution

`invoke()` behaves differently depending on whether tools are configured:

*   **With Tools**: It automatically runs the agent loop, executing tools until a final answer is reached. Returns an `AgentResponse` containing the final text.
*   **Without Tools**: It performs a single-shot completion. Returns an `AgentResponse` containing the text.

## Performance benchmark (test_perf.py)

A small benchmarking script is included at `test_perf.py` to compare the request/response latency of `rusted_chain` vs a LangChain-based client when calling the Google Gemini model (the repository author used `gemini-2.5-flash` for tests).

```bash
pip install langchain-google-genai python-dotenv
```

How to run

* Optional environment variables used by the script:
    * `RC_BENCH_WARMUP` — set to `true`/`false` (default: `true`) to perform warm-up calls before measurements.
    * `RC_BENCH_REPEATS` — number of timed repetitions (default: `4`).

Example usage (from repo root)

```bash
# set API key and run the benchmark
export GOOGLE_API_KEY="your-google-api-key"
python test_perf.py
```

## License

MIT
