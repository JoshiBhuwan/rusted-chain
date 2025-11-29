"""
lang_rain - LangChain-style LLM framework in Rust
"""

# Import Rust bindings  
import lang_rain.lang_rain as _rust
# Import Python helpers
from .tool_helpers import tool, ToolAdapter, ensure_tool_wrapper

# Re-export Rust classes
create_agent = _rust.create_agent
AgentResponse = _rust.AgentResponse
ToolCall = _rust.ToolCall


class AgentWrapper:
    """Base wrapper for Rust agents to handle tool wrapping."""
    
    def __init__(self, rust_class, model=None, tools=None, middleware=None, context_schema=None, api_key=None):
        if tools:
            tools = [ensure_tool_wrapper(t) for t in tools]
        self._agent = rust_class(model, tools, middleware, context_schema, api_key)

    def invoke(self, query: str) -> AgentResponse:
        """
        Invoke the model and return the response.
        Like LangChain's invoke() - returns AgentResponse (text or tool_call).
        """
        return self._agent.invoke(query)
    
    def run(self, query: str) -> str:
        """
        Run the model with automatic tool execution.
        Like LangChain's AgentExecutor - loops until final text response.
        """
        return self._agent.run(query)
        
    def add_tool(self, tool):
        """Add a tool to the model."""
        return self._agent.add_tool(ensure_tool_wrapper(tool))

    def __getattr__(self, name):
        return getattr(self._agent, name)


class GeminiModel(AgentWrapper):
    """Gemini model wrapper."""
    def __init__(self, model=None, tools=None, middleware=None, context_schema=None, api_key=None):
        super().__init__(_rust.GeminiModel, model, tools, middleware, context_schema, api_key)


class OpenAIModel(AgentWrapper):
    """OpenAI model wrapper."""
    def __init__(self, model=None, tools=None, middleware=None, context_schema=None, api_key=None):
        super().__init__(_rust.OpenAIModel, model, tools, middleware, context_schema, api_key)


class ClaudeModel(AgentWrapper):
    """Claude model wrapper."""
    def __init__(self, model=None, tools=None, middleware=None, context_schema=None, api_key=None):
        super().__init__(_rust.ClaudeModel, model, tools, middleware, context_schema, api_key)


__all__ = ['GeminiModel', 'OpenAIModel', 'ClaudeModel', 'create_agent', 'AgentResponse', 'ToolCall', 
           'tool', 'ToolAdapter']
