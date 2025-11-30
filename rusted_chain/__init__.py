"""
rusted_chain - LangChain-style LLM framework in Rust
"""

import rusted_chain.rusted_chain as _rust
from .tool_helpers import tool, ToolAdapter, ensure_tool_wrapper
create_agent = _rust.create_agent
AgentResponse = _rust.AgentResponse
ToolCall = _rust.ToolCall


class AgentWrapper:
    def __init__(self, rust_class, model=None, tools=None, api_key=None):
        if tools:
            tools = [ensure_tool_wrapper(t) for t in tools]
        self._agent = rust_class(model, tools, api_key)

    def invoke(self, query: str) -> AgentResponse:

        return self._agent.invoke(query)
    
    def run(self, query: str) -> str:
        return self._agent.run(query)
        
    def add_tool(self, tool):
        return self._agent.add_tool(ensure_tool_wrapper(tool))

    def __getattr__(self, name):
        return getattr(self._agent, name)


class GeminiModel(AgentWrapper):
    def __init__(self, model=None, tools=None, api_key=None):
        super().__init__(_rust.GeminiModel, model, tools, api_key)


class OpenAIModel(AgentWrapper):
    def __init__(self, model=None, tools=None, api_key=None):
        super().__init__(_rust.OpenAIModel, model, tools, api_key)


class ClaudeModel(AgentWrapper):
    def __init__(self, model=None, tools=None, api_key=None):
        super().__init__(_rust.ClaudeModel, model, tools, api_key)


__all__ = ['GeminiModel', 'OpenAIModel', 'ClaudeModel', 'create_agent', 'AgentResponse', 'ToolCall', 
           'tool', 'ToolAdapter']
