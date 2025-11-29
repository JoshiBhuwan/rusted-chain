"""
Tool helpers for lang_rain.
Leverages LangChain for tool definition and execution.
"""

from langchain_core.tools import tool as lc_tool
from langchain_core.tools import BaseTool

# Re-export the LangChain tool decorator
tool = lc_tool

class ToolAdapter:
    """
    Adapts a LangChain tool to lang_rain's interface.
    Converts LangChain/Pydantic schemas to a clean JSON format suitable for LLMs.
    """
    
    def __init__(self, tool):
        self.tool = tool
        self.__name__ = tool.name
        
        # Extract schema using LangChain's built-in methods
        # The fun stuff is here
        try:
            parameters = tool.get_input_schema().model_json_schema()
        except AttributeError:
            parameters = tool.get_input_schema().schema()
            
        # Clean up schema
        # Many LLMs don't need 'title' fields in parameters
        if "title" in parameters:
            del parameters["title"]
            
        # Remove redundant description if it matches tool description
        if "description" in parameters and parameters["description"] == tool.description:
             del parameters["description"]
             
        self.schema = {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters
        }

    def __call__(self, **kwargs):
        """Execute the underlying LangChain tool."""
        return self.tool.invoke(kwargs)
    
    def to_dict(self):
        """Return the schema."""
        return self.schema


def ensure_tool_wrapper(obj):
    """
    Ensures the object is wrapped in a ToolAdapter.
    
    Accepts:
    1. Raw Python functions (converts to LangChain tool -> Adapter)
    2. LangChain tools (wraps in Adapter)
    3. Existing ToolAdapters (returns as is)
    """
    if isinstance(obj, ToolAdapter):
        return obj
    
    # If it's a raw function (and not a LangChain tool), convert it
    if callable(obj) and not hasattr(obj, "get_input_schema"):
        # Use LangChain's tool function to wrap it
        obj = lc_tool(obj)
        
    # Check for LangChain tool interface
    if hasattr(obj, "name") and hasattr(obj, "description") and hasattr(obj, "invoke") and hasattr(obj, "get_input_schema"):
        return ToolAdapter(obj)
    
    raise ValueError(f"Object {obj} is not a valid tool or callable function")
