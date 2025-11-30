"""
LangChain-style tool helper for rusted_chain.

This allows you to create tools that combine both the schema AND the callable function,
so invoke() can automatically execute them.
"""

class Tool:
    """A tool that combines schema and callable for automatic execution."""
    
    def __init__(self, name: str, description: str, parameters: dict, func: callable):
        """
        Create a tool with both schema and callable.
        
        Args:
            name: Tool name
            description: What the tool does
            parameters: JSON schema for parameters
            func: Python callable that implements the tool
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
    
    def to_schema(self):
        """Convert to Gemini tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def __call__(self, **kwargs):
        """Execute the tool."""
        return self.func(**kwargs)


def tool(name: str, description: str, parameters: dict):
    """Decorator to create a Tool from a function."""
    def decorator(func):
        return Tool(name, description, parameters, func)
    return decorator


# Example usage:
if __name__ == "__main__":
    from rusted_chain import ChatGemini
    
    # Create a tool using the decorator
    @tool(
        name="get_word_length",
        description="Returns the length of a word",
        parameters={
            "type": "object",
            "properties": {
                "word": {"type": "string", "description": "The word to count"}
            },
            "required": ["word"]
        }
    )
    def get_word_length(word: str) -> int:
        return len(word)
    
    # Or create manually
    calculate_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        func=lambda a, b: a + b
    )
    
    # Pass to agent - invoke() will auto-execute!
    agent = ChatGemini("gemini-2.5-flash", [get_word_length, calculate_tool])
    
    # Simple call - tools execute automatically
    result = agent.invoke("How many letters in 'hello'?")
    print(result)
