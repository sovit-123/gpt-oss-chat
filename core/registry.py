"""Registry of the tools the model can call.

Holds, for each tool, the JSON schema sent to the model and the function
that runs it. The engine asks the registry to execute calls instead of
dispatching to the tool functions itself."""


class ToolRegistry:
    """Tool functions raise on failure. execute() catches everything and
    returns (output, error), so a failed call goes back to the model as
    feedback instead of killing the turn."""

    def __init__(self):
        self._schemas = {}
        self._impls = {}

    def register(self, schema, fn):
        """Adds a tool. `schema` is the OpenAI-format description, `fn`
        runs it."""
        name = schema["function"]["name"]
        self._schemas[name] = schema
        self._impls[name] = fn

    def schemas(self, exclude=()):
        """Schema list for an API request, in registration order.
        `exclude` hides tools the session cannot use, e.g. local_rag
        when no document is loaded."""
        return [s for n, s in self._schemas.items() if n not in exclude]

    def execute(self, name, args):
        """Runs a tool and returns (output, error). Never raises: unknown
        tools, bad arguments and exceptions inside the tool all come back
        as errors."""
        if name not in self._impls:
            return f"Error: Unknown tool: {name}", True
        try:
            return str(self._impls[name](**args)), False
        except Exception as e:
            return f"Error: {name} failed: {e}", True


def default_registry():
    """Builds a registry with the four built-in tools. The import stays
    inside the function so `import core.registry` does not pull in the
    tool stack for nothing."""
    from tools.tools import tools, search_web, local_rag, url_search, code_search

    impls = {
        "search_web": search_web,
        "local_rag": local_rag,
        "url_search": url_search,
        "code_search": code_search,
    }
    registry = ToolRegistry()
    for schema in tools:
        registry.register(schema, impls[schema["function"]["name"]])
    return registry
