"""A small registry that keeps track of the tools the assistant can use.
For every tool it holds the JSON schema that gets sent to the model plus
the Python function that actually runs it. When the engine receives a tool
call from the model, it hands that call to the registry rather than
dispatching to tool functions itself."""


class ToolRegistry:
    """One entry per tool: the OpenAI-format schema and the function that
    runs it. The tool functions in tools/tools.py follow an older
    convention: instead of raising an exception on failure, they return a
    string that starts with "Error:". The registry detects that prefix so
    failed calls keep behaving exactly as they did before."""

    def __init__(self):
        self._schemas = {}
        self._impls = {}

    def register(self, schema, fn):
        """Adds a tool under its own name. `schema` is the OpenAI-format
        description, `fn` is the function to call."""
        name = schema["function"]["name"]
        self._schemas[name] = schema
        self._impls[name] = fn

    def schemas(self, exclude=()):
        """Returns the schema list to attach to an API request, in
        registration order. Tools named in `exclude` are left out, which is
        how a tool the session cannot use (local_rag without a loaded
        document) stays hidden from the model."""
        return [s for n, s in self._schemas.items() if n not in exclude]

    def execute(self, name, args):
        """Runs a tool by name and returns (output, error). Everything that
        can go wrong — an unknown tool, bad arguments, an exception inside
        the tool — is caught here, because a failed call should go back to
        the model as feedback instead of crashing the turn."""
        if name not in self._impls:
            return f"Error: Unknown tool: {name}", True
        try:
            output = str(self._impls[name](**args))
        except Exception as e:
            return f"Error: {name} failed: {e}", True
        return output, output.startswith("Error")


def default_registry():
    """Builds a registry with the four built-in tools. The import sits
    inside the function on purpose: tools.tools imports semantic_engine,
    which loads a sentence-transformers model the moment it is imported.
    Keeping the import here makes `import core.registry` instant."""
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
