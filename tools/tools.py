"""The four built-in tools: the JSON schemas the model sees and the
Python functions that run them. Functions raise on failure; the registry
turns exceptions into feedback for the model. Optional arguments need a
default in both the schema and the function, and only required ones go
in the schema's `required` list."""

from web_search import do_web_search, do_url_search
from semantic_engine import search_query

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a given topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "search_engine": {"type": "string", "default": "tavily"}
                },
                "required": ["topic"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local_rag",
            "description": "Search the document for local RAG information on a given topic when a document is passed by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "default": 3},
                    "topic": {"type": "string"}
                },
                "required": ["topic"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "url_search",
            "description": "Search a specific URL for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "search_engine": {"type": "string", "default": "tavily"}
                },
                "required": ["url"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_search",
            "description": "Search a directory for code snippets or context using grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10}
                },
                "required": ["directory", "query"]
            },
        },
    }
]

def search_web(topic: str, search_engine: str = 'tavily') -> str:
    result = do_web_search(topic, search_engine=search_engine)
    if not result:
        return "No web search results found."
    return '\n'.join(result)

def local_rag(topic: str, top_k: int = 3) -> str:
    hits, result = search_query(topic, top_k=top_k)
    if not result:
        return "No relevant passages found in the loaded document."
    return '\n'.join(result)

def url_search(url: str, search_engine: str = 'tavily') -> str:
    result = do_url_search(url, search_engine=search_engine)
    if not result:
        return "No content extracted from the URL."
    return '\n'.join(result)

def code_search(directory: str, query: str, max_results: int = 10) -> str:
    """
    Perform a grep search in the specified directory for the given query.

    Args:
        directory (str): Path to the directory containing code files.
        query (str): Search query (e.g., function name, variable, etc.).
        max_results (int): Maximum number of results to return.

    Returns:
        str: String representation of matching lines with file paths.

    Raises:
        FileNotFoundError: If the directory does not exist or is not a
            directory.
    """
    import subprocess
    from pathlib import Path

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found or invalid: {directory}")

    grep_command = [
        "grep",
        "-rnI",          # recursive, line numbers, ignore binary files
        "-C", "3",       # context lines around each match
        query,
        directory,
    ]
    result = subprocess.run(grep_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        return f"No matches found for query: {query}"

    # Limit by complete matches (grep separates match groups with '--'),
    # not by raw lines, so each returned match keeps its context.
    match_groups = result.stdout.strip().split("\n--\n")
    truncated = len(match_groups) > max_results
    output = "\n--\n".join(match_groups[:max_results])
    if truncated:
        output += f"\n\n[...truncated: showing {max_results} of {len(match_groups)} matches]"
    return output
