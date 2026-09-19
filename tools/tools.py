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
                "required": ["topic", "search_engine"]
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
                    "top_k": {"type": "integer"}, 
                    "topic": {"type": "string"}
                },
                "required": ["top_k", "topic"]
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
                "required": ["url", "search_engine"]
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

def search_web(topic: str, search_engine: str) -> str:
    try:
        result = do_web_search(
            topic,
            search_engine=search_engine
        )
    except Exception as e:
        return f"Error: web search ({search_engine}) failed: {e}"
    if not result:
        return "No web search results found."
    return '\n'.join(result)

def local_rag(topic: str, top_k: int) -> str:
    try:
        hits, result = search_query(topic, top_k=top_k)
    except Exception as e:
        return (
            "Error: local document search is unavailable (no document was "
            f"loaded in this session). Details: {e}"
        )
    if not result:
        return "No relevant passages found in the loaded document."
    return '\n'.join(result)

def url_search(url: str, search_engine: str) -> str:
    try:
        result = do_url_search(
            url,
            search_engine=search_engine
        )
    except Exception as e:
        return f"Error: URL search ({search_engine}) failed: {e}"
    if not result:
        return "No content extracted from the URL."
    return '\n'.join(result)

def code_search(directory: str, query: str, max_results: int = 100) -> str:
    """
    Perform a grep search in the specified directory for the given query.

    Args:
        directory (str): Path to the directory containing code files.
        query (str): Search query (e.g., function name, variable, etc.).
        max_results (int): Maximum number of results to return.

    Returns:
        str: String representation of matching lines with file paths.
    """
    import subprocess
    from pathlib import Path

    try:
        # Ensure the directory exists
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return f"Error: Directory not found or invalid: {directory}"

        # Run grep command to search for the query
        # Include all files in the directory and subdirectories.
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

    except Exception as e:
        return f"Error during code search: {e}"
    

if __name__ == '__main__':
    # Code search example usage
    import argparse

    parser = argparse.ArgumentParser(description='Test code search tool')
    parser.add_argument('--directory', type=str, required=True, help='Path to the directory containing code files')
    parser.add_argument('--query', type=str, required=True, help='Search query (e.g., function name, variable, etc.)')
    parser.add_argument('--max-results', type=int, default=10, help='Maximum number of results to return')
    args = parser.parse_args()

    search_results = code_search(args.directory, args.query, args.max_results)
    print(search_results)