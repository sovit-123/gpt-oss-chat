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
    }
]

def search_web(topic: str, search_engine: str) -> str:
    result =  do_web_search(
        topic, 
        search_engine=search_engine
    )
    return '\n'.join(result)

def local_rag(topic: str, top_k: int) -> str:
    hits, result = search_query(topic, top_k=top_k)

    return '\n'.join(result)

def url_search(url: str, search_engine: str) -> str:
    result = do_url_search(
        url, 
        search_engine=search_engine
    )
    return '\n'.join(result)