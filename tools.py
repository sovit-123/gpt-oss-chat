from web_search import do_web_search

def search_web(topic: str) -> str:
    result =  do_web_search(
        topic, 
        # search_engine='tavily'
    )
    return '\n'.join(result)
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a given topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"]
            },
        },
    }
]