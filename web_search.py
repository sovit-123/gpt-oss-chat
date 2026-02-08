"""
This script contains code to perform:
1. Web search using Tavily and Perplexity APIs
2. URL search using Tavily API
"""

import os

from tavily import TavilyClient
from perplexity import Perplexity
from dotenv import load_dotenv

load_dotenv()

def do_web_search(query=None, search_engine='tavily', max_results=5):
    """
    Perform a web search using Tavily or Perplexity to get context.

    :param query: search query string (required)
    :param search_engine: search engine to use, either 'tavily' or 'perplexity' (default: 'tavily')
    :param max_results: maximum number of results to return (default: 5)

    Returns:
        retrieved_docs: a list of retrieved web search results as strings.
            e.g. ['context 1', 'context 2', ...]
    
    Raises:
        ValueError: if query is None or empty
        KeyError: if required API keys are not found in environment
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    
    if search_engine == 'tavily':
        TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
        if not TAVILY_API_KEY:
            raise KeyError('TAVILY_API_KEY not found in environment. Please check your .env file')
    
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query, max_results=max_results)
    
        results = [res['content'] for res in response['results']]
    
    elif search_engine == 'perplexity':
        PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
        if not PERPLEXITY_API_KEY:
            raise KeyError('PERPLEXITY_API_KEY not found in environment. Please check your .env file')

        ppxl_client = Perplexity()
        response = ppxl_client.search.create(
            query=query,
            max_results=max_results,
            max_tokens_per_page=512
        )

        results = [result.snippet for result in response.results]
    else:
        raise ValueError(f"Unsupported search engine: {search_engine}")

    return results

def do_url_search(url=None, search_engine='tavily'):
    """
    Perform a URL search using Tavily to get context.

    :param url: URL string to search (required)
    :param search_engine: search engine to use, currently only 'tavily' is supported (default: 'tavily')

    Returns:
        retrieved_docs: a list of retrieved URL search results as strings.
            e.g. ['context 1', 'context 2', ...]
    
    Raises:
        ValueError: if url is None or empty
        KeyError: if required API keys are not found in environment
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")
    
    if search_engine == 'tavily':
        TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
        if not TAVILY_API_KEY:
            raise KeyError('TAVILY_API_KEY not found in environment. Please check your .env file')
    
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.extract(url)
    
        results = [response['results'][0]['raw_content']]
    
    else:
        raise ValueError(f"Unsupported search engine for URL search: {search_engine}")

    return results

if __name__ == '__main__':
    # Check web_search.
    print('*' * 50)
    print(f"Checking web search with Tavily API...")
    query = 'What is the capital of France?'
    docs = do_web_search(query, search_engine='tavily')
    print(docs)
    print('*' * 50)

    # Check URL search.
    print('*' * 50)
    print(f"Checking URL search with Tavily API...")
    url = 'https://en.wikipedia.org/wiki/Artificial_intelligence'
    docs = do_url_search(url, search_engine='tavily')
    print(docs)
    print('*' * 50)