from openai import OpenAI, APIError
from web_search import do_web_search
from semantic_engine import (
    read_pdf,
    chunk_text,
    create_and_upload_in_mem_collection,
    search_query
)
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser(
    description='RAG-powered chatbot with optional web search and local PDF support'
)
parser.add_argument(
    '--web-search',
    dest='web_search',
    action='store_true',
    help='Enable web search for answering queries'
)
parser.add_argument(
    '--search-engine',
    type=str,
    default='tavily',
    choices=['tavily', 'perplexity'],
    help='web search engine to use (default: tavily)'
)
parser.add_argument(
    '--local-rag',
    dest='local_rag',
    help='provide path of a local PDF file for RAG',
    default=None
)
parser.add_argument(
    '--model',
    type=str,
    default='model.gguf',
    help='model name to use (default: model.gguf)'
)
parser.add_argument(
    '--api-url',
    type=str,
    default='http://localhost:8080/v1',
    help='OpenAI API base URL (default: http://localhost:8080/v1)'
)
args = parser.parse_args()

system_message = """
You are a helpful assistant. You never say you are an OpenAI model or chatGPT.
You are here to help the user with their requests.
When the user asks who are you, you say that you are a helpful AI assistant.
"""

# Initialize Rich console
console = Console()

# Initialize OpenAI client
try:
    client = OpenAI(base_url=args.api_url, api_key='')
except Exception as e:
    console.print(f"[red]Error: Failed to initialize OpenAI client: {e}[/red]")
    sys.exit(1)

chat_history = []
def append_to_chat_history(role, content, chat_history):
    chat_history.append({'role': role, 'content': content})
    
    return chat_history

chat_history = append_to_chat_history('system', system_message, chat_history)

messages = chat_history

### Embed document for vector search ###
if args.local_rag is not None:
    if not Path(args.local_rag).exists():
        console.print(f"[red]Error: PDF file not found: {args.local_rag}[/red]")
        sys.exit(1)
    
    try:
        console.print("[cyan]Ingesting local document for RAG...[/cyan]")
        console.print("[cyan]Reading and creating chunks...[/cyan]")
        full_text = read_pdf(args.local_rag)
        documents = chunk_text(full_text, chunk_size=512, overlap=50)
        console.print(f"[green]✓ Total chunks created: {len(documents)}[/green]")
        
        console.print("[cyan]Creating Qdrant collection...[/cyan]")
        create_and_upload_in_mem_collection(documents=documents)
        console.print("[green]✓ RAG collection ready[/green]")
    except Exception as e:
        console.print(f"[red]Error processing PDF: {e}[/red]")
        sys.exit(1)
###########################################

# Display available features
console.print("[bold cyan]RAG-Powered Chatbot Started[/bold cyan]")
if args.web_search:
    console.print(f"[cyan]  • Web search enabled ({args.search_engine})[/cyan]")
if args.local_rag:
    console.print(f"[cyan]  • Local RAG enabled[/cyan]")
console.print("[cyan]Type 'exit' or 'quit' to end the conversation[/cyan]\n")

while True:
    try:
        user_input = console.input("[bold]You: [/bold]").strip()
        print()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            console.print("[yellow]Goodbye![/yellow]")
            break

        context_sources = []
        search_results = []
        ### Web search and context addition starts here ###
        if args.web_search:
            try:
                console.print(f"[dim]Searching with {args.search_engine}...[/dim]")
                web_search_results = do_web_search(
                    query=user_input, search_engine=args.search_engine
                )
                # context = "\n".join(web_search_results)
                # user_input = f"Use the following web search results as context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_input}"
                search_results.extend(web_search_results)
                context_sources.append("web search")
            except Exception as e:
                console.print(f"[yellow]Warning: Web search failed: {e}[/yellow]")
        ### Web search and context addition ends here ###

        ### Document retrieval begins here ###
        if args.local_rag is not None:
            try:
                console.print("[dim]Searching local documents...[/dim]")
                hits, local_search_results = search_query(user_input, top_k=3)
                # context = "\n".join(local_search_results)
                # user_input = f"Use the following document search results as context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_input}"
                search_results.extend(local_search_results)
                context_sources.append("local RAG")
            except Exception as e:
                console.print(f"[yellow]Warning: Document search failed: {e}[/yellow]")
        ### Document retrieval ends here ###

        # Update user input if search results are found.
        if len(search_results) > 0:
            context = "\n".join(search_results)
            user_input = f"Use the following search results as context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_input}"      

        messages = append_to_chat_history('user', user_input, messages)
        
        try:
            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True
            )
        except APIError as e:
            console.print(f"[red]Error: API request failed: {e}[/red]")
            messages.pop()  # Remove the user message that failed
            continue

        current_response = ''
        buffer = ''
        console.print("[bold green]Assistant:[/bold green] ")
        with Live(Markdown(''), console=console, refresh_per_second=1) as live:
            for event in stream:
                stream_content = event.choices[0].delta.content
                if stream_content is not None:
                    buffer += stream_content
                    live.update(Markdown(buffer))
                    current_response += stream_content

        messages = append_to_chat_history('assistant', current_response, messages)
        console.print()
        if context_sources:
            console.print(f"[dim](Sources: {', '.join(context_sources)})[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
        break
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        continue