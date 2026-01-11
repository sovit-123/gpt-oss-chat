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
from tools import tools, search_web

import argparse
import sys
import json

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
You have access to the following tools:
1. search_web: Use this tool to search the web for up-to-date information.
Always use the tools when necessary to get accurate information.
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
def append_to_chat_history(
    role=None, 
    content=None, 
    chat_history=None, 
    tool_call_id=None,
    tool_identifier=False,
    tool_name=None,
    tool_args=None
):
    if tool_identifier:
        chat_history.append({
            "role": role,
            "content": content,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function", 
                "function": {
                    "name": tool_name,
                    "arguments": tool_args
                }
            }]
        })
        return chat_history
    if tool_call_id is not None:
        chat_history.append({'role': role, 'content': content, 'tool_call_id': tool_call_id})
    else:
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

def run_chat_loop(client, args, messages, console):
    """
    Reusable chat loop function that can be imported by other modules.
    
    Args:
        client: OpenAI client instance
        args: Parsed arguments containing web_search, search_engine, local_rag, and model
        messages: Chat history list
        console: Rich console instance for output
    
    Returns:
        messages: Updated chat history
    """
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

            # messages = append_to_chat_history({'role': 'user', 'content': user_input})
            messages = append_to_chat_history('user', user_input, messages)
            
            try:
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=True,
                    tools=tools,
                    tool_choice='auto',
                )

                # print(event.choices[0].delta.content for event in stream)  # Debug: Print each event received from the stream
                # print(stream)

            except APIError as e:
                console.print(f"[red]Error: API request failed: {e}[/red]")
                messages.pop()  # Remove the user message that failed
                continue

            # Process tool calls.
            tool_args = ''
            assistant_message_with_tool_call = ''
            tool_name = None
            tool_id = None
            dangling_stream_content = ''
            for event in stream:
                if event.choices[0].delta.tool_calls is not None:
                    if event.choices[0].delta.tool_calls[0].function.name is not None:
                        tool_name = event.choices[0].delta.tool_calls[0].function.name
                        tool_id = event.choices[0].delta.tool_calls[0].id
                        assistant_message_with_tool_call = event
                    tool_args += event.choices[0].delta.tool_calls[0].function.arguments
                if event.choices[0].delta.content is not None:
                    dangling_stream_content = event.choices[0].delta.content
                    break

            if tool_name is not None:
                console.print(f"[bold cyan]Using tool: {tool_name} ::: Args: {tool_args} [/bold cyan]")
                tool_args = json.loads(tool_args)
        
                # Execute tool call.
                if tool_name == 'search_web':
                    result = search_web(**tool_args)
                    # console.print(f"[dim]Tool result: {result}[/dim]")

                # Append assistant message with tool call to chat history.
                # messages = append_to_chat_history(
                #     'assistant',
                #     assistant_message_with_tool_call,
                #     chat_history=messages,
                #     tool_call_id=tool_id
                # )
                messages = append_to_chat_history(
                    role='assistant',
                    content='',
                    chat_history=messages,
                    tool_call_id=tool_id,
                    tool_identifier=True,
                    tool_name=tool_name,
                    tool_args=json.dumps(tool_args)
                )

                # Append tool result.
                messages = append_to_chat_history(
                    'tool',
                    str(result),
                    chat_history=messages,
                    tool_call_id=tool_id
                )

                # Append user query according to tool result.
                messages = append_to_chat_history(
                    'user',
                    "Answer the question based on the above tool result.",
                    chat_history=messages
                )


                # For debugging.
                # print(messages)

                # Make API call again with tool results added to the messages.
                console.print(f"[dim]Fetching final response from assistant...[/dim]")
                console.print()
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=True,
                    tools=tools,
                    tool_choice='auto',
                )

            # No tool call, just collect assistant response.
            # The logical flow also comes here when tool call is done and we 
            # are streaming final response.
            current_response = ''
            buffer = ''
            # print(f"Dangling stream content: '{dangling_stream_content}'")
            if len(dangling_stream_content) > 0:
                # console.print(f"[dim] Dangling string: '{dangling_stream_content}'[/dim]")
                buffer += dangling_stream_content
            console.print("[bold green]Assistant:[/bold green] ")
            with Live(
                Markdown(''), 
                console=console, 
                refresh_per_second=1,
                # vertical_overflow='visible'
                vertical_overflow='ellipsis'
            ) as live:
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

            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue
    
    return messages


def main():
    """Main entry point when running api_call.py directly."""
    run_chat_loop(client, args, messages, console)


if __name__ == '__main__':
    main()