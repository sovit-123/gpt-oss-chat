import os

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
from tools.tools import (
    tools, 
    search_web, 
    local_rag, 
    url_search, 
    code_search
)
from utils.prompt import (
    SYSTEM_MESSAGE, 
    append_to_chat_history, 
    build_system_message, 
    build_system_message
)
from dotenv import load_dotenv

import argparse
import sys
import json

load_dotenv()

API_KEY = os.getenv("MODAL_API_KEY", "default")

# Instruction appended when forcing a final text answer after the tool-call
# budget is exhausted. gpt-oss chat templates keep tool descriptions visible in
# the system prompt, so simply omitting `tools` from the request is not always
# enough to stop the model from emitting raw tool-call text.
FINAL_ANSWER_INSTRUCTION = (
    "SYSTEM NOTE: The tool-call limit has been reached and no tools are "
    "available for this reply. Do not call any tools and do not output any "
    "tool-call syntax. Answer the user's original question directly and "
    "completely, synthesizing only from the tool results already provided above."
)

RETRY_ANSWER_INSTRUCTION = (
    "Your previous reply contained a tool call or tool-error text instead of "
    "an answer. No tools are available for this reply. Write your final answer "
    "to the user's original question now, in plain text only, using the tool "
    "results already provided."
)


def looks_like_leaked_tool_call(text):
    """
    True if text is empty or looks like raw tool-call syntax / tool-error text
    rather than a real assistant answer (llama.cpp wraps tool calls it cannot
    resolve into an error message that arrives as ordinary content).
    """
    if not text or not text.strip():
        return True
    lowered = text.lower()
    markers = (
        '<function=',
        'tried to call unavailable tool',
        'arguments provided to the tool are invalid',
    )
    return any(marker in lowered for marker in markers)


def stream_final_response(console, client, model, messages=None, stream=None, initial_buffer=''):
    """Consume a completion stream via Rich Live and return the full text.

    Pass `messages` to start a new (tool-less) request, or `stream` to consume
    an already-open one.
    """
    if stream is None:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
    buffer = initial_buffer
    console.print("[bold green]Assistant:[/bold green] ")
    with Live(
        Markdown(buffer),
        console=console,
        refresh_per_second=10,
        vertical_overflow='ellipsis'
    ) as live:
        for event in stream:
            stream_content = event.choices[0].delta.content
            if stream_content is not None:
                buffer += stream_content
                live.update(Markdown(buffer))
    console.print()
    return buffer


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
    '--rag-tool',
    dest='rag_tool',
    help='provide a pdf path to enable local RAG tool, \
          the model decides when to call the RAG tool instead of each call'
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
# parser.add_argument(
#     '--code-dir',
#     type=str,
#     default=None,
#     help='Path to the directory containing code files for code search'
# )
args = parser.parse_args()

# Initialize Rich console
console = Console()

# Initialize OpenAI client
try:
    client = OpenAI(base_url=args.api_url, api_key=API_KEY)
except Exception as e:
    console.print(f"[red]Error: Failed to initialize OpenAI client: {e}[/red]")
    sys.exit(1)

# Manage initial chat history.
# Only advertise tools that are actually usable in this session: local_rag
# requires an ingested document, otherwise the model may call it and fail.
rag_loaded = args.local_rag is not None or args.rag_tool is not None
available_tools = tools if rag_loaded else [
    t for t in tools if t["function"]["name"] != "local_rag"
]

chat_history = []
chat_history = append_to_chat_history('system', build_system_message(include_local_rag=rag_loaded), chat_history)
messages = chat_history

### Embed document for vector search ###
if args.local_rag is not None or args.rag_tool is not None:
    if not Path(args.local_rag or args.rag_tool).exists():
        console.print(f"[red]Error: PDF file not found: {args.local_rag}[/red]")
        sys.exit(1)
    
    try:
        console.print("[cyan]Ingesting local document for RAG...[/cyan]")
        console.print("[cyan]Reading and creating chunks...[/cyan]")
        full_text = read_pdf(args.local_rag or args.rag_tool)
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
            user_input = console.input("[bold blue]You: [/bold blue]").strip()
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

            if args.rag_tool is not None:
                user_input += ' User has passed a document that can be used for local_rag tool'

            # messages = append_to_chat_history({'role': 'user', 'content': user_input})
            messages = append_to_chat_history('user', user_input, messages)
            
            try:
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=True,
                    tools=available_tools,
                    tool_choice='auto',
                )

                # print(event.choices[0].delta.content for event in stream)  # Debug: Print each event received from the stream
                # print(stream)

            except APIError as e:
                console.print(f"[red]Error: API request failed: {e}[/red]")
                messages.pop()  # Remove the user message that failed
                continue

            # Multi-turn tool call loop. 
            # The assistant will keep on calling needed tools until it's ready to respond.
            MAX_TOOL_CALLS = 5  # Safety limit to prevent infinite loops
            MAX_TOOL_RESULT_CHARS = 4000  # Cap tool results to avoid context bloat
            tool_call_count = 0
            tool_call_cache = {}  # Dedupe identical tool calls within this turn

            # vLLM/Qwen can emit preamble text AND a tool call in the same
            # response, so each pass collects content and tool-call deltas
            # together instead of assuming "content started => no tool call"
            # (that only holds for gpt-oss on llama.cpp). Text streams to the
            # terminal as it arrives; if a tool call follows, the collected
            # text stays visible as a preamble.
            while tool_call_count < MAX_TOOL_CALLS:
                tool_args_str = ''
                tool_name = None
                tool_id = None
                first_call_index = None
                buffer = ''  # Text of this response (preamble or final answer)
                live = None  # Rich Live display, started lazily on first text

                try:
                    for event in stream:
                        # Skip keep-alive/usage chunks with empty choices (vLLM sends them).
                        if len(event.choices) == 0:
                            continue
                        delta = event.choices[0].delta
                        if delta.tool_calls:
                            tool_call = delta.tool_calls[0]
                            if first_call_index is None:
                                first_call_index = tool_call.index
                            if tool_call.index != first_call_index:
                                # Parallel calls: only the first is executed per pass.
                                console.print(f"[yellow]Ignoring parallel tool call at index {tool_call.index}[/yellow]")
                                continue
                            if tool_call.id is not None:
                                tool_id = tool_call.id
                            if tool_call.function and tool_call.function.name is not None:
                                tool_name = tool_call.function.name
                            if tool_call.function and tool_call.function.arguments:
                                tool_args_str += tool_call.function.arguments
                        if delta.content:
                            if live is None:
                                console.print("[bold green]Assistant:[/bold green] ")
                                live = Live(
                                    Markdown(buffer),
                                    console=console,
                                    refresh_per_second=10,
                                    vertical_overflow='ellipsis'
                                )
                                live.start()
                            buffer += delta.content
                            live.update(Markdown(buffer))
                finally:
                    if live is not None:
                        live.stop()
                        console.print()

                print(f"Tool call detected: {tool_name} with args: {tool_args_str}")

                if tool_name is None:
                    # Pure text response: it is the final answer, already streamed.
                    break

                tool_call_count += 1
                console.print(f"[bold cyan]Tool call {tool_call_count}: {tool_name} ::: Args: {tool_args_str}[/bold cyan]")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    console.print(f"[yellow]Warning: Could not parse tool arguments for {tool_name}: {tool_args_str}[/yellow]")
                    break
        
                # Deduplicate repeated tool calls: the system prompt limits
                # repeat calls, but gpt-oss does not always follow it. If an
                # identical call was already made, return the prior result
                # instead of executing the tool again.
                call_key = f"{tool_name}::{json.dumps(tool_args, sort_keys=True)}"
                if call_key in tool_call_cache:
                    console.print(f"[yellow]Duplicate tool call skipped: {tool_name}[/yellow]")
                    result = tool_call_cache[call_key]
                else:
                    # Execute tool call. Errors are returned to the model as a
                    # tool result so it can recover instead of crashing the turn.
                    try:
                        if tool_name == 'search_web':
                            result = search_web(**tool_args)
                        elif tool_name == 'local_rag':
                            result = local_rag(**tool_args)
                        elif tool_name == 'url_search':
                            result = url_search(**tool_args)
                        elif tool_name == 'code_search':
                            result = code_search(**tool_args)
                        else:
                            console.print(f"[yellow]Warning: Unknown tool: {tool_name}[/yellow]")
                            result = f"Error: Unknown tool: {tool_name}"
                    except TypeError as e:
                        console.print(f"[yellow]Warning: Invalid arguments for {tool_name}: {e}[/yellow]")
                        result = f"Error: invalid arguments for {tool_name}: {e}"
                    except Exception as e:
                        console.print(f"[yellow]Warning: {tool_name} failed: {e}[/yellow]")
                        result = f"Error: {tool_name} failed: {e}"

                    # Truncate oversized tool results to avoid context bloat.
                    result = str(result)
                    if len(result) > MAX_TOOL_RESULT_CHARS:
                        result = result[:MAX_TOOL_RESULT_CHARS] + "\n...[result truncated]"

                    # Cache only successful results so transient errors can be retried.
                    if not result.startswith("Error"):
                        tool_call_cache[call_key] = result

                # Append assistant message with tool call to chat history.
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

                # Make another API call to let the model decide:
                # - Call another tool, OR
                # - Generate the final text response.
                console.print(f"[dim]Checking if more tools are needed... ({tool_call_count}/{MAX_TOOL_CALLS})[/dim]")
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=True,
                    tools=available_tools,
                    tool_choice='auto',
                )

            # `current_response` now holds the final assistant response in
            # every path: either the tool loop already streamed it (the model
            # answered with text only), or it is streamed below after the
            # tool-call budget forced a text-only reply.
            if tool_call_count >= MAX_TOOL_CALLS:
                console.print(f"[yellow]Warning: Reached maximum tool calls ({MAX_TOOL_CALLS})[/yellow]")
                # Force a final text response: the last stream in the loop was
                # requested with tools, so the model may emit another tool call
                # whose content would never render. Re-issue without tools and
                # with an explicit instruction to synthesize an answer.
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages + [{
                        'role': 'user',
                        'content': FINAL_ANSWER_INSTRUCTION
                    }],
                    stream=True,
                )
                current_response = stream_final_response(
                    console, client, args.model, stream=stream
                )
            else:
                current_response = buffer

            # Safety net: if the model still emitted raw tool-call text (or
            # nothing at all), retry once with a stronger instruction.
            if looks_like_leaked_tool_call(current_response):
                console.print("[yellow]Model returned a tool call instead of an answer; retrying...[/yellow]")
                current_response = stream_final_response(
                    console, client, args.model,
                    messages=messages + [
                        {'role': 'user', 'content': FINAL_ANSWER_INSTRUCTION},
                        {'role': 'assistant', 'content': current_response},
                        {'role': 'user', 'content': RETRY_ANSWER_INSTRUCTION},
                    ]
                )

            messages = append_to_chat_history('assistant', current_response, messages)
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