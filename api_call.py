"""Terminal chat frontend: a thin renderer over the core engine.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from core.engine import Engine, EngineConfig, Session
from core.events import (
    AssistantDelta, Error, Status, ToolCallStarted, TurnFinished,
)
from core.registry import default_registry

load_dotenv()

console = Console()

# Status events are plain data, so the engine cannot know which messages
# deserve alarm coloring. This frontend guesses from the wording, which
# keeps every display decision in this file.
WARNING_MARKERS = ("failed", "duplicate", "could not")


def parse_args():
    """Same CLI as before the refactor. The interface did not change; the
    flags now feed an EngineConfig instead of a hand-rolled loop."""
    parser = argparse.ArgumentParser(
        description='RAG-powered chatbot with optional web search and local PDF support'
    )
    parser.add_argument(
        '--web-search', dest='web_search', action='store_true',
        help='Enable pre-query web search for answering queries'
    )
    parser.add_argument(
        '--search-engine', type=str, default='tavily',
        choices=['tavily', 'perplexity'],
        help='web search engine to use (default: tavily)'
    )
    parser.add_argument(
        '--local-rag', dest='local_rag', default=None,
        help='provide path of a local PDF file for RAG'
    )
    parser.add_argument(
        '--rag-tool', dest='rag_tool', default=None,
        help='provide a pdf path to enable local RAG as a tool; '
             'the model decides when to call it'
    )
    parser.add_argument(
        '--model', type=str, default='model.gguf',
        help='model name to use (default: model.gguf)'
    )
    parser.add_argument(
        '--api-url', type=str, default='http://localhost:8080/v1',
        help='OpenAI API base URL (default: http://localhost:8080/v1)'
    )
    return parser.parse_args()


def build_config(args):
    """Maps the CLI flags onto the engine settings for one turn. Built
    once before the loop because terminal flags cannot change mid-session."""
    if args.local_rag:
        rag_mode = "always_on"
    elif args.rag_tool:
        rag_mode = "as_tool"
    else:
        rag_mode = "off"
    return EngineConfig(
        model=args.model,
        pre_query_web_search=args.web_search,
        search_engine=args.search_engine,
        rag_mode=rag_mode,
    )


def ingest_document(pdf_path):
    """Reads the PDF, chunks it and fills the in-memory Qdrant collection.
    semantic_engine is imported here rather than at the top, so starting
    the chat without a document never loads the sentence-transformers
    model that semantic_engine pulls in at import time."""
    from semantic_engine import (
        read_pdf, chunk_text, create_and_upload_in_mem_collection,
    )
    console.print("[cyan]Ingesting local document for RAG...[/cyan]")
    console.print("[cyan]Reading and creating chunks...[/cyan]")
    full_text = read_pdf(pdf_path)
    documents = chunk_text(full_text, chunk_size=512, overlap=50)
    console.print(f"[green]✓ Total chunks created: {len(documents)}[/green]")
    console.print("[cyan]Creating Qdrant collection...[/cyan]")
    create_and_upload_in_mem_collection(documents=documents)
    console.print("[green]✓ RAG collection ready[/green]")


def render(event_stream):
    """Turns one turn's engine events into Rich output: streamed markdown
    for assistant text, cyan lines for tool calls, dim or yellow for
    status notes. Tool results are not shown, matching the old TUI."""
    live = None
    buffer = ""
    try:
        for event in event_stream:
            if isinstance(event, AssistantDelta):
                if live is None:
                    console.print("[bold green]Assistant:[/bold green] ")
                    live = Live(
                        Markdown(buffer), console=console,
                        refresh_per_second=10, vertical_overflow='ellipsis',
                    )
                    live.start()
                buffer += event.text
                live.update(Markdown(buffer))
                continue
            if live is not None:
                # A tool call or status note interrupts the text segment;
                # close it so the next text opens a fresh block.
                live.stop()
                console.print()
                live = None
                buffer = ""
            if isinstance(event, ToolCallStarted):
                console.print(
                    f"[bold cyan]Tool call {event.number}: {event.name} "
                    f"::: Args: {json.dumps(event.args)}[/bold cyan]"
                )
            elif isinstance(event, Status):
                style = "yellow" if any(
                    m in event.message.lower() for m in WARNING_MARKERS
                ) else "dim"
                console.print(f"[{style}]{event.message}[/{style}]")
            elif isinstance(event, TurnFinished):
                if event.sources:
                    console.print(f"[dim](Sources: {', '.join(event.sources)})[/dim]")
            elif isinstance(event, Error):
                console.print(f"[red]Error: {event.message}[/red]")
    finally:
        if live is not None:
            live.stop()
            console.print()


def main():
    args = parse_args()
    try:
        client = OpenAI(
            base_url=args.api_url,
            api_key=os.getenv("MODAL_API_KEY", "default"),
        )
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize OpenAI client: {e}[/red]")
        sys.exit(1)

    session = Session()
    engine = Engine(client, default_registry())

    if args.local_rag or args.rag_tool:
        pdf_path = args.local_rag or args.rag_tool
        if not Path(pdf_path).exists():
            console.print(f"[red]Error: PDF file not found: {pdf_path}[/red]")
            sys.exit(1)
        try:
            ingest_document(pdf_path)
        except Exception as e:
            console.print(f"[red]Error processing PDF: {e}[/red]")
            sys.exit(1)
        session.rag_ready = True

    console.print("[bold cyan]RAG-Powered Chatbot Started[/bold cyan]")
    if args.web_search:
        console.print(f"[cyan]  • Web search enabled ({args.search_engine})[/cyan]")
    if args.local_rag:
        console.print("[cyan]  • Local RAG enabled[/cyan]")
    console.print("[cyan]Type 'exit' or 'quit' to end the conversation[/cyan]\n")

    config = build_config(args)
    while True:
        try:
            user_input = console.input("[bold blue]You: [/bold blue]").strip()
            print()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            render(engine.run_turn(session, user_input, config))
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break


if __name__ == '__main__':
    main()
