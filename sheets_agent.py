"""
Sheets Agent - Interactive Excel/CSV Analysis Agent

A human-in-the-loop agent for comprehensive spreadsheet analysis.
Powered by gpt-oss-20b via llama.cpp.

Usage:
    python sheets_agent.py --file /path/to/data.xlsx
    python sheets_agent.py --file data.csv --output-dir ./analysis_output
    python sheets_agent.py --file data.xlsx --auto  # Reduced interaction mode
"""

from openai import OpenAI, APIError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.prompt import Prompt, Confirm
from pathlib import Path

from sheets.reader import SheetReader, format_dataframe_for_llm
from sheets.analyzer import SheetAnalyzer
from sheets.connection_finder import ConnectionFinder
from sheets.summarizer import SheetSummarizer
from sheets.tools import sheets_tools, execute_sheets_tool, set_sheets_context
from utils.prompt import SHEETS_AGENT_SYSTEM_MESSAGE, append_to_chat_history

import argparse
import sys
import json
import os
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Interactive Excel/CSV Analysis Agent powered by gpt-oss-20b'
    )
    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Path to Excel (.xlsx, .xls) or CSV file to analyze'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./sheets_output',
        help='Directory for output files (default: ./sheets_output)'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Reduced interaction mode - agent decides analysis focus'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='model.gguf',
        help='Model name to use (default: model.gguf)'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8080/v1',
        help='OpenAI API base URL (default: http://localhost:8080/v1)'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=5000,
        help='Maximum rows to analyze per sheet (default: 5000)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size for large files (default: 1000)'
    )
    parser.add_argument(
        '--max-tool-calls',
        type=int,
        default=15,
        help='Maximum tool calls before generating summary (default: 15)'
    )
    
    return parser.parse_args()


def display_workbook_overview(console: Console, reader: SheetReader):
    """Display an overview of the workbook to the user."""
    info = reader.get_sheet_info()
    
    console.print()
    console.print(Panel.fit(
        f'[bold cyan]📊 Workbook: {reader.file_path.name}[/bold cyan]',
        border_style='cyan'
    ))
    console.print()
    
    # Create overview table
    table = Table(title='Sheet Overview', show_header=True, header_style='bold magenta')
    table.add_column('Sheet Name', style='cyan')
    table.add_column('Rows', justify='right', style='green')
    table.add_column('Columns', justify='right', style='green')
    table.add_column('Headers Preview', style='dim')
    
    for sheet_name, sheet_info in info.items():
        headers = sheet_info.get('headers', [])
        headers_preview = ', '.join(headers[:5])
        if len(headers) > 5:
            headers_preview += f' ... (+{len(headers) - 5})'
        
        row_str = f'{sheet_info.get("rows", 0):,}'
        if sheet_info.get('is_sampled'):
            row_str += ' ⚠️'
        
        table.add_row(
            sheet_name,
            row_str,
            str(sheet_info.get('columns', 0)),
            headers_preview
        )
    
    console.print(table)
    console.print()
    
    # Check for sampling
    sampled_sheets = [
        name for name, info in info.items() 
        if info.get('is_sampled')
    ]
    if sampled_sheets:
        console.print(
            f'[yellow]⚠️ Note: {len(sampled_sheets)} sheet(s) have >5000 rows '
            f'and will be sampled for analysis.[/yellow]'
        )
        console.print()


def get_user_context(console: Console, reader: SheetReader, auto_mode: bool) -> tuple:
    """
    Get context and focus from the user.
    
    Returns:
        tuple: (user_context, focus_sheets)
    """
    if auto_mode:
        console.print('[dim]Auto mode: Agent will analyze all sheets automatically.[/dim]')
        return '', reader.get_sheet_names()
    
    console.print('[bold green]Please provide some context to guide the analysis:[/bold green]')
    console.print()
    
    # Get description of what the data is about
    user_context = Prompt.ask(
        '[cyan]What is this spreadsheet about?[/cyan]\n'
        '(e.g., "Sales data for Q1-Q4 2024", "Customer database", etc.)'
    )
    console.print()
    
    # Get focus sheets
    sheet_names = reader.get_sheet_names()
    
    if len(sheet_names) > 1:
        console.print(f'[cyan]Available sheets: {", ".join(sheet_names)}[/cyan]')
        focus_input = Prompt.ask(
            'Which sheets should I focus on? (comma-separated, or "all" for all sheets)',
            default='all'
        )
        
        if focus_input.lower() == 'all':
            focus_sheets = sheet_names
        else:
            focus_sheets = [s.strip() for s in focus_input.split(',')]
            # Validate sheet names
            focus_sheets = [s for s in focus_sheets if s in sheet_names]
            if not focus_sheets:
                console.print('[yellow]No valid sheets specified, analyzing all.[/yellow]')
                focus_sheets = sheet_names
    else:
        focus_sheets = sheet_names
    
    console.print()
    console.print(f'[green]✓ Will focus on: {", ".join(focus_sheets)}[/green]')
    console.print()
    
    return user_context, focus_sheets


def run_agent_loop(
    console: Console,
    client: OpenAI,
    args,
    reader: SheetReader,
    analyzer: SheetAnalyzer,
    connection_finder: ConnectionFinder,
    summarizer: SheetSummarizer,
    user_context: str,
    focus_sheets: list
):
    """
    Run the main agent analysis loop.
    
    The agent will use tools to analyze the data and may ask
    for user feedback during the process.
    """
    # Set up the sheets context for tools
    set_sheets_context(
        reader=reader,
        analyzer=analyzer,
        connection_finder=connection_finder,
        summarizer=summarizer,
        user_context=user_context,
        focus_sheets=focus_sheets
    )
    
    # Build initial context for the agent
    workbook_context = summarizer.get_llm_context(focus_sheets)
    
    initial_prompt = f"""Analyze this workbook and provide comprehensive insights.

User Context: {user_context if user_context else "Not provided"}

Focus Sheets: {', '.join(focus_sheets)}

Workbook Structure:
{workbook_context}

Please:
1. Start by listing the sheets to understand the structure
2. Analyze the headers and columns of the focus sheets
3. Look for patterns, key columns, and data quality issues
4. Find connections between sheets if there are multiple
5. Generate a comprehensive summary when done

If you need clarification about the data, use request_user_feedback to ask the user.
"""
    
    # Initialize chat history
    chat_history = []
    chat_history = append_to_chat_history('system', SHEETS_AGENT_SYSTEM_MESSAGE, chat_history)
    chat_history = append_to_chat_history('user', initial_prompt, chat_history)
    
    console.print(Panel.fit(
        '[bold cyan]🤖 Starting Analysis...[/bold cyan]',
        border_style='cyan'
    ))
    console.print()
    
    # Run single agent turn (initial analysis or follow-up)
    def run_agent_turn(chat_history, max_tool_calls):
        """Run a single agent turn with tool calls."""
        tool_call_count = 0
        
        while tool_call_count < max_tool_calls:
            try:
                # Make API call
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=chat_history,
                    stream=True,
                    tools=sheets_tools,
                    tool_choice='auto'
                )
                
                # Process stream for tool calls
                tool_args = ''
                tool_name = None
                tool_id = None
                response_content = ''
                
                for event in stream:
                    if event.choices[0].delta.tool_calls is not None:
                        if event.choices[0].delta.tool_calls[0].function.name is not None:
                            tool_name = event.choices[0].delta.tool_calls[0].function.name
                            tool_id = event.choices[0].delta.tool_calls[0].id
                        tool_args += event.choices[0].delta.tool_calls[0].function.arguments
                    
                    if event.choices[0].delta.content is not None:
                        response_content += event.choices[0].delta.content
                
                # If no tool call, return the response
                if tool_name is None:
                    if response_content:
                        chat_history = append_to_chat_history(
                            'assistant', response_content, chat_history
                        )
                        console.print('[bold green]Agent:[/bold green]')
                        console.print()
                        console.print(Markdown(response_content))
                    return chat_history, tool_call_count
                
                # Execute tool call
                tool_call_count += 1
                console.print(f'[cyan]🔧 Tool Call {tool_call_count}: {tool_name}[/cyan]')
                
                try:
                    tool_kwargs = json.loads(tool_args) if tool_args else {}
                except json.JSONDecodeError:
                    tool_kwargs = {}
                
                result = execute_sheets_tool(tool_name, **tool_kwargs)
                
                # Check for user feedback request
                if result.startswith('[USER_FEEDBACK_REQUESTED]'):
                    question = result.replace('[USER_FEEDBACK_REQUESTED]', '').strip()
                    console.print()
                    console.print(Panel(
                        f'[yellow]🤔 Agent Question:[/yellow]\n{question}',
                        border_style='yellow'
                    ))
                    
                    user_response = Prompt.ask('[cyan]Your response[/cyan]')
                    result = f'User response: {user_response}'
                    console.print()
                else:
                    # Display tool result summary
                    result_lines = result.split('\n')
                    preview = '\n'.join(result_lines[:5])
                    if len(result_lines) > 5:
                        preview += f'\n... ({len(result_lines) - 5} more lines)'
                    console.print(f'[dim]{preview}[/dim]')
                    console.print()
                
                # Append to chat history
                chat_history = append_to_chat_history(
                    role='assistant',
                    content='',
                    chat_history=chat_history,
                    tool_call_id=tool_id,
                    tool_identifier=True,
                    tool_name=tool_name,
                    tool_args=json.dumps(tool_kwargs)
                )
                
                chat_history = append_to_chat_history(
                    'tool',
                    result,
                    chat_history=chat_history,
                    tool_call_id=tool_id
                )
                
            except APIError as e:
                console.print(f'[red]API Error: {e}[/red]')
                return chat_history, tool_call_count
            except KeyboardInterrupt:
                console.print('\n[yellow]Interrupted by user.[/yellow]')
                return chat_history, tool_call_count
        
        if tool_call_count >= max_tool_calls:
            console.print(f'[yellow]Reached maximum tool calls ({max_tool_calls}).[/yellow]')
        
        return chat_history, tool_call_count
    
    # Run initial analysis
    chat_history, _ = run_agent_turn(chat_history, args.max_tool_calls)
    
    # Follow-up conversation loop
    console.print()
    console.print('[bold cyan]─' * 50 + '[/bold cyan]')
    console.print()
    console.print('[bold green]💬 Follow-up Mode[/bold green]')
    console.print('[dim]You can now ask follow-up questions about the data.')
    console.print('[dim]Type "done", "exit", or "quit" to finish and save outputs.[/dim]')
    console.print()
    
    while True:
        try:
            user_input = console.input('[bold blue]You: [/bold blue]').strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['done', 'exit', 'quit', 'q']:
                console.print('[dim]Ending follow-up session...[/dim]')
                break
            
            console.print()
            
            # Add user message to history
            chat_history = append_to_chat_history('user', user_input, chat_history)
            
            # Run agent turn for follow-up
            chat_history, _ = run_agent_turn(chat_history, 5)  # Fewer tool calls for follow-ups
            console.print()
            
        except KeyboardInterrupt:
            console.print('\n[yellow]Ending follow-up session...[/yellow]')
            break
        except EOFError:
            break
    
    return chat_history


def save_outputs(
    console: Console,
    summarizer: SheetSummarizer,
    output_dir: str,
    focus_sheets: list
):
    """Save analysis outputs to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    console.print()
    console.print('[bold cyan]💾 Saving outputs...[/bold cyan]')
    
    # Save markdown summary
    md_path = output_path / f'summary_{timestamp}.md'
    summarizer.generate_markdown_report(str(md_path), focus_sheets)
    console.print(f'  [green]✓[/green] Markdown summary: {md_path}')
    
    # Save JSON hierarchy
    json_path = output_path / f'hierarchy_{timestamp}.json'
    summarizer.generate_json_output(str(json_path), focus_sheets)
    console.print(f'  [green]✓[/green] JSON hierarchy: {json_path}')
    
    # Save connections if available
    try:
        conn_path = output_path / f'connections_{timestamp}.json'
        summarizer.generate_connections_json(str(conn_path))
        console.print(f'  [green]✓[/green] Connections map: {conn_path}')
    except Exception:
        pass  # No connection finder or no connections
    
    console.print()
    console.print(f'[bold green]✅ All outputs saved to: {output_path}[/bold green]')


def main():
    """Main entry point for the sheets agent."""
    args = parse_args()
    console = Console()
    
    # Banner
    console.print()
    console.print(Panel.fit(
        '[bold magenta]📊 Sheets Analysis Agent[/bold magenta]\n'
        '[dim]Powered by gpt-oss-20b via llama.cpp[/dim]',
        border_style='magenta'
    ))
    console.print()
    
    # Validate file
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f'[red]Error: File not found: {args.file}[/red]')
        sys.exit(1)
    
    # Initialize OpenAI client
    try:
        client = OpenAI(base_url=args.api_url, api_key='')
        console.print(f'[dim]Connected to: {args.api_url}[/dim]')
    except Exception as e:
        console.print(f'[red]Error: Failed to initialize OpenAI client: {e}[/red]')
        sys.exit(1)
    
    # Load workbook
    console.print(f'[cyan]Loading: {args.file}[/cyan]')
    
    try:
        reader = SheetReader(
            str(file_path),
            chunk_size=args.chunk_size,
            max_rows=args.max_rows
        )
    except Exception as e:
        console.print(f'[red]Error loading file: {e}[/red]')
        sys.exit(1)
    
    # Display overview
    display_workbook_overview(console, reader)
    
    # Get user context
    user_context, focus_sheets = get_user_context(console, reader, args.auto)
    
    # Initialize analysis components
    analyzer = SheetAnalyzer(reader)
    connection_finder = ConnectionFinder(reader) if len(reader.get_sheet_names()) > 1 else None
    summarizer = SheetSummarizer(reader, analyzer, connection_finder)
    
    # Confirm before starting
    if not args.auto:
        if not Confirm.ask('[cyan]Ready to start analysis?[/cyan]', default=True):
            console.print('[yellow]Analysis cancelled.[/yellow]')
            sys.exit(0)
    
    # Run agent loop
    run_agent_loop(
        console=console,
        client=client,
        args=args,
        reader=reader,
        analyzer=analyzer,
        connection_finder=connection_finder,
        summarizer=summarizer,
        user_context=user_context,
        focus_sheets=focus_sheets
    )
    
    # Save outputs
    if Confirm.ask('\n[cyan]Save analysis outputs to files?[/cyan]', default=True):
        save_outputs(console, summarizer, args.output_dir, focus_sheets)
    
    console.print()
    console.print('[bold green]🎉 Analysis complete![/bold green]')


if __name__ == '__main__':
    main()
