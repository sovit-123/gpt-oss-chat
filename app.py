"""
Gradio Web UI for RAG-powered chatbot with terminal-style aesthetics.
Styled to look like Rich console output with dark theme, syntax highlighting,
and formatted tables.
"""

import json
import gradio as gr
import os

from dotenv import load_dotenv

from openai import OpenAI, APIError

from web_search import do_web_search
from semantic_engine import (
    read_pdf,
    chunk_text,
    create_and_upload_in_mem_collection,
    search_query,
)
from tools.tools import (
    tools,
    search_web,
    local_rag,
    url_search,
    code_search,
)
from utils.prompt import SYSTEM_MESSAGE, append_to_chat_history, build_system_message

load_dotenv()

API_KEY = os.getenv("MODAL_API_KEY", "default")
MAX_TOOL_CALLS = 5
MAX_TOOL_RESULT_CHARS = 4000

# Terminal-style CSS to mimic Rich console
# Palette: Omarchy "Matte Black" theme (dark background, orange accent)
TERMINAL_CSS = """
/* Import monospace font */
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&display=swap');

/* Main container styling */
html, body {
    height: 100% !important;
    margin: 0 !important;
    overflow: hidden !important;
}

.gradio-container {
    background-color: #121212 !important;
    font-family: 'Fira Code', 'Ubuntu Mono', 'Consolas', 'Monaco', monospace !important;
    max-width: 100% !important;
    height: 100vh !important;
    height: 100dvh !important;
    overflow: hidden !important;
    padding: 8px 16px !important;
    display: flex !important;
    flex-direction: column !important;
}

/* One-page layout */
.header,
.gradio-container > .header {
    flex: 0 0 auto !important;
    margin-bottom: 2px !important;
}

.header h1 {
    font-size: 1.2rem !important;
    margin: 0 !important;
}

.header h3 {
    font-size: 0.85rem !important;
    margin: 0 !important;
    color: #8a8a8d !important;
}

.gradio-container > .main,
.gradio-container > .wrap,
.gradio-container > div:not(.header) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

#main-row {
    display: flex !important;
    flex-direction: row !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: auto !important;
    align-items: stretch !important;
}

#chat-column {
    position: relative !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
}

/* Full-viewport chat area: chat expands, input row stays visible */
#chatbox {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 96px !important;
    height: auto !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

#chatbox label,
#chatbox .label-wrap {
    display: none !important;
}

#chatbox .wrap,
#chatbox > div,
#chatbox .chatbot,
#chatbox [data-testid="chatbot"] {
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    overflow-y: auto !important;
}

#chat-input-row {
    position: absolute !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    overflow: visible !important;
    z-index: 20 !important;
    background-color: #121212 !important;
}

#chat-input-row .block {
    margin-bottom: 0 !important;
}

#sidebar {
    height: 100% !important;
    min-height: 0 !important;
    overflow-y: auto !important;
}

footer {
    display: none !important;
}

/* Chat container */
.chatbot {
    background-color: #121212 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

/* Message bubbles - Gradio 6.0 specific */
[data-testid="bot"], [data-testid="user"] {
    font-family: 'Fira Code', 'Ubuntu Mono', 'Consolas', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* User messages */
[data-testid="user"] {
    background-color: #1e1e1e !important;
    border-left: 3px solid #f59e0b !important;
}

/* Bot messages */
[data-testid="bot"] {
    background-color: #0d0d0d !important;
    border-left: 3px solid #e68e0d !important;
}

/* Code blocks - terminal style with proper syntax highlighting */
pre {
    background-color: #090909 !important;
    border: 1px solid #333333 !important;
    border-radius: 6px !important;
    padding: 16px !important;
    font-family: 'Fira Code', 'Ubuntu Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    overflow-x: auto !important;
    color: #bebebe !important;
    text-decoration: none !important;
}

pre code {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    font-family: inherit !important;
    font-size: inherit !important;
    color: inherit !important;
    text-decoration: none !important;
}

/* Inline code */
code:not(pre code) {
    background-color: #1e1e1e !important;
    border: 1px solid #333333 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.9em !important;
    color: #f59e0b !important;
    text-decoration: none !important;
}

/* Remove any strikethrough effects */
pre *, code *, pre, code {
    text-decoration: none !important;
    text-decoration-line: none !important;
}

/* Syntax highlighting - Omarchy Matte Black palette (orange accents) */
.hljs-keyword, .token.keyword { color: #e68e0d !important; }
.hljs-built_in, .token.builtin { color: #f59e0b !important; }
.hljs-type, .token.class-name { color: #f59e0b !important; }
.hljs-literal, .token.boolean { color: #ffc107 !important; }
.hljs-number, .token.number { color: #ffc107 !important; }
.hljs-string, .token.string { color: #ffc107 !important; }
.hljs-comment, .token.comment { color: #555555 !important; font-style: italic; }
.hljs-function, .token.function { color: #f59e0b !important; }
.hljs-params { color: #d35f5f !important; }
.hljs-attr, .token.attr-name { color: #e68e0d !important; }
.hljs-variable, .token.variable { color: #bebebe !important; }
.hljs-punctuation, .token.punctuation { color: #8a8a8d !important; }
.hljs-operator, .token.operator { color: #e68e0d !important; }

/* Tables styling */
table {
    border-collapse: collapse !important;
    background-color: #090909 !important;
    border: 1px solid #333333 !important;
    margin: 10px 0 !important;
    width: 100% !important;
}

th {
    background-color: #1e1e1e !important;
    color: #f59e0b !important;
    padding: 10px 14px !important;
    border: 1px solid #333333 !important;
    font-weight: 600 !important;
    text-align: left !important;
}

td {
    padding: 8px 14px !important;
    border: 1px solid #333333 !important;
    color: #bebebe !important;
}

tr:nth-child(even) {
    background-color: #0d0d0d !important;
}

/* Input textbox */
textarea, input[type="text"] {
    background-color: #090909 !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
    font-family: 'Fira Code', 'Ubuntu Mono', monospace !important;
}

/* Buttons */
button.primary, .primary {
    background-color: #e68e0d !important;
    color: #121212 !important;
    border: none !important;
}

button.primary:hover, .primary:hover {
    background-color: #f59e0b !important;
}

/* Labels and text */
label, .label-text, span {
    color: #bebebe !important;
    font-family: 'Fira Code', monospace !important;
}

/* Accordion headers */
.accordion {
    background-color: #0d0d0d !important;
    border: 1px solid #333333 !important;
}

/* File upload */
.file-upload {
    background-color: #090909 !important;
    border: 2px dashed #333333 !important;
}

/* Checkboxes */
input[type="checkbox"] {
    accent-color: #e68e0d !important;
}

/* Markdown list styling */
ul, ol {
    color: #bebebe !important;
}

li::marker {
    color: #f59e0b !important;
}

/* Bold text */
strong, b {
    color: #e68e0d !important;
    font-weight: 600 !important;
}

/* Italic text */
em, i {
    color: #f59e0b !important;
}

/* Links */
a {
    color: #f59e0b !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
    background-color: #121212;
}

::-webkit-scrollbar-thumb {
    background-color: #333333;
    border-radius: 4px;
}

/* Header styling */
h1, h2, h3, h4 {
    color: #e68e0d !important;
    font-family: 'Fira Code', monospace !important;
}
"""

# Global state for RAG collection
rag_ready = False
pdf_path = None


def process_pdf(file):
    """Process uploaded PDF for RAG."""
    global rag_ready, pdf_path

    if file is None:
        return "No file uploaded"

    try:
        pdf_path = file if isinstance(file, str) else file.name
        full_text = read_pdf(pdf_path)
        documents = chunk_text(full_text, chunk_size=512, overlap=50)
        create_and_upload_in_mem_collection(documents=documents)
        rag_ready = True
        return f"PDF processed: {len(documents)} chunks created"
    except Exception as e:
        rag_ready = False
        return f"Error processing PDF: {e}"


def dispatch_tool(tool_name, tool_args):
    """Route a tool call to the correct function and return the result string."""
    print(f"Dispatching tool: {tool_name} with args: {tool_args}")
    if tool_name == 'search_web':
        return search_web(**tool_args)
    elif tool_name == 'local_rag':
        return local_rag(**tool_args)
    elif tool_name == 'url_search':
        return url_search(**tool_args)
    elif tool_name == 'code_search':
        return code_search(**tool_args)
    else:
        return f"Error: Unknown tool: {tool_name}"


def looks_like_leaked_tool_call(text):
    """
    Detect responses where the model emitted raw tool-call syntax as text
    (happens when tools are removed but the model still tries to call one,
    e.g. llama.cpp's "Model tried to call unavailable tool" wrapper).
    """
    if not text or not text.strip():
        return True
    lowered = text.lower()
    markers = (
        'function=',
        '<function',
        'tried to call unavailable tool',
        'arguments provided to the tool are invalid',
    )
    return any(marker in lowered for marker in markers)


NO_MORE_TOOLS_MSG = (
    "SYSTEM NOTE: The tool-call limit has been reached. Do NOT call any more "
    "tools and do NOT output tool-call syntax. Answer the user's original "
    "question directly and completely, using only the tool results you have "
    "already gathered."
)

RETRY_NO_TOOLS_MSG = (
    "Your previous reply contained a tool call or tool-error text instead of "
    "an answer. No tools are available. Write your final answer to the user's "
    "original question now, in plain text only, using the tool results already "
    "provided."
)


def chat(message, history, api_messages, api_url, model_name,
         enable_web_search, search_engine, rag_mode):
    """
    Main chat generator with full multi-turn tool-calling support,
    mirroring the logic in api_call.py.

    Yields: (history, "", api_messages) tuples for Gradio outputs.
    """
    global rag_ready

    if not message.strip():
        yield history, "", api_messages
        return

    try:
        client = OpenAI(base_url=api_url, api_key=API_KEY)
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error initializing client: {e}"})
        yield history, "", api_messages
        return

    user_input = message
    search_results = []
    context_sources = []

    # Only expose tools that are usable in this session. local_rag requires an
    # ingested PDF collection, so hide it (and its prompt description) until a
    # PDF has been uploaded.
    available_tools = tools if rag_ready else [
        t for t in tools if t["function"]["name"] != "local_rag"
    ]
    if api_messages and api_messages[0].get('role') == 'system':
        api_messages[0] = {
            'role': 'system',
            'content': build_system_message(include_local_rag=rag_ready),
        }

    # Pre-query web search (optional, user-toggled)
    if enable_web_search:
        try:
            web_results = do_web_search(query=message, search_engine=search_engine)
            search_results.extend(web_results)
            context_sources.append(f"web search ({search_engine})")
        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Warning: Web search failed: {e}"})
            yield history, "", api_messages

    # Pre-query local RAG retrieval (always-on mode)
    if rag_mode == "Always-on retrieval" and rag_ready:
        try:
            _, local_results = search_query(message, top_k=3)
            search_results.extend(local_results)
            context_sources.append("local RAG")
        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Warning: Document search failed: {e}"})
            yield history, "", api_messages

    if search_results:
        context = "\n".join(search_results)
        user_input = (
            "Use the following search results as context to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {message}"
        )

    # RAG-as-tool hint (mirrors api_call.py --rag-tool behaviour)
    if rag_mode == "As tool (model decides)" and rag_ready:
        user_input += ' User has passed a document that can be used for local_rag tool'

    # Append user message to API history
    api_messages = append_to_chat_history('user', user_input, api_messages)

    # Show user message in chat
    history.append({"role": "user", "content": message})
    yield history, "", api_messages

    # First API call with tools
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=api_messages,
            stream=True,
            tools=available_tools,
            tool_choice='auto',
        )
    except APIError as e:
        api_messages.pop()
        history.append({"role": "assistant", "content": f"API Error: {e}"})
        yield history, "", api_messages
        return

    # Multi-turn tool-call loop
    tool_call_count = 0
    tool_call_cache = {}  # Dedupe identical tool calls within this turn

    # vLLM/Qwen can emit preamble text AND a tool call in the same response,
    # so each pass collects content and tool-call deltas together instead of
    # assuming "content started => no tool call" (that only holds for gpt-oss
    # on llama.cpp). Text streams to the UI as it arrives; if a tool call
    # follows, the collected text stays visible as a preamble message.
    while tool_call_count < MAX_TOOL_CALLS:
        tool_args_str = ''
        tool_name = None
        tool_id = None
        first_call_index = None
        buffer = ''  # Text of this response (preamble or final answer)

        history.append({"role": "assistant", "content": buffer})
        yield history, "", api_messages

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
                    print(f"Ignoring parallel tool call at index {tool_call.index}")
                    continue
                if tool_call.id is not None:
                    tool_id = tool_call.id
                if tool_call.function and tool_call.function.name is not None:
                    tool_name = tool_call.function.name
                if tool_call.function and tool_call.function.arguments:
                    tool_args_str += tool_call.function.arguments
            if delta.content:
                buffer += delta.content
                history[-1] = {"role": "assistant", "content": buffer}
                yield history, "", api_messages

        print(f"Tool call detected: {tool_name} with args: {tool_args_str}")

        if tool_name is None:
            # Pure text response: it is the final answer, already streamed.
            break

        tool_call_count += 1

        # Drop the placeholder bubble when the model emitted no preamble
        # text before the tool call.
        if not buffer:
            history.pop()

        # Show tool-call status in the UI (any preamble text stays above)
        history.append({
            "role": "assistant",
            "content": f"**Tool call {tool_call_count}: {tool_name}**\nArgs: `{tool_args_str}`"
        })
        yield history, "", api_messages

        try:
            tool_args = json.loads(tool_args_str)
        except json.JSONDecodeError:
            history.append({
                "role": "assistant",
                "content": f"Error: Could not parse tool arguments: `{tool_args_str}`"
            })
            yield history, "", api_messages
            return

        # Deduplicate identical tool calls within this turn: the system prompt
        # limits repeat calls, but gpt-oss does not always follow it.
        call_key = f"{tool_name}::{json.dumps(tool_args, sort_keys=True)}"
        if call_key in tool_call_cache:
            result = tool_call_cache[call_key]
        else:
            # Errors are returned to the model as a tool result so it can
            # recover instead of the whole turn crashing.
            try:
                result = dispatch_tool(tool_name, tool_args)
            except Exception as e:
                result = f"Error: {tool_name} failed: {e}"
            result = str(result)
            if len(result) > MAX_TOOL_RESULT_CHARS:
                result = result[:MAX_TOOL_RESULT_CHARS] + "\n...[result truncated]"
            # Cache only successful results so transient errors can be retried.
            if not result.startswith("Error"):
                tool_call_cache[call_key] = result

        # Record assistant tool-call in API history
        api_messages = append_to_chat_history(
            role='assistant',
            content='',
            chat_history=api_messages,
            tool_call_id=tool_id,
            tool_identifier=True,
            tool_name=tool_name,
            tool_args=json.dumps(tool_args),
        )

        # Record tool result in API history
        api_messages = append_to_chat_history(
            'tool',
            str(result),
            chat_history=api_messages,
            tool_call_id=tool_id,
        )

        # Show progress
        history.append({
            "role": "assistant",
            "content": f"*Checking if more tools are needed... ({tool_call_count}/{MAX_TOOL_CALLS})*"
        })
        yield history, "", api_messages

        # Next API call so model can call another tool or respond
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                stream=True,
                tools=available_tools,
                tool_choice='auto',
            )
        except APIError as e:
            history.append({"role": "assistant", "content": f"API Error during tool loop: {e}"})
            yield history, "", api_messages
            return

    # From here on, `buffer` and `history[-1]` must hold the final assistant
    # response in every path: either the tool loop streamed it (the model
    # answered with text only), or it is streamed below after the tool-call
    # budget forced a text-only reply.
    if tool_call_count >= MAX_TOOL_CALLS:
        history.append({
            "role": "assistant",
            "content": f"Warning: Reached maximum tool calls ({MAX_TOOL_CALLS}). Generating final response without tools."
        })
        yield history, "", api_messages
        # Force a text-only response: omit tools and explicitly instruct the
        # model to synthesize an answer (gpt-oss templates keep tool
        # descriptions visible in the system prompt, so omitting `tools`
        # alone may not stop it from emitting raw tool-call text).
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=api_messages + [{'role': 'user', 'content': NO_MORE_TOOLS_MSG}],
                stream=True,
            )
        except APIError as e:
            history.append({"role": "assistant", "content": f"API Error: {e}"})
            yield history, "", api_messages
            return

        buffer = ''
        history.append({"role": "assistant", "content": buffer})
        yield history, "", api_messages
        try:
            for event in stream:
                if len(event.choices) == 0:
                    continue
                stream_content = event.choices[0].delta.content
                if stream_content:
                    buffer += stream_content
                    history[-1] = {"role": "assistant", "content": buffer}
                    yield history, "", api_messages
        except Exception as e:
            print(f"Error during streaming: {e}")
            history[-1] = {"role": "assistant", "content": buffer + f"\n\nError during streaming: {e}"}
            yield history, "", api_messages
            return

    # Safety net: if the response is empty or still looks like raw tool-call
    # text (llama.cpp wraps unresolvable tool calls in an error message that
    # arrives as content), retry once with a stronger no-tools instruction.
    if looks_like_leaked_tool_call(buffer):
        history[-1] = {
            "role": "assistant",
            "content": "*Model attempted another tool call instead of answering; retrying...*"
        }
        yield history, "", api_messages
        retry_messages = api_messages + [
            {'role': 'user', 'content': NO_MORE_TOOLS_MSG},
            {'role': 'assistant', 'content': buffer},
            {'role': 'user', 'content': RETRY_NO_TOOLS_MSG},
        ]
        buffer = ''
        history[-1] = {"role": "assistant", "content": ""}
        yield history, "", api_messages
        try:
            retry_stream = client.chat.completions.create(
                model=model_name,
                messages=retry_messages,
                stream=True,
            )
            for event in retry_stream:
                if len(event.choices) == 0:
                    continue
                stream_content = event.choices[0].delta.content
                if stream_content is not None:
                    buffer += stream_content
                    history[-1] = {"role": "assistant", "content": buffer}
                    yield history, "", api_messages
        except Exception as e:
            history[-1] = {"role": "assistant", "content": buffer + f"\n\nError during retry streaming: {e}"}
            yield history, "", api_messages
            return

        if looks_like_leaked_tool_call(buffer):
            buffer = (
                "The model reached the tool-call limit and could not produce a "
                "final answer. Please rephrase the question or narrow the scope."
            )
            history[-1] = {"role": "assistant", "content": buffer}
            yield history, "", api_messages

    # Append sources footer
    if context_sources:
        sources_text = f"\n\n---\n*Sources: {', '.join(context_sources)}*"
        history[-1] = {"role": "assistant", "content": buffer + sources_text}
        yield history, "", api_messages

    # Record final assistant response in API history
    api_messages = append_to_chat_history('assistant', buffer, api_messages)
    yield history, "", api_messages


def clear_chat():
    """Clear chat history and reset API messages."""
    return [], "", [{'role': 'system', 'content': SYSTEM_MESSAGE}]


# Define theme for Gradio 6.0 (passed to launch())
# Palette: Omarchy "Matte Black" theme (dark background, orange accent)
TERMINAL_THEME = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="gray",
).set(
    body_background_fill="#121212",
    body_background_fill_dark="#121212",
    block_background_fill="#0d0d0d",
    block_background_fill_dark="#0d0d0d",
    body_text_color="#bebebe",
    body_text_color_dark="#bebebe",
    block_label_text_color="#e68e0d",
    block_label_text_color_dark="#e68e0d",
    input_background_fill="#090909",
    input_background_fill_dark="#090909",
    button_primary_background_fill="#e68e0d",
    button_primary_background_fill_dark="#e68e0d",
    button_primary_background_fill_hover="#f59e0b",
    button_primary_background_fill_hover_dark="#f59e0b",
)

# Build the Gradio interface
with gr.Blocks(
    title="RAG Chatbot - Terminal Style",
    fill_height=True,
    theme=TERMINAL_THEME,
    css=TERMINAL_CSS,
) as demo:

    gr.Markdown(
        """
        # RAG-Powered Chatbot
        ### Terminal-Style Interface with Tool Calling
        """,
        elem_classes=["header"],
    )

    # Persistent API message history (survives across turns, holds tool-call entries)
    api_messages_state = gr.State([{'role': 'system', 'content': SYSTEM_MESSAGE}])

    with gr.Row(elem_id="main-row"):
        # Main chat area
        with gr.Column(scale=3, elem_id="chat-column"):
            chatbot = gr.Chatbot(
                label="Chat",
                elem_id="chatbox",
                height=None,
                scale=1,
            )

            with gr.Row(elem_id="chat-input-row"):
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    scale=4,
                    show_label=False,
                    lines=1,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)

        # Settings sidebar
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.Markdown("### Settings")

            with gr.Accordion("API Configuration", open=True):
                api_url = gr.Textbox(
                    label="API URL",
                    value="http://localhost:8080/v1",
                    info="e.g. http://localhost:8080/v1 or https://api.openai.com/v1 " \
                         "or any OpenAI compatible endpoint",
                )
                model_name = gr.Textbox(
                    label="Model Name",
                    value="model.gguf",
                    placeholder="model.gguf",
                )

            with gr.Accordion("Web Search (pre-query)", open=True):
                enable_web_search = gr.Checkbox(
                    label="Enable Web Search",
                    value=False,
                )
                search_engine = gr.Dropdown(
                    label="Search Engine",
                    choices=["tavily", "perplexity"],
                    value="tavily",
                )

            with gr.Accordion("Local RAG", open=True):
                rag_mode = gr.Dropdown(
                    label="RAG Mode",
                    choices=[
                        "Off",
                        "Always-on retrieval",
                        "As tool (model decides)",
                    ],
                    value="Off",
                )
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                pdf_status = gr.Textbox(
                    label="Status",
                    value="No PDF loaded",
                    interactive=False,
                )

            with gr.Accordion("Tool Calling", open=False):
                gr.Markdown(
                    "The assistant **always** has access to tools:\n"
                    "- `search_web` (Tavily / Perplexity)\n"
                    "- `url_search`\n"
                    "- `code_search` (grep)\n"
                    "- `local_rag` (when a PDF is loaded in tool mode)\n\n"
                    "It decides autonomously when to call them."
                )

    # Event handlers
    pdf_upload.change(
        fn=process_pdf,
        inputs=[pdf_upload],
        outputs=[pdf_status],
    )

    chat_inputs = [
        msg, chatbot, api_messages_state, api_url, model_name,
        enable_web_search, search_engine, rag_mode,
    ]
    chat_outputs = [chatbot, msg, api_messages_state]

    submit_btn.click(
        fn=chat,
        inputs=chat_inputs,
        outputs=chat_outputs,
    )

    msg.submit(
        fn=chat,
        inputs=chat_inputs,
        outputs=chat_outputs,
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg, api_messages_state],
    )


if __name__ == "__main__":
    demo.launch(share=False)
