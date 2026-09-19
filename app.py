"""
Gradio Web UI for RAG-powered chatbot with terminal-style aesthetics.
Styled to look like Rich console output with dark theme, syntax highlighting,
and formatted tables.
"""

import json

import gradio as gr
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

MAX_TOOL_CALLS = 5
MAX_TOOL_RESULT_CHARS = 4000

# Terminal-style CSS to mimic Rich console
TERMINAL_CSS = """
/* Import monospace font */
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&display=swap');

/* Main container styling */
.gradio-container {
    background-color: #300a24 !important;
    font-family: 'Fira Code', 'Ubuntu Mono', 'Consolas', 'Monaco', monospace !important;
}

/* Chat container */
.chatbot {
    background-color: #300a24 !important;
    border: 1px solid #5c3566 !important;
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
    background-color: #3b0f2b !important;
    border-left: 3px solid #8be9fd !important;
}

/* Bot messages */
[data-testid="bot"] {
    background-color: #2d0922 !important;
    border-left: 3px solid #50fa7b !important;
}

/* Code blocks - terminal style with proper syntax highlighting */
pre {
    background-color: #1a0514 !important;
    border: 1px solid #5c3566 !important;
    border-radius: 6px !important;
    padding: 16px !important;
    font-family: 'Fira Code', 'Ubuntu Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    overflow-x: auto !important;
    color: #f8f8f2 !important;
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
    background-color: #3b0f2b !important;
    border: 1px solid #5c3566 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.9em !important;
    color: #50fa7b !important;
    text-decoration: none !important;
}

/* Remove any strikethrough effects */
pre *, code *, pre, code {
    text-decoration: none !important;
    text-decoration-line: none !important;
}

/* Syntax highlighting - Dracula theme colors */
.hljs-keyword, .token.keyword { color: #ff79c6 !important; }
.hljs-built_in, .token.builtin { color: #8be9fd !important; }
.hljs-type, .token.class-name { color: #8be9fd !important; }
.hljs-literal, .token.boolean { color: #bd93f9 !important; }
.hljs-number, .token.number { color: #bd93f9 !important; }
.hljs-string, .token.string { color: #f1fa8c !important; }
.hljs-comment, .token.comment { color: #6272a4 !important; font-style: italic; }
.hljs-function, .token.function { color: #50fa7b !important; }
.hljs-params { color: #ffb86c !important; }
.hljs-attr, .token.attr-name { color: #50fa7b !important; }
.hljs-variable, .token.variable { color: #f8f8f2 !important; }
.hljs-punctuation, .token.punctuation { color: #f8f8f2 !important; }
.hljs-operator, .token.operator { color: #ff79c6 !important; }

/* Tables styling */
table {
    border-collapse: collapse !important;
    background-color: #1a0514 !important;
    border: 1px solid #5c3566 !important;
    margin: 10px 0 !important;
    width: 100% !important;
}

th {
    background-color: #3b0f2b !important;
    color: #50fa7b !important;
    padding: 10px 14px !important;
    border: 1px solid #5c3566 !important;
    font-weight: 600 !important;
    text-align: left !important;
}

td {
    padding: 8px 14px !important;
    border: 1px solid #5c3566 !important;
    color: #e0e0e0 !important;
}

tr:nth-child(even) {
    background-color: #2d0922 !important;
}

/* Input textbox */
textarea, input[type="text"] {
    background-color: #1a0514 !important;
    color: #ffffff !important;
    border: 1px solid #5c3566 !important;
    font-family: 'Fira Code', 'Ubuntu Mono', monospace !important;
}

/* Buttons */
button.primary, .primary {
    background-color: #5c3566 !important;
    color: #ffffff !important;
    border: none !important;
}

button.primary:hover, .primary:hover {
    background-color: #7c4d8a !important;
}

/* Labels and text */
label, .label-text, span {
    color: #e0e0e0 !important;
    font-family: 'Fira Code', monospace !important;
}

/* Accordion headers */
.accordion {
    background-color: #2d0922 !important;
    border: 1px solid #5c3566 !important;
}

/* File upload */
.file-upload {
    background-color: #1a0514 !important;
    border: 2px dashed #5c3566 !important;
}

/* Checkboxes */
input[type="checkbox"] {
    accent-color: #50fa7b !important;
}

/* Markdown list styling */
ul, ol {
    color: #e0e0e0 !important;
}

li::marker {
    color: #ffb86c !important;
}

/* Bold text */
strong, b {
    color: #ff79c6 !important;
    font-weight: 600 !important;
}

/* Italic text */
em, i {
    color: #bd93f9 !important;
}

/* Links */
a {
    color: #8be9fd !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
    background-color: #300a24;
}

::-webkit-scrollbar-thumb {
    background-color: #5c3566;
    border-radius: 4px;
}

/* Header styling */
h1, h2, h3, h4 {
    color: #50fa7b !important;
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
        client = OpenAI(base_url=api_url, api_key='default')
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
    dangling_stream_content = ''

    while tool_call_count < MAX_TOOL_CALLS:
        tool_args_str = ''
        tool_name = None
        tool_id = None

        for event in stream:
            if event.choices[0].delta.tool_calls is not None:
                if event.choices[0].delta.tool_calls[0].function.name is not None:
                    tool_name = event.choices[0].delta.tool_calls[0].function.name
                    tool_id = event.choices[0].delta.tool_calls[0].id
                tool_args_str += event.choices[0].delta.tool_calls[0].function.arguments
            if event.choices[0].delta.content is not None:
                dangling_stream_content = event.choices[0].delta.content
                break

        if tool_name is None:
            break

        tool_call_count += 1

        # Show tool-call status in the UI
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
        dangling_stream_content = ''
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
    elif tool_call_count > 0:
        history.append({
            "role": "assistant",
            "content": f"*Total tools called: {tool_call_count}. Fetching final response...*"
        })
        yield history, "", api_messages

    # Stream the final assistant response
    buffer = dangling_stream_content
    history.append({"role": "assistant", "content": buffer})
    yield history, "", api_messages

    try:
        for event in stream:
            stream_content = event.choices[0].delta.content
            if stream_content is not None:
                buffer += stream_content
                history[-1] = {"role": "assistant", "content": buffer}
                yield history, "", api_messages
    except Exception as e:
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
TERMINAL_THEME = gr.themes.Base(
    primary_hue="purple",
    neutral_hue="gray",
).set(
    body_background_fill="#300a24",
    body_background_fill_dark="#300a24",
    block_background_fill="#2d0922",
    block_background_fill_dark="#2d0922",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
    block_label_text_color="#50fa7b",
    block_label_text_color_dark="#50fa7b",
    input_background_fill="#1a0514",
    input_background_fill_dark="#1a0514",
    button_primary_background_fill="#5c3566",
    button_primary_background_fill_dark="#5c3566",
    button_primary_background_fill_hover="#7c4d8a",
    button_primary_background_fill_hover_dark="#7c4d8a",
)

# Build the Gradio interface
with gr.Blocks(title="RAG Chatbot - Terminal Style") as demo:

    gr.Markdown(
        """
        # RAG-Powered Chatbot
        ### Terminal-Style Interface with Tool Calling
        """,
        elem_classes=["header"],
    )

    # Persistent API message history (survives across turns, holds tool-call entries)
    api_messages_state = gr.State([{'role': 'system', 'content': SYSTEM_MESSAGE}])

    with gr.Row():
        # Main chat area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="You:",
                    placeholder="Type your message here...",
                    scale=4,
                    show_label=True,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("Clear Chat", variant="secondary")

        # Settings sidebar
        with gr.Column(scale=1):
            gr.Markdown("### Settings")

            with gr.Accordion("API Configuration", open=True):
                api_url = gr.Textbox(
                    label="API URL",
                    value="http://localhost:8080/v1",
                    placeholder="http://localhost:8080/v1",
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
    demo.launch(share=False, theme=TERMINAL_THEME, css=TERMINAL_CSS)
