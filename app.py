"""
Gradio Web UI for RAG-powered chatbot.
Styled to look like Rich console output with dark theme, syntax highlighting,
and formatted tables.
"""

import json
import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from core.engine import Engine, EngineConfig, Session
from core.events import (
    AssistantDelta, Error, Status, ToolCallStarted, TurnFinished,
)
from core.registry import default_registry

load_dotenv()

API_KEY = os.getenv("MODAL_API_KEY", "default")

# The dropdown shows human labels; the engine wants the machine names.
RAG_MODES = {
    "Off": "off",
    "Always-on retrieval": "always_on",
    "As tool (model decides)": "as_tool",
}

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


def process_pdf(file, session):
    """Reads the uploaded PDF into the Qdrant collection and marks the
    session RAG-ready, which unlocks the local_rag tool. semantic_engine
    is imported here to keep startup light."""
    if file is None:
        return "No file uploaded", session
    try:
        from semantic_engine import (
            read_pdf, chunk_text, create_and_upload_in_mem_collection,
        )
        documents = chunk_text(read_pdf(file), chunk_size=512, overlap=50)
        create_and_upload_in_mem_collection(documents=documents)
        session.rag_ready = True
        return f"PDF processed: {len(documents)} chunks created", session
    except Exception as e:
        session.rag_ready = False
        return f"Error processing PDF: {e}", session


def chat(message, history, session, api_url, model_name,
         enable_web_search, search_engine, rag_mode):
    """Runs one turn and appends the engine's events to the chat as they
    arrive. `history` is display only; the real conversation lives in the
    session."""
    if not message.strip():
        yield history, "", session
        return

    # The client is rebuilt every turn because the URL and model widgets
    # can change. The 120s timeout replaces the SDK's ten-minute default.
    client = OpenAI(base_url=api_url, api_key=API_KEY, timeout=120.0)
    engine = Engine(client, default_registry())
    config = EngineConfig(
        model=model_name,
        pre_query_web_search=enable_web_search,
        search_engine=search_engine,
        rag_mode=RAG_MODES[rag_mode],
    )

    history.append({"role": "user", "content": message})
    yield history, "", session

    bubble = None  # index of the streaming assistant bubble, if there is one

    for event in engine.run_turn(session, message, config):
        if isinstance(event, AssistantDelta):
            if bubble is None:
                history.append({"role": "assistant", "content": ""})
                bubble = len(history) - 1
            history[bubble]["content"] += event.text
            yield history, "", session
        elif isinstance(event, ToolCallStarted):
            # Text streamed before the call stays behind as its own bubble.
            bubble = None
            history.append({
                "role": "assistant",
                "content": f"**Tool call {event.number}: {event.name}**\n"
                           f"Args: `{json.dumps(event.args)}`",
            })
            yield history, "", session
        elif isinstance(event, Status):
            history.append({"role": "assistant", "content": f"*{event.message}*"})
            yield history, "", session
        elif isinstance(event, TurnFinished):
            if bubble is None:
                history.append({"role": "assistant", "content": ""})
                bubble = len(history) - 1
            content = event.answer
            if event.sources:
                content += f"\n\n---\n*Sources: {', '.join(event.sources)}*"
            # Replace with the canonical answer; on the leaked-text retry
            # path this drops the garbage that streamed.
            history[bubble] = {"role": "assistant", "content": content}
            yield history, "", session
        elif isinstance(event, Error):
            history.append({"role": "assistant", "content": f"**Error:** {event.message}"})
            yield history, "", session
        # Tool results are not displayed; the model consumes them, not
        # the user.


def clear_chat():
    """Clears the display and starts a fresh session."""
    return [], "", Session()


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

    # The Session survives across turns and holds the real conversation.
    session_state = gr.State(Session())

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
        inputs=[pdf_upload, session_state],
        outputs=[pdf_status, session_state],
    )

    chat_inputs = [
        msg, chatbot, session_state, api_url, model_name,
        enable_web_search, search_engine, rag_mode,
    ]
    chat_outputs = [chatbot, msg, session_state]

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
        outputs=[chatbot, msg, session_state],
    )


if __name__ == "__main__":
    demo.launch(share=False)
