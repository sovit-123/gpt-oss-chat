"""
Gradio Web UI for RAG-powered chatbot with terminal-style aesthetics.
Styled to look like Rich console output with dark theme, syntax highlighting,
and formatted tables.
"""

import gradio as gr
from openai import OpenAI, APIError
from web_search import do_web_search
from semantic_engine import (
    read_pdf,
    chunk_text,
    create_and_upload_in_mem_collection,
    search_query
)

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

# System message
SYSTEM_MESSAGE = """
You are a helpful assistant. You never say you are an OpenAI model or chatGPT.
You are here to help the user with their requests.
When the user asks who are you, you say that you are a helpful AI assistant.
"""

# Global state for RAG collection
rag_ready = False
pdf_path = None


def process_pdf(file):
    """Process uploaded PDF for RAG."""
    global rag_ready, pdf_path
    
    if file is None:
        return "⚠️ No file uploaded"
    
    try:
        pdf_path = file.name
        full_text = read_pdf(pdf_path)
        documents = chunk_text(full_text, chunk_size=512, overlap=50)
        create_and_upload_in_mem_collection(documents=documents)
        rag_ready = True
        return f"✓ PDF processed: {len(documents)} chunks created"
    except Exception as e:
        rag_ready = False
        return f"✗ Error: {str(e)}"


def chat(message, history, api_url, model_name, enable_web_search, search_engine, enable_local_rag):
    """Main chat function with streaming."""
    global rag_ready
    
    if not message.strip():
        yield history, ""
        return
    
    # Initialize OpenAI client
    try:
        client = OpenAI(base_url=api_url, api_key='')
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error initializing client: {str(e)}"})
        yield history, ""
        return
    
    # Build messages from history (Gradio 6.0 format: list of dicts with role/content)
    messages = [{'role': 'system', 'content': SYSTEM_MESSAGE}]
    for msg in history:
        messages.append({'role': msg['role'], 'content': msg['content']})
    
    # Gather context from search sources
    search_results = []
    context_sources = []
    user_input = message
    
    # Web search
    if enable_web_search:
        try:
            web_results = do_web_search(query=message, search_engine=search_engine)
            search_results.extend(web_results)
            context_sources.append(f"🌐 {search_engine}")
        except Exception as e:
            pass  # Silently fail web search
    
    # Local RAG search
    if enable_local_rag and rag_ready:
        try:
            hits, local_results = search_query(message, top_k=3)
            search_results.extend(local_results)
            context_sources.append("📄 local RAG")
        except Exception as e:
            pass  # Silently fail RAG search
    
    # Add context to user input
    if search_results:
        context = "\n".join(search_results)
        user_input = f"Use the following search results as context to answer the question.\n\nContext:\n{context}\n\nQuestion: {message}"
    
    messages.append({'role': 'user', 'content': user_input})
    
    # Add user message to history (Gradio 6.0 format)
    history.append({"role": "user", "content": message})
    
    # Stream response
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )
        
        current_response = ""
        # Add placeholder for assistant response
        history.append({"role": "assistant", "content": ""})
        
        for event in stream:
            content = event.choices[0].delta.content
            if content:
                current_response += content
                history[-1] = {"role": "assistant", "content": current_response}
                yield history, ""
        
        # Add sources footer
        if context_sources:
            sources_text = f"\n\n---\n*Sources: {', '.join(context_sources)}*"
            history[-1] = {"role": "assistant", "content": current_response + sources_text}
            yield history, ""
            
    except APIError as e:
        history.append({"role": "assistant", "content": f"❌ API Error: {str(e)}"})
        yield history, ""
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        yield history, ""


def clear_chat():
    """Clear chat history."""
    return [], ""


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
        # 🖥️ RAG-Powered Chatbot
        ### Terminal-Style Interface
        """,
        elem_classes=["header"]
    )
    
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
            
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
        
        # Settings sidebar
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            with gr.Accordion("🔌 API Configuration", open=True):
                api_url = gr.Textbox(
                    label="API URL",
                    value="http://localhost:8080/v1",
                    placeholder="http://localhost:8080/v1"
                )
                model_name = gr.Textbox(
                    label="Model Name",
                    value="model.gguf",
                    placeholder="model.gguf"
                )
            
            with gr.Accordion("🌐 Web Search", open=True):
                enable_web_search = gr.Checkbox(
                    label="Enable Web Search",
                    value=False
                )
                search_engine = gr.Dropdown(
                    label="Search Engine",
                    choices=["tavily", "perplexity"],
                    value="tavily"
                )
            
            with gr.Accordion("📄 Local RAG", open=True):
                enable_local_rag = gr.Checkbox(
                    label="Enable Local RAG",
                    value=False
                )
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                pdf_status = gr.Textbox(
                    label="Status",
                    value="No PDF loaded",
                    interactive=False
                )
    
    # Event handlers
    pdf_upload.change(
        fn=process_pdf,
        inputs=[pdf_upload],
        outputs=[pdf_status]
    )
    
    submit_btn.click(
        fn=chat,
        inputs=[msg, chatbot, api_url, model_name, enable_web_search, search_engine, enable_local_rag],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, api_url, model_name, enable_web_search, search_engine, enable_local_rag],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg]
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=TERMINAL_THEME, css=TERMINAL_CSS)
