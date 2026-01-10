# gpt-oss-chat

A simple local RAG + web search pipeline powered by gpt-oss-20b via llama.cpp and similar scale models like NVIDIA Nemotron 3 Nano 30B A3B.

**Terminal chat powered by Rich Console UI**

![](assets/gpt-oss-chat-terminal.png) 

**Gradio chat with a terminal theme**

![](assets/gpt-oss-chat-ui.png)

## Setup Steps

* Install llama.cpp with CUDA

* Install Qdrant docker (optional). Qdrant Python client is mandatory for in memory vector DB and RAG

* Run `pip install -r requirements.txt`

* Create a `.env` file and add the [Tavily](https://www.tavily.com/) API key for web search. Optionally, you can also add the [Perplexity API key](https://docs.perplexity.ai/guides/search-quickstart).

  ```
  TAVILY_API_KEY=YOUR_TAVILY_API_KEY
  PERPLEXITY_API_KEY=YOUR_PERPLEXITY_API_KEY
  ```

## Running

**Start the llama.cpp server:**

```
./build/bin/llama-server -hf ggml-org/gpt-oss-20b-GGUF
```

**Terminal chat**

```
python api_call.py
```

**OR Gradio UI**

```
python app.py
```

