# README

A simple local RAG + web search pipeline powered by gpt-oss-20b via llama.cpp and similar scale models like NVIDIA Nemotron 3 Nano 30B A3B.

**Terminal chat powered by Rich Console UI**

![](assets/gpt-oss-chat-terminal.png) 

**Gradio chat with a terminal theme**

![](assets/gpt-oss-chat-ui.png)

## Setup Steps

* Install llama.cpp with CUDA
* Install Qdrant docker (optional). Qdrant Python client is mandatory for in memory vector DB and RAG
* Run `pip install -r requirements.txt`

## Running

**Terminal chat**

```
python api_call.py
```

**Gradio UI**

```
python app.py
```

