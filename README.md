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

### Advanced Usage in Terminal Mode

**Using local RAG to chat with any PDF file:**

```
python api_call.py --local-rag <path/to/pdf>
```

**Using local RAG as a tool where the LLM assistant decides when to search the document:**

```
python api_call.py --rag-tool <path/to/pdf>
```

***Other than the above, the LLM assistant always has access to web search. According to the user prompt it can decide when to search the web. It can either use Tavily or Perplexity search API based on the user prompt. Default is Tavily web search API. It can call multiple tools at its own discretion.***

e.g. Execute the script. The following shows the **web_search** tool call usage.

```
python api_call.py
```

**Prompt to search the internet to find about the latest information about LLMs:**

```bash
You: Find the latest information to search about LLMs  

Tool call 1: search_web ::: Args: {"topic":"latest information about LLMs","search_engine":"tavily"}
Checking if more tools are needed...
Tool call 2: search_web ::: Args: {"topic":"latest developments in large language models 2026","search_engine":"tavily"}
Checking if more tools are needed...
Tool call 3: search_web ::: Args: {"topic":"latest LLM developments 2026","search_engine":"tavily"}
Checking if more tools are needed...
 Total tools called: 3. Fetching final response...

Assistant: 

Here’s a quick snapshot of the most recent developments in large‑language‑model (LLM) research and product releases (as of early 2026).                                   
Feel free to dive deeper into any of the items below or use them as a starting point for your own searches.                                                                                                                                                                                                                              
  Category                                 Highlights                                                      Where to read more                                             
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  New flagship models                      • GPT‑5 – OpenAI’s next‑generation model with 10 × the          • TechCrunch “OpenAI unveils GPT‑5” (Feb 2026)• Wired          
                                           parameter count of GPT‑4 and new “reversible reasoning”         “Google’s Gemini 3: The multimodal LLM that’s changing the     
                                           layers.• Gemini 3 – Google’s multimodal LLM that blends text,   game” (Mar 2026)• Anthropic blog post on Constitutional AI     
                                           image, and video understanding in a single inference engine.•   (Jan 2026)• Meta AI research page (Feb 2026)                   
                                           Claude 4 – Anthropic’s most aligned model yet, built on a new                                                                  
                                           “Constitutional AI” framework that reduces hallucinations.•                                                                    
                                           Llama 4 – Meta’s open‑source, 70B‑parameter model that                                                                         
                                           introduces a “token‑budget” training technique to keep                    
```

**We can prompt to use the Perplexity web search API instead.**

```
RAG-Powered Chatbot Started
Type 'exit' or 'quit' to end the conversation

You: Find the latest information to search about LLMs. Use the perplexity web search api.
.
.
.
Checking if more tools are needed...
Tool call 2: search_web ::: Args: {"topic":"latest information about LLMs","search_engine":"perplexity"}
Checking if more tools are needed...
```

We can also provide a particular URL and ask it to search. The following uses the **url_search** tool call.

```
You: What does this article contain? https://debuggercafe.com/hunyuan3d-2-0-explanation-and-runpod-docker-image/

Tool call 1: url_search ::: Args: {"url":"https://debuggercafe.com/hunyuan3d-2-0-explanation-and-runpod-docker-image/","search_engine":"tavily"}
Checking if more tools are needed...
 Total tools called: 1. Fetching final response...

Assistant: 
Short answer:                                                                                                                                                             
The article explains the new Hunyuan 3D 2.0 image‑to‑3D pipeline (its architecture, benchmarks, and contributions) and then walks through building a Runpod‑ready Docker  
image so you can run the whole workflow on a cloud GPU with minimal setup.                                                                                                

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                                                   1.  Hunyuan 3D 2.0 – Paper Overview                                                                    


                                                                                                                                            
  Section                      What it covers                                                                                               
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  Motivation                   Why an end‑to‑end, open‑source image‑to‑3D system is needed for gaming, animation, movies, and education.    
  Overall System               Two‑stage pipeline: Shape Generation → Texturing (Inpainting).                                               
  Stage 1 – Shape Generation   • ShapeVAE (image → latent 3D mesh). • Hunyuan3D‑DiT (diffusion model that maps the latent to a full mesh).  
  Stage 2 – Texturing          Hunyuan3D‑Paint produces a texture map for the mesh using the input image.
```

