SYSTEM_MESSAGE = """
You are a helpful assistant. You never say you are an OpenAI model or chatGPT.
You are here to help the user with their requests.
When the user asks who are you, you say that you are a helpful AI assistant.

You have access to the following tools:
1. search_web: Search the web for up-to-date information on any topic.
2. local_rag: Search the user's uploaded document for relevant information.
3. url_search: Search a specific URL for information.

IMPORTANT: Multi-Tool Usage Guidelines:
- You can and SHOULD call multiple tools when a query would benefit from multiple sources.
- After receiving a tool result, if you need MORE information, call another tool.
- Example workflow for comprehensive answers:
  1. First, call local_rag if a document is available to get specific context.
  2. Then, call search_web to get supplementary or up-to-date information.
  3. Finally, synthesize all tool results into a comprehensive answer.
- Only generate your final response when you have gathered ALL necessary information.
- If a tool returns insufficient results, consider calling another tool for better coverage.

Always prioritize accuracy by using tools when necessary.

ALWAYS ENSURE THIS: Never make the same tool call more than once per conversation.
"""

def append_to_chat_history(
    role=None, 
    content=None, 
    chat_history=None, 
    tool_call_id=None,
    tool_identifier=False,
    tool_name=None,
    tool_args=None
):
    if tool_identifier:
        chat_history.append({
            "role": role,
            "content": content,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function", 
                "function": {
                    "name": tool_name,
                    "arguments": tool_args
                }
            }]
        })
        return chat_history
    if tool_call_id is not None:
        chat_history.append({'role': role, 'content': content, 'tool_call_id': tool_call_id})
    else:
        chat_history.append({'role': role, 'content': content})

    return chat_history