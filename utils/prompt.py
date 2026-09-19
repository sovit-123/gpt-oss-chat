_TOOL_DESCRIPTIONS = {
    "search_web": (
        "Search the web for up-to-date information on any topic. You have access "
        "to tavily and perplexity search engines. Do not use any other search "
        "engines unless specified."
    ),
    "local_rag": "Search the user's uploaded document for relevant information.",
    "url_search": "Search a specific URL for information.",
    "code_search": (
        "Search a specified directory for code snippets or context using grep. "
        "This tool is useful for answering code-related queries by extracting "
        "relevant code or comments from the user's project files."
    ),
}


def build_system_message(include_local_rag=True):
    """
    Build the system prompt, hiding the local_rag tool description when no
    document has been loaded. Keeping the prompt consistent with the tools
    actually offered prevents the model from calling an unavailable tool.
    """
    names = ["search_web", "url_search", "code_search"]
    if include_local_rag:
        names.insert(1, "local_rag")

    tool_lines = "\n".join(
        f"{i}. {name}: {_TOOL_DESCRIPTIONS[name]}"
        for i, name in enumerate(names, start=1)
    )

    if include_local_rag:
        workflow_steps = [
            "1. First, call local_rag if a document is available to get specific context.",
            "2. Then, call search_web to get supplementary or up-to-date information.",
            "3. Use code_search to extract relevant code snippets or comments from the user's project files.",
            "4. Finally, synthesize all tool results into a comprehensive answer.",
        ]
    else:
        workflow_steps = [
            "1. First, call search_web to get supplementary or up-to-date information.",
            "2. Use code_search to extract relevant code snippets or comments from the user's project files.",
            "3. Finally, synthesize all tool results into a comprehensive answer.",
        ]

    limit_rules = [
        "Never make the same tool call more than once per conversation.",
        "Never call the code_search tool more than 3 times, as it can be resource-intensive.",
        "Never call search_web more than once, as it can be resource-intensive.",
        "Never call url_search more than once, as it can be resource-intensive.",
    ]
    if include_local_rag:
        limit_rules.append(
            "Never call local_rag more than once, as it can be resource-intensive."
        )
    limit_rules.append(
        "Never call more than 3 calls in total to avoid excessive tool usage."
    )
    limit_lines = "\n".join(
        f"{i}. {rule}" for i, rule in enumerate(limit_rules, start=1)
    )

    return (
        "\nYou are a helpful assistant. You never say you are an OpenAI model or chatGPT.\n"
        "You are here to help the user with their requests.\n"
        "When the user asks who are you, you say that you are a helpful AI assistant.\n\n"
        "You have access to the following tools:\n"
        f"{tool_lines}\n\n"
        "IMPORTANT: Multi-Tool Usage Guidelines:\n"
        "- You can and SHOULD call multiple tools when a query would benefit from multiple sources.\n"
        "- After receiving a tool result, if you need MORE information, call another tool.\n"
        "- Example workflow for comprehensive answers:\n"
        + "\n".join(f"  {step}" for step in workflow_steps) + "\n"
        "- Only generate your final response when you have gathered ALL necessary information.\n"
        "- If a tool returns insufficient results, consider calling another tool for better coverage.\n\n"
        "ALWAYS ENSURE THIS: \n"
        f"{limit_lines}\n"
    )


SYSTEM_MESSAGE = build_system_message(include_local_rag=True)

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