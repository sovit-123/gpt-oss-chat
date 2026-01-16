"""
Verfication script to simulate tool use with gpt-oss model.
Mostly for debugging purpose.

USAGE:
python -m tools.tool_simulation
"""


from openai import OpenAI
from web_search import do_web_search

import json

def search_web(topic: str) -> str:
    result =  do_web_search(
        topic, 
        search_engine='perplexity'
    )
    return '\n'.join(result)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a given topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"]
            },
        },
    }
]

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="llama.cpp"
)

user_prompt = """Which day of the week is it in India now?"""
messages = [{"role": "user", "content": user_prompt}]

# Need to process tool in loop as the model does not call the tool on its
# own. It selects the tool, we need to make manual call based on selection
# and append the context for next turn.
# The loop should break after one turn as the next turn does not select
# any tool.
while True:
    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )

    # The loop should break after one turn as the next turn does not select
    # any tool.
    if not response.choices[0].message.tool_calls:
        print('Assistant:', response.choices[0].message.content)
        break

    messages.append(response.choices[0].message)

    print('Messages after first call:\n')
    print(messages)

    # Process tool calls.
    for tool_call in response.choices[0].message.tool_calls:
        tool_name = tool_call.function.name
        print(f"Using tool: {tool_name}")
        tool_args = json.loads(tool_call.function.arguments)

        # Execute tool call.
        if tool_name == 'search_web':
            result = search_web(**tool_args)

        # Add tool call to conversation.
        messages.append({
            'role': 'tool',
            'tool_call_id': tool_call.id,
            'content': str(result)
        })