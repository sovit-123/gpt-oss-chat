"""Runs a conversation turn and reports what happens as events.

A turn starts with the user's message and ends with the assistant's
final answer. Along the way the engine may fetch web or document
context, run tool calls the model asks for, and call the model several
times. Everything worth showing is yielded as an event (see
core.events). Tools always run through the registry, and nothing in
this file ever prints."""

import json
from dataclasses import dataclass, field

from core.events import (
    AssistantDelta, Error, Status, ToolCallStarted, ToolResult, TurnFinished,
)
from utils.prompt import build_system_message
from web_search import do_web_search

FINAL_ANSWER_INSTRUCTION = (
    "SYSTEM NOTE: The tool-call limit has been reached and no tools are "
    "available for this reply. Do not call any tools and do not output any "
    "tool-call syntax. Answer the user's original question directly and "
    "completely, synthesizing only from the tool results already provided above."
)

RETRY_ANSWER_INSTRUCTION = (
    "Your previous reply contained a tool call or tool-error text instead of "
    "an answer. No tools are available for this reply. Write your final answer "
    "to the user's original question now, in plain text only, using the tool "
    "results already provided."
)

GIVE_UP_MESSAGE = (
    "The model reached the tool-call limit and could not produce a final "
    "answer. Please rephrase the question or narrow the scope."
)

LEAKED_TOOL_MARKERS = (
    "function=",
    "<function",
    "tried to call unavailable tool",
    "arguments provided to the tool are invalid",
)


@dataclass
class Session:
    """Conversation state: the message history and whether a document is
    loaded for RAG. The frontend owns the Session, the engine is the only
    one that changes it."""
    messages: list = field(default_factory=list)
    rag_ready: bool = False


@dataclass
class EngineConfig:
    """Per-turn settings. Built from CLI flags or sidebar widgets, so the
    values can change between turns of one conversation."""
    model: str
    pre_query_web_search: bool = False
    search_engine: str = "tavily"
    rag_mode: str = "off"          # "off", "always_on", or "as_tool"
    max_tool_calls: int = 5
    max_tool_result_chars: int = 4000


@dataclass
class StreamResult:
    """What one streamed response contained: the text, plus the tool call
    if the response made one."""
    text: str = ""
    tool_name: str | None = None
    tool_id: str | None = None
    tool_args: str = ""


def _looks_leaked(text):
    """True when text is empty or looks like raw tool-call syntax or a
    tool error rather than a real answer. Some models emit tool-call text
    instead of using the protocol, and llama.cpp reports calls to unknown
    tools as ordinary content. The markers are a heuristic, so an answer
    that legitimately contains "function=" could trip it."""
    if not text or not text.strip():
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in LEAKED_TOOL_MARKERS)


class Engine:
    """Runs conversation turns against one OpenAI-compatible endpoint."""

    def __init__(self, client, registry):
        """`client` talks to the model endpoint, `registry` holds the
        tools. Both are constructor arguments so tests can pass fakes."""
        self.client = client
        self.registry = registry

    def run_turn(self, session, user_input, config):
        """Yields the events of one turn, updating the session along the
        way. A failure becomes a final Error event; the session stays
        usable for the next turn."""
        try:
            yield from self._turn(session, user_input, config)
        except Exception as e:
            yield Error(f"API request failed: {e}")

    def _turn(self, session, user_input, config):
        self._sync_system_message(session, config)
        user_input, sources = yield from self._gather_context(user_input, session, config)
        session.messages.append({"role": "user", "content": user_input})
        tools = self.registry.schemas(
            exclude=() if session.rag_ready else ("local_rag",)
        )
        try:
            stream = self._create(config, session.messages, tools)
        except Exception:
            # The turn never started, so the user message should not
            # linger in the history.
            session.messages.pop()
            raise
        answer = yield from self._tool_loop(session, config, tools, stream)
        if answer is None:
            yield Status(f"Reached maximum tool calls ({config.max_tool_calls})")
            answer = yield from self._final_no_tools(session, config)
        if _looks_leaked(answer):
            yield Status("Model returned a tool call instead of an answer; retrying...")
            answer = yield from self._final_no_tools(session, config, leaked_answer=answer)
        session.messages.append({"role": "assistant", "content": answer})
        yield TurnFinished(answer, sources)

    def _gather_context(self, user_input, session, config):
        """Fetches pre-query context (web search, always-on RAG) and wraps
        it into the user message. A failure only drops the context; the
        turn continues without it."""
        results, sources = [], []
        if config.pre_query_web_search:
            try:
                results.extend(do_web_search(user_input, config.search_engine))
                sources.append(f"web search ({config.search_engine})")
            except Exception as e:
                yield Status(f"Web search failed: {e}")
        if config.rag_mode == "always_on" and session.rag_ready:
            # Imported here: semantic_engine pulls in a heavy model stack.
            from semantic_engine import search_query
            try:
                _, passages = search_query(user_input, top_k=3)
                results.extend(passages)
                sources.append("local RAG")
            except Exception as e:
                yield Status(f"Document search failed: {e}")
        if results:
            context = "\n".join(results)
            user_input = (
                "Use the following search results as context to answer the question."
                f"\n\nContext:\n{context}\n\nQuestion: {user_input}"
            )
        if config.rag_mode == "as_tool" and session.rag_ready:
            user_input += " User has passed a document that can be used for local_rag tool"
        return user_input, sources

    def _tool_loop(self, session, config, tools, stream):
        """Cycles between calling the model and running the tool it asked
        for, until the model answers in plain text or the tool budget runs
        out. Returns the answer, or None if the budget was exhausted."""
        count = 0
        cache = {}
        while True:
            result = yield from self._consume(stream)
            if result.tool_name is None:
                return result.text
            count += 1
            try:
                args = json.loads(result.tool_args)
            except json.JSONDecodeError:
                yield Status(
                    f"Could not parse tool arguments for {result.tool_name}: "
                    f"{result.tool_args}"
                )
                return result.text
            yield ToolCallStarted(count, result.tool_name, args)
            key = f"{result.tool_name}::{json.dumps(args, sort_keys=True)}"
            if key in cache:
                output, error = cache[key], False
                yield Status(f"Duplicate tool call skipped: {result.tool_name}")
            else:
                output, error = self.registry.execute(result.tool_name, args)
                if len(output) > config.max_tool_result_chars:
                    output = output[:config.max_tool_result_chars] + "\n...[result truncated]"
                if not error:
                    cache[key] = output
            yield ToolResult(count, result.tool_name, output, error)
            session.messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": result.tool_id, "type": "function",
                    "function": {"name": result.tool_name, "arguments": json.dumps(args)},
                }],
            })
            session.messages.append({
                "role": "tool", "tool_call_id": result.tool_id, "content": output,
            })
            if count >= config.max_tool_calls:
                return None
            yield Status(f"Checking if more tools are needed... ({count}/{config.max_tool_calls})")
            stream = self._create(config, session.messages, tools)

    def _final_no_tools(self, session, config, leaked_answer=None):
        """Streams one last response with no tools offered. Used when the
        tool budget is exhausted, and again if the model still tried to
        call a tool instead of answering. The instruction messages go on a
        copy of the history, so the real conversation stays clean."""
        messages = list(session.messages)
        messages.append({"role": "user", "content": FINAL_ANSWER_INSTRUCTION})
        if leaked_answer is not None:
            messages.append({"role": "assistant", "content": leaked_answer})
            messages.append({"role": "user", "content": RETRY_ANSWER_INSTRUCTION})
        stream = self.client.chat.completions.create(
            model=config.model, messages=messages, stream=True,
        )
        text = (yield from self._consume(stream)).text
        if leaked_answer is not None and _looks_leaked(text):
            return GIVE_UP_MESSAGE
        return text

    def _create(self, config, messages, tools):
        return self.client.chat.completions.create(
            model=config.model, messages=messages, stream=True,
            tools=tools, tool_choice="auto",
        )

    def _consume(self, stream):
        """Reads one streamed response, yielding an AssistantDelta per
        chunk of text. Chunks with no choices (vLLM keep-alives) are
        skipped, and of several parallel tool calls only the first is
        kept."""
        result = StreamResult()
        first_index = None
        seen_parallel = set()
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if delta.tool_calls:
                call = delta.tool_calls[0]
                if first_index is None:
                    first_index = call.index
                elif call.index != first_index:
                    if call.index not in seen_parallel:
                        seen_parallel.add(call.index)
                        yield Status(f"Ignoring parallel tool call at index {call.index}")
                    continue
                if call.id is not None:
                    result.tool_id = call.id
                if call.function and call.function.name is not None:
                    result.tool_name = call.function.name
                if call.function and call.function.arguments:
                    result.tool_args += call.function.arguments
            if delta.content:
                result.text += delta.content
                yield AssistantDelta(delta.content)
        return result

    def _sync_system_message(self, session, config):
        """Keeps the system prompt in sync with the session. Rebuilt every
        turn because a PDF can be uploaded mid-conversation, and the tool
        limit in the text comes from the config so it cannot drift from
        the limit the engine enforces."""
        system = {
            "role": "system",
            "content": build_system_message(
                include_local_rag=session.rag_ready,
                max_tool_calls=config.max_tool_calls,
            ),
        }
        if not session.messages:
            session.messages.append(system)
        else:
            session.messages[0] = system
