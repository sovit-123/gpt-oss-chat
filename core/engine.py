"""The chat engine: the single place where one conversation turn happens.

A turn starts when the user sends a message and ends when the assistant
has produced a final answer. Along the way the engine may run web
searches, retrieve passages from an uploaded document, execute tools the
model asks for, and call the model several times. Everything worth
showing the user is yielded as an event from core.events, so the terminal
and the Gradio frontends can render the same conversation each in their
own way.

Two rules hold throughout this file. The engine never touches the
screen: no print(), no Rich, no Gradio, only events. And it never runs a
tool itself, because that is the registry's job. New tools plug into
the registry and this file does not change.
"""

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
    """What survives between turns of one conversation: the OpenAI-format
    message history and whether a document is currently loaded for RAG.
    The frontend owns the Session and hands it to the engine on every
    turn; the engine is the only thing that mutates it."""
    messages: list = field(default_factory=list)
    rag_ready: bool = False


@dataclass
class EngineConfig:
    """Settings for a single turn. A frontend builds one of these from its
    CLI flags or sidebar widgets, so values can change between turns of
    the same conversation, for example when the user edits the model name
    or uploads a document."""
    model: str
    pre_query_web_search: bool = False
    search_engine: str = "tavily"
    rag_mode: str = "off"          # "off", "always_on", or "as_tool"
    max_tool_calls: int = 5
    max_tool_result_chars: int = 4000


@dataclass
class StreamResult:
    """What a single streamed response contained, for the engine's internal
    use: any text the model streamed, plus the name, id and raw JSON
    arguments of a tool call if it made one."""
    text: str = ""
    tool_name: str | None = None
    tool_id: str | None = None
    tool_args: str = ""


def _looks_leaked(text):
    """True when a supposed final answer is actually raw tool-call syntax
    or a server error about one. Small models sometimes emit tool-call
    text instead of using the proper protocol, and llama.cpp wraps calls
    to unknown tools in an error message that arrives as ordinary
    content. The marker matching is a heuristic; it can in principle
    misfire on an answer that legitimately contains the word "function="
    and stays until something better replaces it."""
    if not text or not text.strip():
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in LEAKED_TOOL_MARKERS)


class Engine:
    """Runs conversation turns against one OpenAI-compatible endpoint."""

    def __init__(self, client, registry):
        """`client` is an OpenAI client pointed at whatever endpoint the
        user configured. `registry` supplies the tool schemas and executes
        tool calls. Both are constructor arguments so tests can substitute
        fakes and future frontends can share one engine across sessions."""
        self.client = client
        self.registry = registry

    def run_turn(self, session, user_input, config):
        """Yields the events of one turn and updates the session in place.
        Any failure is converted into a final Error event, after which the
        session is still usable for the next turn."""
        try:
            yield from self._turn(session, user_input, config)
        except Exception as e:
            yield Error(f"API request failed: {e}")

    def _turn(self, session, user_input, config):
        self._sync_system_message(session)
        user_input, sources = yield from self._gather_context(user_input, session, config)
        session.messages.append({"role": "user", "content": user_input})
        tools = self.registry.schemas(
            exclude=() if session.rag_ready else ("local_rag",)
        )
        try:
            stream = self._create(config, session.messages, tools)
        except Exception:
            # The turn never really started, so the user message should
            # not linger in the history. Both old frontends did this.
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
        """Optional pre-query retrieval, decided by the user rather than
        the model. Results are stitched into the user message exactly the
        way the old frontends did, and a failure only costs the context:
        the turn continues without it."""
        results, sources = [], []
        if config.pre_query_web_search:
            try:
                results.extend(do_web_search(user_input, config.search_engine))
                sources.append(f"web search ({config.search_engine})")
            except Exception as e:
                yield Status(f"Web search failed: {e}")
        if config.rag_mode == "always_on" and session.rag_ready:
            # Imported here on purpose: semantic_engine loads a
            # sentence-transformers model at import time, and only turns
            # that actually query the document should pay for that.
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
        """The call-model / run-tool cycle. Each pass consumes one streamed
        response: plain text means the model answered and the loop ends,
        while a tool call goes through the registry, into the history, and
        back to the model for another pass. Returns the final answer, or
        None when the tool budget ran out first."""
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
        """Streams one last response with tools removed from the request.
        Called when the tool budget is exhausted, and a second time when
        the model still tried to call a tool instead of answering. The
        extra instructions are appended to a throwaway copy of the history
        so they never pollute the real conversation."""
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
        """Reads one streamed response, yielding an AssistantDelta for
        every piece of text, and returns what the response contained.
        Keep-alive chunks with no choices are skipped (vLLM sends those),
        and if the model emits several tool calls at once only the first
        is kept, which is what the old frontends did."""
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

    def _sync_system_message(self, session):
        """Keeps the first history entry a system prompt that matches the
        tools actually available. The old Gradio frontend rebuilt this on
        every turn because a PDF can be uploaded mid-conversation; doing
        it in the engine gives both frontends the same behavior."""
        system = {
            "role": "system",
            "content": build_system_message(include_local_rag=session.rag_ready),
        }
        if not session.messages:
            session.messages.append(system)
        else:
            session.messages[0] = system
