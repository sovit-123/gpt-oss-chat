"""Engine tests against a scripted fake model: no server, no network.

The FakeModel plays scripted responses, one per API call, where each
response is a list of stream chunks shaped like real OpenAI stream
events. Everything the engine does is verified through three channels:
the events it yields, the history it leaves in the Session, and the
requests the FakeModel recorded.
"""

import sys
import types
from types import SimpleNamespace as NS

from core.engine import Engine, EngineConfig, Session
from core.events import (
    AssistantDelta, Error, Status, ToolCallStarted, ToolResult, TurnFinished,
)


class FakeStream:
    """Iterates over scripted chunks; the engine reads it exactly like a
    real OpenAI stream object."""
    def __init__(self, chunks):
        self._iter = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class FakeModel:
    """Stand-in for an OpenAI client. `responses` is a list of scripted
    responses, consumed in order, one per create() call. Running out of
    scripts raises, which conveniently doubles as a failure case."""
    def __init__(self, responses):
        self.responses = [list(r) for r in responses]
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, model, messages, stream, tools=None, tool_choice=None):
        self.requests.append({
            "model": model,
            "messages": [dict(m) for m in messages],
            "tools": tools,
        })
        if not self.responses:
            raise RuntimeError("FakeModel: no scripted response left")
        return FakeStream(self.responses.pop(0))


class FakeRegistry:
    """Stand-in for core.registry.ToolRegistry with the same contract:
    schemas() lists available tools, execute() never raises."""
    def __init__(self, impls):
        self.impls = impls
        self.calls = []

    def schemas(self, exclude=()):
        return [{"function": {"name": n}} for n in self.impls if n not in exclude]

    def execute(self, name, args):
        self.calls.append((name, args))
        if name not in self.impls:
            return f"Error: Unknown tool: {name}", True
        try:
            return str(self.impls[name](**args)), False
        except Exception as e:
            return f"Error: {name} failed: {e}", True


def text_chunks(*pieces):
    """One scripted response that streams plain text."""
    return [NS(choices=[NS(delta=NS(content=p, tool_calls=None))]) for p in pieces]


def tool_chunks(name, args, call_id="call-1"):
    """One scripted response that streams a single tool call."""
    call = NS(index=0, id=call_id, function=NS(name=name, arguments=args))
    return [NS(choices=[NS(delta=NS(content=None, tool_calls=[call]))])]


def keepalive():
    """A vLLM-style chunk with no choices at all."""
    return NS(choices=[])


def make_engine(responses, impls):
    """Builds an Engine wired to fakes and hands back all three objects,
    because most tests also want to inspect the registry or the requests
    the model recorded."""
    model = FakeModel(responses)
    registry = FakeRegistry(impls)
    return Engine(model, registry), model, registry


def run(engine, session, text="hi", config=None):
    config = config or EngineConfig(model="m")
    return list(engine.run_turn(session, text, config))


def deltas(events):
    return "".join(e.text for e in events if isinstance(e, AssistantDelta))


def finished(events):
    return next(e for e in events if isinstance(e, TurnFinished))


def of_type(events, kind):
    return [e for e in events if isinstance(e, kind)]


def tool_names(request):
    return [t["function"]["name"] for t in request["tools"]]


# All four tool names present, so visibility tests can observe local_rag
# appearing and disappearing from the request the engine sends.
FOUR_TOOLS = {name: (lambda **kwargs: "ok") for name in
              ("search_web", "local_rag", "url_search", "code_search")}


def test_plain_answer_streams_and_finishes():
    engine, _, _ = make_engine([[keepalive()] + text_chunks("Hello", " world")], {})
    session = Session()
    events = run(engine, session)
    assert deltas(events) == "Hello world"
    assert finished(events).answer == "Hello world"
    assert finished(events).sources == []
    assert [m["role"] for m in session.messages] == ["system", "user", "assistant"]
    assert session.messages[-1]["content"] == "Hello world"


def test_tool_call_executes_and_feeds_result_back():
    engine, model, registry = make_engine(
        [tool_chunks("search_web", '{"topic": "llms", "search_engine": "tavily"}'),
         text_chunks("Done")],
        impls={"search_web": lambda topic, search_engine: f"found:{topic}"},
    )
    session = Session()
    events = run(engine, session)
    started = of_type(events, ToolCallStarted)[0]
    assert (started.number, started.name) == (1, "search_web")
    assert started.args == {"topic": "llms", "search_engine": "tavily"}
    result = of_type(events, ToolResult)[0]
    assert (result.output, result.error) == ("found:llms", False)
    assert registry.calls == [("search_web", {"topic": "llms", "search_engine": "tavily"})]
    assert [m["role"] for m in session.messages] == \
        ["system", "user", "assistant", "tool", "assistant"]
    # The follow-up request the engine made must contain the tool result.
    assert model.requests[1]["messages"][3]["content"] == "found:llms"


def test_duplicate_tool_call_is_served_from_cache():
    call = tool_chunks("search_web", '{"topic": "x", "search_engine": "tavily"}')
    engine, _, registry = make_engine(
        [call, call, text_chunks("final")],
        impls={"search_web": lambda topic, search_engine: "result"},
    )
    events = run(engine, Session())
    assert len(registry.calls) == 1
    assert [r.output for r in of_type(events, ToolResult)] == ["result", "result"]
    assert any("Duplicate" in e.message for e in of_type(events, Status))


def test_budget_exhaustion_forces_answer_without_extra_request():
    # Five distinct calls: identical ones would be served from the dedup
    # cache and never reach the registry, which is a different test.
    calls = [
        tool_chunks("search_web", f'{{"topic": "q{i}", "search_engine": "tavily"}}')
        for i in range(5)
    ]
    engine, model, registry = make_engine(
        calls + [text_chunks("final answer")],
        impls={"search_web": lambda topic, search_engine: "r"},
    )
    session = Session()
    events = run(engine, session)
    assert len(registry.calls) == 5
    assert any("maximum tool calls" in e.message for e in of_type(events, Status))
    forced = model.requests[-1]
    assert forced["tools"] is None
    assert forced["messages"][-1]["content"].startswith("SYSTEM NOTE")
    assert finished(events).answer == "final answer"
    # Exactly six requests: five tool passes plus one forced answer. The
    # old code fired a seventh it never read.
    assert len(model.requests) == 6
    # The forcing instruction must not persist in the real history.
    assert not any(m.get("content", "").startswith("SYSTEM NOTE") for m in session.messages)


def test_leaked_tool_call_text_triggers_one_retry():
    engine, model, _ = make_engine(
        [text_chunks("<function=search_web>{}</function>"),
         text_chunks("clean answer")],
        {},
    )
    session = Session()
    events = run(engine, session)
    assert any("retrying" in e.message for e in of_type(events, Status))
    assert len(model.requests) == 2
    retry = model.requests[-1]
    assert retry["tools"] is None
    assert retry["messages"][-1]["content"].startswith("Your previous reply")
    assert finished(events).answer == "clean answer"
    # Retry scaffolding must not persist in the real history.
    assert [m["role"] for m in session.messages] == ["system", "user", "assistant"]
    assert session.messages[-1]["content"] == "clean answer"


def test_second_leak_yields_fallback_message():
    leak = text_chunks("<function=x>")
    engine, _, _ = make_engine([leak, leak], {})
    events = run(engine, Session())
    assert finished(events).answer.startswith("The model reached the tool-call limit")


def test_pre_query_web_search_injects_context(monkeypatch):
    monkeypatch.setattr(
        "core.engine.do_web_search",
        lambda query, search_engine, max_results=5: ["fact one", "fact two"],
    )
    engine, model, _ = make_engine([text_chunks("ok")], {})
    session = Session()
    config = EngineConfig(model="m", pre_query_web_search=True, search_engine="tavily")
    events = run(engine, session, "what is new?", config)
    sent = model.requests[0]["messages"][1]["content"]
    assert "Context:\nfact one\nfact two" in sent
    assert "Question: what is new?" in sent
    assert finished(events).sources == ["web search (tavily)"]


def test_rag_always_on_uses_document_search(monkeypatch):
    # A fake semantic_engine module keeps the heavy sentence-transformers
    # stack out of the test suite; the engine's lazy import picks it up.
    fake = types.ModuleType("semantic_engine")
    fake.search_query = lambda query, top_k=3: (None, ["doc chunk one"])
    monkeypatch.setitem(sys.modules, "semantic_engine", fake)
    engine, model, _ = make_engine([text_chunks("ok")], {})
    session = Session(rag_ready=True)
    events = run(engine, session, "summarize", EngineConfig(model="m", rag_mode="always_on"))
    assert "doc chunk one" in model.requests[0]["messages"][1]["content"]
    assert finished(events).sources == ["local RAG"]


def test_rag_as_tool_appends_hint():
    engine, model, _ = make_engine([text_chunks("ok")], {})
    session = Session(rag_ready=True)
    run(engine, session, "question", EngineConfig(model="m", rag_mode="as_tool"))
    assert model.requests[0]["messages"][1]["content"].endswith(
        "User has passed a document that can be used for local_rag tool"
    )


def test_local_rag_advertised_only_when_document_loaded():
    engine, model, _ = make_engine([text_chunks("a"), text_chunks("b")], FOUR_TOOLS)
    session = Session()
    run(engine, session)
    assert "local_rag" not in tool_names(model.requests[0])
    session.rag_ready = True
    run(engine, session, "again")
    assert "local_rag" in tool_names(model.requests[1])


def test_first_request_failure_rolls_back_user_message():
    engine, _, _ = make_engine([], {})  # no scripted responses: create() raises
    session = Session()
    events = run(engine, session, "hello")
    assert isinstance(events[-1], Error)
    assert [m["role"] for m in session.messages] == ["system"]


def test_unknown_tool_error_goes_back_to_model():
    engine, _, _ = make_engine(
        [tool_chunks("nope", "{}"), text_chunks("recovered")],
        impls={},
    )
    session = Session()
    events = run(engine, session)
    result = of_type(events, ToolResult)[0]
    assert result.error is True
    assert "Unknown tool" in result.output
    assert session.messages[3]["content"].startswith("Error:")
    assert finished(events).answer == "recovered"


def test_long_tool_results_are_truncated():
    engine, _, _ = make_engine(
        [tool_chunks("search_web", '{"topic": "x", "search_engine": "tavily"}'),
         text_chunks("ok")],
        impls={"search_web": lambda topic, search_engine: "x" * 5000},
    )
    config = EngineConfig(model="m", max_tool_result_chars=100)
    events = run(engine, Session(), config=config)
    output = of_type(events, ToolResult)[0].output
    assert output.endswith("[result truncated]")
    assert len(output) == 100 + len("\n...[result truncated]")


def test_system_message_reflects_rag_availability():
    engine, _, _ = make_engine([text_chunks("a"), text_chunks("b")], {})
    session = Session()
    run(engine, session)
    assert "local_rag" not in session.messages[0]["content"]
    session.rag_ready = True
    run(engine, session, "second")
    assert "local_rag" in session.messages[0]["content"]
