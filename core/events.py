"""Events emitted by the engine during a turn.

Each event describes one thing that happened: a chunk of streamed text,
a tool call, the final answer, and so on. They carry data only; the
frontends decide how to show them."""

from dataclasses import dataclass


@dataclass
class AssistantDelta:
    """Streamed assistant text. Either the final answer, or a preamble
    the model writes before calling a tool."""
    text: str


@dataclass
class ToolCallStarted:
    """The model requested a tool. `number` is the 1-based count of tool
    calls within this turn."""
    number: int
    name: str
    args: dict


@dataclass
class ToolResult:
    """Result of a tool call. The frontends do not display it, but tests
    read it and an API client would want it."""
    number: int
    name: str
    output: str
    error: bool


@dataclass
class Status:
    """A progress or warning message shown to the user, e.g. "Checking
    if more tools are needed (2/5)"."""
    message: str


@dataclass
class TurnFinished:
    """The final answer, yielded once at the end of the turn. `sources`
    names where the context came from, e.g. "web search (tavily)"."""
    answer: str
    sources: list


@dataclass
class Error:
    """The turn failed. The session is still usable for the next turn."""
    message: str
