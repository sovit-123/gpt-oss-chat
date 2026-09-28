"""The chat engine yields these events while it works through a user's
message. Each one describes a single thing that happened, like a piece of
streamed text, a tool call, or the finished answer. Events are plain data
with no display logic in them: deciding how they look on screen is
entirely the job of the frontend that receives them."""

from dataclasses import dataclass


@dataclass
class AssistantDelta:
    """A piece of streamed assistant text. It may be a preamble the model
    writes before calling a tool, or the final answer itself."""
    text: str


@dataclass
class ToolCallStarted:
    """The model wants to run a tool. `number` counts tool calls within
    the current turn, starting at 1, so a frontend can print things like
    "Tool call 2 of 5"."""
    number: int
    name: str
    args: dict


@dataclass
class ToolResult:
    """The result of running a tool. The chat frontends do not show this
    directly, but tests read it to check what the engine did, and a future
    API can send it to clients unchanged."""
    number: int
    name: str
    output: str
    error: bool


@dataclass
class Status:
    """A progress or warning message meant for the user to read, such as
    "Checking if more tools are needed (2/5)"."""
    message: str


@dataclass
class TurnFinished:
    """The complete final answer for the turn, yielded once at the end.
    `sources` lists where any pre-fetched context came from, for example
    "web search (tavily)" or "local RAG"."""
    answer: str
    sources: list


@dataclass
class Error:
    """Something went wrong and the turn ended early. The session is still
    usable afterwards, so the frontend can show the error and let the user
    try again."""
    message: str
