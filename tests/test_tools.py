"""Tests for the real registry and the tools that can run offline. The
web-search tools need network and keys, so for those only the calling
contract (defaults, schema requirements) is checked."""

import inspect

from core.registry import default_registry
from tools.tools import code_search, local_rag, search_web, url_search


def test_unknown_tool_reports_error():
    output, error = default_registry().execute("nope", {})
    assert error is True
    assert "Unknown tool" in output


def test_bad_arguments_report_error():
    output, error = default_registry().execute("code_search", {"bogus": 1})
    assert error is True


def test_raised_exceptions_become_errors(tmp_path):
    output, error = default_registry().execute(
        "code_search", {"directory": str(tmp_path / "missing"), "query": "x"}
    )
    assert error is True
    assert "code_search failed" in output


def test_code_search_finds_matches(tmp_path):
    (tmp_path / "sample.py").write_text("def hello():\n    pass\n")
    output, error = default_registry().execute(
        "code_search", {"directory": str(tmp_path), "query": "hello"}
    )
    assert error is False
    assert "def hello():" in output


def test_optional_arguments_have_defaults():
    assert inspect.signature(search_web).parameters["search_engine"].default == "tavily"
    assert inspect.signature(url_search).parameters["search_engine"].default == "tavily"
    assert inspect.signature(local_rag).parameters["top_k"].default == 3
    assert inspect.signature(code_search).parameters["max_results"].default == 10


def test_optional_arguments_are_not_required_in_schemas():
    schemas = {
        s["function"]["name"]: s["function"]["parameters"]
        for s in default_registry().schemas()
    }
    assert schemas["search_web"]["required"] == ["topic"]
    assert schemas["url_search"]["required"] == ["url"]
    assert schemas["local_rag"]["required"] == ["topic"]
    assert schemas["code_search"]["required"] == ["directory", "query"]
