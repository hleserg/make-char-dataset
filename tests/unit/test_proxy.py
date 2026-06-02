"""Tests for make_char_dataset.proxy (Gemini proxy client).

Only the pure payload parser + the URL guard are covered; the live ``caption``
HTTP/SSE call is ``# pragma: no cover`` (network).
"""

from __future__ import annotations

import json

import pytest

from make_char_dataset.proxy import GeminiProxyClient, parse_proxy_payload


def test_parse_proxy_payload_double_encoded() -> None:
    # Gradio wraps outputs in a list; our Space returns a JSON string inside it.
    inner = json.dumps({"caption": "kael_char, standing", "in_tokens": 5, "out_tokens": 9})
    assert parse_proxy_payload(json.dumps([inner])) == {
        "caption": "kael_char, standing",
        "in_tokens": 5,
        "out_tokens": 9,
    }


def test_parse_proxy_payload_list_of_dict() -> None:
    assert parse_proxy_payload(json.dumps([{"caption": "x"}])) == {"caption": "x"}


def test_parse_proxy_payload_empty() -> None:
    assert parse_proxy_payload(None) == {"error": "empty_response"}
    assert parse_proxy_payload("") == {"error": "empty_response"}


def test_parse_proxy_payload_malformed() -> None:
    result = parse_proxy_payload("{ not json")
    assert result["error"] == "parse_failed"
    assert "detail" in result


def test_client_rejects_non_https() -> None:
    with pytest.raises(ValueError, match="https://"):
        GeminiProxyClient("tok", url="http://insecure.example/call/caption")


def test_client_accepts_https() -> None:
    client = GeminiProxyClient("tok", url="https://example.hf.space/gradio_api/call/caption")
    assert isinstance(client, GeminiProxyClient)
