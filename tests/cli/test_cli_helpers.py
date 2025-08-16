import pytest

from openagent.cli import _build_ws_url_and_headers, _build_http_headers


def test_ws_url_converts_http_to_ws_and_preserves_host_and_path():
    ws_url, headers = _build_ws_url_and_headers(
        api_url="http://example.com:8080", ws_path="/ws/chat", api_token=None, token_query_key=None, auth_header_value=None
    )
    assert ws_url == "ws://example.com:8080/ws/chat"
    assert headers is None


def test_ws_url_converts_https_to_wss():
    ws_url, _ = _build_ws_url_and_headers(
        api_url="https://api.example.com", ws_path="/ws/chat", api_token=None, token_query_key=None, auth_header_value=None
    )
    assert ws_url.startswith("wss://api.example.com/")


def test_ws_path_normalization_without_leading_slash():
    ws_url, _ = _build_ws_url_and_headers(
        api_url="http://localhost:8000", ws_path="ws/chat", api_token=None, token_query_key=None, auth_header_value=None
    )
    assert ws_url == "ws://localhost:8000/ws/chat"


def test_ws_url_includes_token_as_query_when_requested():
    ws_url, _ = _build_ws_url_and_headers(
        api_url="https://api.example.com", ws_path="/ws/chat", api_token="abc123", token_query_key="token", auth_header_value=None
    )
    assert ws_url == "wss://api.example.com/ws/chat?token=abc123"


def test_ws_extra_headers_include_authorization_when_provided():
    ws_url, extra_headers = _build_ws_url_and_headers(
        api_url="http://localhost:8000", ws_path="/ws/chat", api_token=None, token_query_key=None, auth_header_value="Bearer abc123"
    )
    assert ws_url == "ws://localhost:8000/ws/chat"
    assert extra_headers is not None
    assert ("Authorization", "Bearer abc123") in extra_headers


def test_http_headers_accept_sse_and_authorization():
    headers = _build_http_headers(accept_sse=True, auth_header_value="Bearer tkn")
    assert headers.get("Accept") == "text/event-stream"
    assert headers.get("Authorization") == "Bearer tkn"


def test_http_headers_no_accept_when_not_sse_and_raw_token():
    headers = _build_http_headers(accept_sse=False, auth_header_value="rawtoken")
    assert "Accept" not in headers
    assert headers.get("Authorization") == "rawtoken"

