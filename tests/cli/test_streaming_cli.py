import urllib.parse as _url

import pytest

from openagent.cli import _build_ws_url_and_headers, _build_http_headers


def test_build_ws_url_default_path_http_no_auth():
    api_url = "http://localhost:8000"
    ws_path = "/ws/chat"
    ws_url, headers = _build_ws_url_and_headers(api_url, ws_path, api_token=None, token_query_key=None, auth_header_value=None)
    assert ws_url == "ws://localhost:8000/ws/chat"
    assert headers is None


def test_build_ws_url_https_with_auth_header():
    api_url = "https://api.example.com"
    ws_path = "/ws/chat"
    ws_url, headers = _build_ws_url_and_headers(api_url, ws_path, api_token=None, token_query_key=None, auth_header_value="Bearer xyz")
    assert ws_url == "wss://api.example.com/ws/chat"
    assert headers is not None
    # headers as list of tuples
    assert ("Authorization", "Bearer xyz") in headers


def test_build_ws_url_with_query_token_and_custom_path():
    api_url = "http://example.org:8080/base"
    # path can be provided without leading slash and should be normalized
    ws_path = "ws/custom"
    ws_url, headers = _build_ws_url_and_headers(api_url, ws_path, api_token="abc123", token_query_key="token", auth_header_value=None)
    parsed = _url.urlparse(ws_url)
    assert parsed.scheme == "ws"
    assert parsed.netloc == "example.org:8080"
    assert parsed.path == "/ws/custom"
    # token should be in query string
    qs = _url.parse_qs(parsed.query)
    assert qs.get("token") == ["abc123"]
    assert headers is None


def test_build_http_headers_sse_with_auth():
    headers = _build_http_headers(accept_sse=True, auth_header_value="Bearer tkn")
    assert headers.get("Accept") == "text/event-stream"
    assert headers.get("Authorization") == "Bearer tkn"


def test_build_http_headers_non_sse_no_auth():
    headers = _build_http_headers(accept_sse=False, auth_header_value=None)
    assert headers == {}


def test_build_http_headers_non_sse_raw_token():
    # Simulate --auth-scheme "" which results in raw token in Authorization
    headers = _build_http_headers(accept_sse=False, auth_header_value="raw-token-value")
    assert headers.get("Authorization") == "raw-token-value"

