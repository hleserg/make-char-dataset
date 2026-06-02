"""Client for the Gemini API proxy HF Space.

This box's region is geo-blocked by Gemini (``400 "User location is not
supported"``), so **every** Gemini call goes through the private HF Space
``hleserg/proxy_gemini_api``, which relays to ``generateContent`` from a
supported region. The network I/O (stdlib ``urllib``) is hidden behind the
:class:`CaptionClient` Protocol so the VLM captioner is unit-tested with a fake —
the same pure-core / lazy-backend split used across the stages.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from typing import Protocol

#: Default Gradio endpoint of the proxy Space (overridable via ``APP_VLM_PROXY_URL``).
PROXY_CAPTION_URL = "https://hleserg-proxy-gemini-api.hf.space/gradio_api/call/caption"


class CaptionClient(Protocol):
    """Returns a Gemini caption result dict for ``(model, prompt, image_bytes)``."""

    def caption(self, model: str, prompt: str, image_bytes: bytes) -> dict[str, object]:
        """Return ``{caption, in_tokens, out_tokens, latency_s, ...}`` or ``{error: ...}``."""
        ...


def parse_proxy_payload(last_data: str | None) -> dict[str, object]:
    """Parse the proxy's final SSE ``data:`` line into the inner result dict.

    Gradio wraps outputs in a JSON list and our Space returns a JSON string, so the
    payload is doubly-encoded: ``json.loads(json.loads(line)[0])``. Returns an
    ``{"error": ...}`` dict rather than raising, so a batch keeps going.
    """
    if not last_data:
        return {"error": "empty_response"}
    try:
        outer = json.loads(last_data)
        inner = outer[0] if isinstance(outer, list) and outer else outer
        return json.loads(inner) if isinstance(inner, str) else dict(inner)
    except Exception as exc:
        return {"error": "parse_failed", "detail": str(exc)[:160]}


class GeminiProxyClient:
    """Calls the proxy Space over HTTP (two-step Gradio ``/call`` + SSE)."""

    def __init__(
        self, hf_token: str, *, url: str = PROXY_CAPTION_URL, timeout: float = 300.0
    ) -> None:
        if not url.startswith("https://"):
            raise ValueError("proxy url must be https:// (no file:/ or custom schemes)")
        self._token = hf_token
        self._url = url
        self._timeout = timeout

    def caption(
        self, model: str, prompt: str, image_bytes: bytes
    ) -> dict[str, object]:  # pragma: no cover - network
        """Relay one caption request through the proxy and return its result dict."""
        b64 = base64.b64encode(image_bytes).decode()
        auth = {"Authorization": f"Bearer {self._token}"}
        submit = urllib.request.Request(
            self._url,
            data=json.dumps({"data": [model, prompt, b64]}).encode(),
            headers={**auth, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(submit, timeout=self._timeout) as resp:  # nosec B310 - https only
            event_id = json.load(resp)["event_id"]
        stream = urllib.request.Request(f"{self._url}/{event_id}", headers=auth)
        last: str | None = None
        with urllib.request.urlopen(stream, timeout=self._timeout) as events:  # nosec B310 - https only
            for raw in events:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload and payload != "null":
                        last = payload
        return parse_proxy_payload(last)
