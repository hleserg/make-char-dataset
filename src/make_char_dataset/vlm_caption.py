"""VLM (Gemini prose) captioner for the CHARACTER LoRA — via the proxy Space.

Produces trigger-first natural-language captions that describe **only what varies**
(pose, expression, outfit, props, background, shot, lighting) and **never** the
character's fixed identity (face/hair/eyes/skin/body/age) or the art style — so the
identity binds to the ``<char>_char`` trigger and the style stays with the external
``cmcstyle`` LoRA. This is the inverse of the *style* captioner in make-style-dataset.

Every Gemini call goes through :mod:`make_char_dataset.proxy` (this box is
geo-blocked). The network client is injected via the ``CaptionClient`` Protocol, so
the prompt/downscale/normalise/retry logic here is unit-tested with a fake.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

    from make_char_dataset.proxy import CaptionClient


class VlmCaptionError(Exception):
    """Raised when the VLM proxy cannot produce a caption for an image."""


def build_char_prompt(trigger: str) -> str:
    """Return the character-captioner instruction (trigger substituted).

    Describe-to-exclude: the model is told to omit the constant identity (so it binds
    to ``trigger``) and the art style (owned by the external style LoRA), and to
    describe only the variable content — including the outfit, which we keep
    *re-dressable* by binding it to words rather than the trigger.
    """
    return (
        "You are captioning training images for a CHARACTER LoRA. The character's identity "
        "must bind to the trigger token, so describe ONLY what varies between images and "
        "NEVER describe the character's fixed identity or the art style.\n\n"
        "Rules:\n"
        f'- Begin the caption with the exact trigger token: "{trigger}".\n'
        "- Describe ONLY: pose and action; facial expression / emotion; clothing and outfit; "
        "held props or objects; the background and setting; the camera shot (close-up / "
        "waist-up / full-body); the angle; and the lighting.\n"
        "- DO NOT describe the character's fixed identity: face shape, hair colour or style, "
        "eye colour, skin tone, body type / build, age, gender, or permanent marks — anything "
        "constant across images. (If a feature genuinely differs in THIS image, you may "
        "describe that difference.)\n"
        "- DO NOT name the art style, medium, palette, colours, or rendering technique.\n"
        "- Write natural-language prose, 3 to 6 sentences. No tag lists, no bullet points.\n"
        "Output only the caption text."
    )


def normalize_caption(text: str, trigger: str) -> str:
    """Collapse to one line and guarantee a single leading ``'<trigger>, '`` prefix.

    Strips an existing leading trigger only when it is a whole token (followed by a
    separator or end of string), so a word that merely *starts with* the trigger
    (e.g. ``'kaelin'`` for trigger ``'kael'``) is never split mid-word.
    """
    one_line = " ".join((text or "").split())
    rest = one_line[len(trigger) :]
    if one_line.lower().startswith(trigger.lower()) and (not rest or rest[0] in " ,.;:"):
        one_line = rest.lstrip(" ,")
    return f"{trigger}, {one_line}".rstrip()


def downscale_image(data: bytes, max_side: int) -> bytes:
    """Downscale ``data`` to ``max_side`` longest edge as JPEG (smaller proxy upload).

    Full-resolution (1024²+) bodies stall the proxy's HTTPS upload (write timeout); a
    768 px JPEG (~40 KB) uploads instantly and is ample for a VLM caption.
    """
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((max_side, max_side))
        out = io.BytesIO()
        rgb.save(out, format="JPEG", quality=90)
    return out.getvalue()


def _is_transient(error: object) -> bool:
    """True for retryable errors (5xx, 429, network/timeout, empty/parse).

    Definitive client errors — HTTP 4xx other than 429 — are not retried; everything
    else (including transient network exceptions) is worth another attempt.
    """
    err = str(error)
    if not err:
        return False
    return not (err.startswith("http_4") and err != "http_429")


class VlmCaptioner:
    """Caption one image as character prose via the proxy ``CaptionClient``."""

    def __init__(
        self,
        client: CaptionClient,
        *,
        trigger: str,
        model: str = "gemini-2.5-flash",
        max_side: int = 768,
        retries: int = 5,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._trigger = trigger
        self._model = model
        self._max_side = max_side
        self._retries = retries
        self._backoff = backoff
        self._sleep = sleep
        self._prompt = build_char_prompt(trigger)

    def caption(self, image: Path) -> str:
        """Return a normalised, trigger-first prose caption for ``image`` (retries transient)."""
        data = downscale_image(image.read_bytes(), self._max_side)
        result: dict[str, object] = {"error": "not_run"}
        for attempt in range(self._retries):
            try:
                result = self._client.caption(self._model, self._prompt, data)
            except Exception as exc:
                result = {"error": "exception", "detail": f"{type(exc).__name__}: {exc}"[:160]}
            if "error" not in result or not _is_transient(result["error"]):
                break
            if attempt < self._retries - 1:  # no backoff after the final attempt
                self._sleep(self._backoff * (attempt + 1))
        text = result.get("caption")
        if result.get("error") or not text:
            raise VlmCaptionError(
                f"VLM caption failed for {image.name}: {result.get('error') or 'empty_caption'}"
            )
        return normalize_caption(str(text), self._trigger)
