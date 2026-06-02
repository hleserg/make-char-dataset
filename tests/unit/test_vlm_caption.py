"""Tests for make_char_dataset.vlm_caption (character VLM captioner).

Pure prompt/normalise/downscale helpers + the captioner orchestration with a fake
``CaptionClient`` — no network. The real proxy client is tested/marked elsewhere.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from make_char_dataset.vlm_caption import (
    VlmCaptioner,
    VlmCaptionError,
    _is_transient,
    build_char_prompt,
    downscale_image,
    normalize_caption,
)


def _image(path: Path, size: int = 1024) -> Path:
    # Random noise so the full-res PNG is genuinely large (incompressible) and the
    # downscaled JPEG is unambiguously smaller.
    pixels = np.random.default_rng(0).integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


class FakeClient:
    """Returns queued results (dicts) or raises queued exceptions, recording calls."""

    def __init__(self, results: list[object]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str, bytes]] = []

    def caption(self, model: str, prompt: str, image_bytes: bytes) -> dict[str, object]:
        self.calls.append((model, prompt, image_bytes))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return dict(result)  # type: ignore[arg-type]


# --- prompt / normalise ------------------------------------------------------


def test_build_char_prompt_inversion() -> None:
    prompt = build_char_prompt("kael_char")
    assert 'Begin the caption with the exact trigger token: "kael_char"' in prompt
    assert "DO NOT describe the character's fixed identity" in prompt
    assert "body type / build" in prompt  # build is fixed -> binds to trigger
    assert "clothing and outfit" in prompt  # outfit is described -> re-dressable
    assert "DO NOT name the art style" in prompt
    assert prompt.strip().endswith("Output only the caption text.")


def test_normalize_caption_adds_trigger_prefix() -> None:
    assert normalize_caption("a person standing", "kael") == "kael, a person standing"


def test_normalize_caption_dedupes_existing_trigger() -> None:
    # The VLM often emits "kael The character..." (no comma); collapse to one prefix.
    assert normalize_caption("kael  The character stands", "kael") == "kael, The character stands"


def test_normalize_caption_collapses_multiline() -> None:
    assert normalize_caption("line one\n  line two", "kael") == "kael, line one line two"


def test_normalize_caption_does_not_split_a_word_starting_with_trigger() -> None:
    # 'kaelin' merely starts with 'kael' — must not be split into 'kael, in ...'.
    assert normalize_caption("kaelin standing", "kael") == "kael, kaelin standing"


# --- downscale ---------------------------------------------------------------


def test_downscale_image_shrinks_and_is_jpeg(tmp_path: Path) -> None:
    big = _image(tmp_path / "big.png", size=1024).read_bytes()
    small = downscale_image(big, 256)
    with Image.open(io.BytesIO(small)) as out:
        assert max(out.size) == 256
        assert out.format == "JPEG"
    assert len(small) < len(big)  # smaller payload for the proxy upload


# --- transient classification ------------------------------------------------


@pytest.mark.parametrize("err", ["http_503", "http_429", "exception", "empty_response", "timeout"])
def test_is_transient_true(err: str) -> None:
    assert _is_transient(err) is True


@pytest.mark.parametrize("err", ["http_400", "http_401", "http_403", "http_404"])
def test_is_transient_false(err: str) -> None:
    assert _is_transient(err) is False


def test_is_transient_empty_is_false() -> None:
    assert _is_transient("") is False


# --- VlmCaptioner ------------------------------------------------------------


def test_captioner_success_normalises_and_downscales(tmp_path: Path) -> None:
    img = _image(tmp_path / "var_000.png")
    client = FakeClient([{"caption": "a figure standing in a field"}])
    captioner = VlmCaptioner(client, trigger="kael", model="m", max_side=256)
    assert captioner.caption(img) == "kael, a figure standing in a field"
    model, prompt, sent = client.calls[0]
    assert model == "m" and "kael" in prompt
    assert len(sent) < img.stat().st_size  # the downscaled JPEG was sent, not the 1 MB PNG


def test_captioner_retries_then_succeeds(tmp_path: Path) -> None:
    img = _image(tmp_path / "var_000.png")
    client = FakeClient([{"error": "http_503"}, {"caption": "standing"}])
    captioner = VlmCaptioner(client, trigger="kael", retries=3, backoff=0.0, sleep=lambda _: None)
    assert captioner.caption(img) == "kael, standing"
    assert len(client.calls) == 2  # retried once


def test_captioner_definitive_error_no_retry(tmp_path: Path) -> None:
    img = _image(tmp_path / "var_000.png")
    client = FakeClient([{"error": "http_400"}])
    captioner = VlmCaptioner(client, trigger="kael", retries=3, backoff=0.0, sleep=lambda _: None)
    with pytest.raises(VlmCaptionError, match="http_400"):
        captioner.caption(img)
    assert len(client.calls) == 1  # 4xx is not retried


def test_captioner_exhausts_transient_then_raises(tmp_path: Path) -> None:
    img = _image(tmp_path / "var_000.png")
    client = FakeClient([{"error": "http_503"}, {"error": "http_503"}])
    sleeps: list[float] = []
    captioner = VlmCaptioner(client, trigger="kael", retries=2, backoff=1.0, sleep=sleeps.append)
    with pytest.raises(VlmCaptionError):
        captioner.caption(img)
    assert len(client.calls) == 2
    assert sleeps == [1.0]  # backoff between the 2 attempts, but NOT after the final one


def test_captioner_handles_client_exception(tmp_path: Path) -> None:
    img = _image(tmp_path / "var_000.png")
    client = FakeClient([RuntimeError("boom"), {"caption": "standing"}])
    captioner = VlmCaptioner(client, trigger="kael", retries=3, backoff=0.0, sleep=lambda _: None)
    assert captioner.caption(img) == "kael, standing"  # exception treated as transient → retried
