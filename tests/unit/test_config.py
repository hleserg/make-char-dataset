"""Tests for make_char_dataset.config."""

from __future__ import annotations

import pytest

from make_char_dataset.config import Settings, get_settings


def test_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.environment == "development"
    assert settings.debug is False
    assert settings.sentry_dsn == ""


def test_singleton_is_cached() -> None:
    assert get_settings() is get_settings()


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    monkeypatch.setenv("APP_DEBUG", "true")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.environment == "production"
    assert settings.debug is True


def test_training_defaults() -> None:
    settings = get_settings()
    # Char-LoRA trains on Flux via ai-toolkit, opt-in, with qint4 low-VRAM quant.
    assert settings.train_tool == "ai-toolkit"
    assert settings.train_base_model == "black-forest-labs/FLUX.1-dev"
    assert settings.train_quantize is True
    assert settings.train_qtype == "qint4"
    assert settings.train_low_vram is True
    assert settings.run_train is False


def test_huggingface_token_reads_hf_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The HF token is aliased to the standard HF_TOKEN (no APP_ prefix needed).
    monkeypatch.setenv("HF_TOKEN", "hf_abc123")
    get_settings.cache_clear()
    assert get_settings().huggingface_token == "hf_abc123"
