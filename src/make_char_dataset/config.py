"""Application settings.

Single, typed source of runtime configuration. Reads from environment
variables and an optional local ``.env`` file. Secrets must never be
committed — keep them in ``.env`` (git-ignored) and document keys in
``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment.

    # PLAYBOOK-START
    # id: typed-settings-singleton
    # title: Typed settings as a cached singleton
    # status: refined
    # category: configuration
    # tags: [pydantic, config, 12factor]
    # Centralize all env access in one typed object resolved once via an
    # lru_cache'd accessor. Code never reads os.environ directly; tests
    # override by clearing the cache. Substitution test passes: useful in
    # any 12-factor service regardless of domain.
    # PLAYBOOK-END
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    environment: str = "development"
    debug: bool = False

    sentry_dsn: str = ""
    sentry_environment: str = "development"
    sentry_release: str = ""
    sentry_traces_sample_rate: float = 0.0

    # --- Pipeline: workspace + dataset identity ---
    workspace: Path = Field(
        default=Path("workspace"),
        description="Root directory holding every stage folder (input, generated, dataset).",
    )
    trigger_token: str = Field(
        default="character",
        description="Character LoRA trigger word; also names the kohya training folder.",
    )
    dataset_repeats: int = Field(
        default=10,
        ge=1,
        description="kohya repeat count; the dataset folder is named '<repeats>_<trigger>'.",
    )
    keep_tokens: int = Field(
        default=1,
        ge=0,
        description="kohya keep_tokens: leading caption tokens (the trigger) pinned in shuffle.",
    )

    # --- Pipeline: generation backend (heavy; injected behind a Protocol) ---
    backend: str = Field(
        default="comfyui",
        description="Generation backend: 'comfyui' (HTTP+websocket), 'diffusers', or 'stub'.",
    )
    comfy_url: str = Field(
        default="http://127.0.0.1:8188",
        description="ComfyUI server base URL (used when backend='comfyui').",
    )
    base_model: str = Field(
        default="sdxl",
        description="Build-against base model family: 'sdxl' (default today) or 'flux' (HLE-670).",
    )
    variants_per_character: int = Field(
        default=36,
        ge=1,
        description="How many local variants to multiply each character into (recipe: ~30-40).",
    )
    controlnet: str = Field(
        default="openpose",
        description="License-safe pose/structure guidance: 'openpose', 'depth', or 'none'. "
        "InsightFace-based tools (PuLID/IP-Adapter/InstantID) are forbidden for commercial use.",
    )
    controlnet_model: str = Field(
        default="",
        description="ControlNet checkpoint filename for the chosen mode (human-provided, "
        "HLE-759); empty disables the ControlNet subgraph until a model is supplied.",
    )
    img2img_denoise: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="img2img denoise strength for anchor multiplication (recipe: 0.7-0.8).",
    )

    # --- Pipeline: external style LoRA (Style Locker — style lives outside the character) ---
    style_lora_path: str = Field(
        default="",
        description="Path to the pre-trained style LoRA .safetensors (provided by the human).",
    )
    style_lora_weight: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        description="Weight for the style LoRA at generation time (recipe: 0.8-1.0).",
    )
    style_prompt: str = Field(
        default="",
        description="Style prompt / [STYLE] trigger applied at generation (kept out of captions).",
    )

    # --- Pipeline: dataset cleaning + layout ---
    dedup_hamming_distance: int = Field(
        default=6,
        ge=0,
        description="Perceptual-hash distance below which two generated variants are near-dups.",
    )
    min_side_px: int = Field(
        default=512,
        ge=1,
        description="Drop generated variants whose shorter side is below this many pixels.",
    )
    target_side: int = Field(
        default=1024,
        ge=1,
        description="Target shorter-side length (px) for dataset images (SDXL=1024, Flux=512).",
    )
    include_rejected: bool = Field(
        default=False,
        description="Include the passport's rejected/ attempts as augmentation (off: they are "
        "failed frames). Golden anchors are used as generation conditioning, never "
        "added to the training set (conditioning-only doctrine).",
    )

    # --- Pipeline: stage flags (gate which stages 'run-all' executes) ---
    run_import: bool = True
    run_generate: bool = True
    run_clean: bool = True
    run_caption: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the environment is parsed once. In tests, call
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()
