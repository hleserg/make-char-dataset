"""Application settings.

Single, typed source of runtime configuration. Reads from environment
variables and an optional local ``.env`` file. Secrets must never be
committed — keep them in ``.env`` (git-ignored) and document keys in
``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
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

    # --- Pipeline: captioning (VLM prose by default) ---
    # The caption stage describes only what VARIES (pose/outfit/expression/bg/shot)
    # and omits the invariant identity + the art style, so identity binds to the
    # <char>_char trigger and style stays with the external cmcstyle LoRA. VLM prose
    # (Gemini via the proxy Space) is the default; WD14 tagging is a flagged fallback.
    captioner: str = Field(
        default="vlm",
        description="Captioning backend: 'vlm' (Gemini prose via the proxy; default), "
        "'wd14' (legacy booru tagger + Character-Locker), or 'stub' (deterministic, no "
        "network — used by tests/CI).",
    )
    vlm_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for VLM captioning (agent default; the UI button uses pro).",
    )
    vlm_concurrency: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Concurrent proxy caption requests (the proxy Space queue limit is 16).",
    )
    vlm_max_image_side: int = Field(
        default=768,
        ge=64,
        description="Downscale each image to this longest side before captioning. Full-res "
        "(1024+) bodies time out the proxy upload; 768 keeps it fast and is plenty for a VLM.",
    )
    vlm_proxy_url: str = Field(
        default="https://hleserg-proxy-gemini-api.hf.space/gradio_api/call/caption",
        description="Gradio endpoint of the Gemini proxy Space (Gemini is geo-blocked here, so "
        "every call is relayed through it).",
    )
    # WD14 fallback (only when APP_CAPTIONER=wd14): the ONNX model + label CSV are
    # human-provided (HLE-759) and have no defaults, so the stage fails clearly if
    # wd14 is selected without them rather than crashing deep in onnxruntime.
    wd14_model_path: str = Field(
        default="", description="WD14 ONNX model path (required when APP_CAPTIONER=wd14)."
    )
    wd14_labels_path: str = Field(
        default="", description="WD14 selected_tags.csv path (required when APP_CAPTIONER=wd14)."
    )
    wd14_threshold: float = Field(
        default=0.35, ge=0.0, le=1.0, description="WD14 general-tag confidence cutoff."
    )

    # --- Pipeline: char-LoRA training (heavy; opt-in; runs via ai-toolkit) ---
    # The char-LoRA trains on Flux.1-dev so it stacks with the cmcstyle style LoRA
    # (`Flux + cmcstyle + <char>_char`, HLE-802). Training shells out to ostris
    # ai-toolkit — NOT kohya sd-scripts: kohya loads the whole Flux DiT before
    # block-swap offload and OOMs ~16 GB, whereas ai-toolkit qfloat8-quantizes the
    # transformer + text encoder so Flux LoRA training fits (the bit kohya can't do).
    train_tool: str = Field(
        default="ai-toolkit",
        description="LoRA training toolchain. Only 'ai-toolkit' (ostris) is wired.",
    )
    aitoolkit_dir: Path = Field(
        default=Path("/home/serg/ai-toolkit"),
        description="Local clone of ostris/ai-toolkit (holds run.py and its own venv).",
    )
    train_python: str = Field(
        default="",
        description="Interpreter to launch ai-toolkit with; empty uses "
        "<aitoolkit_dir>/venv/bin/python (its cu128/sm_120 venv).",
    )
    train_base_model: str = Field(
        default="black-forest-labs/FLUX.1-dev",
        description="Flux base (HF id or local path). MUST equal the style LoRA's base "
        "so 'Flux + cmcstyle + <char>_char' stacks (HLE-802).",
    )
    train_output_name: str = Field(
        default="",
        description="char-LoRA filename stem (and ai-toolkit run name); empty uses the "
        "trigger token, so the LoRA is '<trigger>.safetensors'.",
    )
    train_network_dim: int = Field(default=32, ge=1, description="LoRA rank (network.linear).")
    train_network_alpha: int = Field(
        default=16, ge=1, description="LoRA alpha (network.linear_alpha)."
    )
    train_steps: int = Field(
        default=2200, ge=1, description="Total training steps (500-4000 is a good range)."
    )
    train_learning_rate: float = Field(default=1e-4, gt=0.0, description="Optimizer learning rate.")
    train_batch_size: int = Field(default=1, ge=1, description="Training batch size.")
    train_resolution: int = Field(
        default=512,
        ge=64,
        description="Training/bucket resolution (px). Low-VRAM Flux fits 512 on ~16 GB; "
        "bump to 768 only after a fit-check.",
    )
    train_optimizer: str = Field(
        default="adafactor",
        description="ai-toolkit optimizer. adafactor has near-zero optimizer state, so it "
        "fits ~16 GB; adamw8bit needs more VRAM.",
    )
    train_save_every: int = Field(
        default=250, ge=1, description="Save an intermittent LoRA every N steps."
    )
    train_gradient_checkpointing: bool = Field(
        default=True, description="Trade compute for VRAM (needed to fit Flux)."
    )
    train_quantize: bool = Field(
        default=True,
        description="Quantize the Flux base for low-VRAM training (ai-toolkit; kohya cannot).",
    )
    train_qtype: str = Field(
        default="qfloat8",
        description="Transformer quantization type. Use 'qfloat8' — fp8 casts on CPU under "
        "low_vram. NOT 'qint4': its int4pack kernel is CUDA-only and clashes with low_vram's "
        "CPU quantization, throwing and silently producing no weights.",
    )
    train_qtype_te: str = Field(
        default="qfloat8", description="Text-encoder quantization type (see train_qtype)."
    )
    train_low_vram: bool = Field(
        default=True,
        description="ai-toolkit low_vram mode (quantize on CPU; slower, far less VRAM). "
        "Mandatory here — the GPU drives the display, so the load peak must stay low.",
    )
    train_disable_sampling: bool = Field(
        default=True,
        description="Skip ai-toolkit's in-training sample renders. Stack eval (cmcstyle + "
        "<char>_char) is a separate stage; ai-toolkit can only sample the char LoRA alone.",
    )
    train_use_ema: bool = Field(
        default=False,
        description="Enable ai-toolkit EMA weight smoothing. Off to match the proven "
        "low-VRAM Flux recipe (ohwxwoman_flux.yaml: use_ema=false); turn on for a "
        "potentially smoother result if VRAM allows.",
    )
    huggingface_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "APP_HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"
        ),
        description="HF token (read from HF_TOKEN) used by ai-toolkit to fetch the gated "
        "FLUX.1-dev base. Empty is fine when train_base_model is a local path. Secret — "
        "keep it in .env, never commit it.",
    )

    # --- Pipeline: in-stack eval grid (Flux + cmcstyle + <char>_char; opt-in) ---
    # The DoD acceptance harness: render trigger-isolation cells (base / cmcstyle-only
    # / <char>_char-only / stack) through ComfyUI so a human can judge that identity
    # holds, style holds, and neither overrides the other.
    eval_style_trigger: str = Field(
        default="cmcstyle", description="Style LoRA trigger token, prepended to eval prompts."
    )
    eval_char_lora_path: str = Field(
        default="",
        description="char-LoRA .safetensors (in ComfyUI/models/loras) for eval; empty derives "
        "'<train_output_name|trigger>.safetensors' from the train stage.",
    )
    eval_char_lora_weight: float = Field(
        default=0.9, ge=0.0, le=2.0, description="char-LoRA weight at eval inference."
    )
    eval_prompts_file: str = Field(
        default="",
        description="Optional file (one scene prompt per line) overriding the default eval scenes.",
    )
    eval_steps: int = Field(default=20, ge=1, description="Flux sampling steps at eval.")
    eval_guidance: float = Field(
        default=3.5,
        gt=0.0,
        description="Flux inference guidance (~3.5; NOT the training guidance).",
    )
    eval_size: int = Field(default=1024, ge=64, description="Eval render size (px, square).")
    eval_seed: int = Field(default=42, ge=0, description="Fixed eval seed (cells stay comparable).")
    eval_unet: str = Field(
        default="flux1-dev-fp8.safetensors",
        description="ComfyUI UNETLoader filename (models/unet).",
    )
    eval_clip_l: str = Field(default="clip_l.safetensors", description="ComfyUI clip_l filename.")
    eval_t5xxl: str = Field(default="t5xxl_fp16.safetensors", description="ComfyUI t5xxl filename.")
    eval_vae: str = Field(default="ae.safetensors", description="ComfyUI Flux VAE filename.")

    # --- Pipeline: stage flags (gate which stages 'run-all' executes) ---
    run_import: bool = True
    run_generate: bool = True
    run_clean: bool = True
    run_caption: bool = True
    # Training is heavy (a GPU + the Flux base) and opt-in: 'run-all' skips it by
    # default; run it explicitly with `make-char-dataset train`.
    run_train: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the environment is parsed once. In tests, call
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()
