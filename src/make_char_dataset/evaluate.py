"""Eval stage: the in-stack acceptance grid for ``Flux + cmcstyle + <char>_char``.

Renders a fixed-seed contact sheet so the trained char-LoRA can be judged **in the
stack** (the HLE-804 DoD): rows are prompts, columns are the four **trigger-isolation**
cells —

* ``base``  — no LoRA (proves the LoRAs are what *add* identity + style);
* ``style`` — cmcstyle only (style without the specific character);
* ``char``  — ``<char>_char`` only (identity without the style);
* ``stack`` — both (the target: the character rendered in the comic style).

The same prompt (carrying both triggers) is used across a row, so only the loaded
LoRAs vary — making (a) identity holds, (b) style holds, (c) neither overrides, and
(d) the triggers stay isolated all readable at a glance.

Pure-core / lazy-backend (the split used across the stages): the ComfyUI Flux
txt2img **graph builder**, the cell plan and the Pillow **grid assembler** are pure
and unit-tested; only the ComfyUI HTTP sampler (:class:`ComfyStackSampler`) is
uncovered. A :class:`StubStackSampler` renders deterministic placeholder tiles so
the grid + ``--dry-run`` work with no GPU and no ComfyUI server.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from PIL import Image, ImageDraw, ImageFont

from make_char_dataset.config import get_settings
from make_char_dataset.workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from make_char_dataset.config import Settings

GRID_FILENAME = "stack_eval_grid.png"
MANIFEST_FILENAME = "stack_eval.json"

#: The four trigger-isolation columns, in display order.
MODES: tuple[str, ...] = ("base", "style", "char", "stack")

#: Default scene prompts (the character + style triggers are prepended per cell).
#: A few varied poses/shots so identity + prompt-following are both visible.
DEFAULT_SCENES: tuple[str, ...] = (
    "standing in a sunlit forest clearing, full body",
    "close-up portrait, determined expression",
    "walking down a rain-slicked street at night, medium shot",
    "sitting at a wooden tavern table, mid shot",
    "low-angle hero shot against a cloudy sky",
)

#: ComfyUI model filenames on this box (models/{unet,clip,vae}); overridable.
DEFAULT_UNET = "flux1-dev-fp8.safetensors"
DEFAULT_CLIP_L = "clip_l.safetensors"
DEFAULT_T5XXL = "t5xxl_fp16.safetensors"
DEFAULT_VAE = "ae.safetensors"


class EvalError(Exception):
    """Raised when the eval stage cannot run (e.g. no char LoRA configured)."""


@dataclass(frozen=True)
class EvalCell:
    """One grid cell: a prompt row crossed with a trigger-isolation column."""

    row: int
    prompt: str
    mode: str  # one of MODES


@runtime_checkable
class StackSampler(Protocol):
    """Renders one ComfyUI API-format graph to PNG image bytes."""

    def sample(self, graph: dict) -> bytes:
        """Run ``graph`` and return the output image's PNG bytes. Raise on failure."""
        ...


# --- Pure: prompt + LoRA cell plan, the Flux graph, the contact sheet -------


def build_prompt(scene: str, *, style_trigger: str, char_trigger: str) -> str:
    """Compose the shared per-row prompt: both triggers first, then the scene."""
    parts = [part for part in (style_trigger.strip(), char_trigger.strip(), scene.strip()) if part]
    return ", ".join(parts)


def loras_for_mode(
    mode: str,
    *,
    style_lora: str,
    style_weight: float,
    char_lora: str,
    char_weight: float,
) -> list[tuple[str, float]]:
    """LoRAs (name, weight) active for a trigger-isolation ``mode`` (stack order)."""
    style = [(style_lora, style_weight)] if style_lora else []
    char = [(char_lora, char_weight)] if char_lora else []
    if mode == "base":
        return []
    if mode == "style":
        return style
    if mode == "char":
        return char
    if mode == "stack":
        return style + char
    raise EvalError(f"unknown eval mode {mode!r}; expected one of {MODES}")


def make_cells(prompts: Sequence[str], modes: Sequence[str] = MODES) -> list[EvalCell]:
    """Cross every prompt row with every trigger-isolation column, in display order."""
    return [
        EvalCell(row=row, prompt=prompt, mode=mode)
        for row, prompt in enumerate(prompts)
        for mode in modes
    ]


def build_flux_workflow(
    *,
    prompt: str,
    loras: Sequence[tuple[str, float]],
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    unet: str = DEFAULT_UNET,
    clip_l: str = DEFAULT_CLIP_L,
    t5xxl: str = DEFAULT_T5XXL,
    vae: str = DEFAULT_VAE,
) -> dict:
    """Return a ComfyUI API-format Flux txt2img graph stacking ``loras`` (pure; no I/O).

    Each ``(name, weight)`` chains a ``LoraLoaderModelOnly`` onto the model, so 0
    (``base``), 1 (``style``/``char``) or 2 (``stack``) LoRAs compose cleanly.
    Core ComfyUI node ``class_type``s only; cfg=1.0 (Flux uses FluxGuidance).
    """
    graph: dict[str, dict] = {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
        "11": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": clip_l, "clip_name2": t5xxl, "type": "flux"},
        },
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model_ref: list = ["10", 0]
    for index, (name, weight) in enumerate(loras):
        node_id = f"13{index}"
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": name, "strength_model": weight},
        }
        model_ref = [node_id, 0]

    graph["20"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": prompt}}
    graph["21"] = {
        "class_type": "FluxGuidance",
        "inputs": {"conditioning": ["20", 0], "guidance": guidance},
    }
    graph["22"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": ""}}
    graph["30"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    graph["40"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["21", 0],
            "negative": ["22", 0],
            "latent_image": ["30", 0],
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }
    graph["50"] = {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["12", 0]}}
    graph["60"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["50", 0], "filename_prefix": "stack_eval"},
    }
    return graph


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:  # pragma: no cover - font availability is environment-dependent
            continue
    return ImageFont.load_default()  # pragma: no cover - only if no TrueType font is installed


def _mode_label(mode: str, char_trigger: str) -> str:
    return {
        "base": "base (no LoRA)",
        "style": "cmcstyle only",
        "char": f"{char_trigger} only",
        "stack": "stack (both)",
    }.get(mode, mode)


def assemble_grid(
    images: dict[tuple[int, str], Image.Image | None],
    scenes: Sequence[str],
    char_trigger: str,
    *,
    modes: Sequence[str] = MODES,
    thumb: int = 320,
    label_w: int = 240,
    header_h: int = 36,
) -> Image.Image:
    """Lay out the labelled contact sheet: rows = scenes, columns = isolation modes."""
    cols, rows = len(modes), len(scenes)
    sheet = Image.new("RGB", (label_w + cols * thumb, header_h + rows * thumb), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    head_font, body_font = _font(18), _font(14)

    for col, mode in enumerate(modes):
        draw.text(
            (label_w + col * thumb + 8, 8),
            _mode_label(mode, char_trigger),
            fill=(20, 20, 20),
            font=head_font,
        )
    for row, scene in enumerate(scenes):
        y = header_h + row * thumb
        for i, line in enumerate(_wrap(scene, body_font, label_w - 16)):
            draw.text((8, y + 8 + i * 17), line, fill=(20, 20, 20), font=body_font)
        for col, mode in enumerate(modes):
            x = label_w + col * thumb
            img = images.get((row, mode))
            if img is None:
                draw.rectangle([x + 2, y + 2, x + thumb - 2, y + thumb - 2], outline=(200, 60, 60))
                draw.text(
                    (x + 10, y + thumb // 2), "(no image)", fill=(200, 60, 60), font=body_font
                )
            else:
                fitted = img.copy()
                fitted.thumbnail((thumb - 4, thumb - 4))
                sheet.paste(fitted, (x + 2, y + 2))
    return sheet


def _wrap(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if font.getbbox(trial)[2] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


# --- Backends: real ComfyUI sampler + a GPU-free stub -----------------------


class ComfyStackSampler:  # pragma: no cover - network/GPU
    """Render a graph through a running ComfyUI server (submit → poll → fetch)."""

    def __init__(self, base_url: str, *, timeout: float = 600.0, poll: float = 1.5) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._poll = poll

    def sample(self, graph: dict) -> bytes:
        """Queue ``graph``, wait for completion, return the first output image's bytes."""
        import time
        import urllib.parse
        import urllib.request

        body = json.dumps({"prompt": graph}).encode()
        req = urllib.request.Request(
            f"{self._base}/prompt", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310 - configured comfy url
            prompt_id = json.loads(resp.read())["prompt_id"]
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            with urllib.request.urlopen(  # nosec B310
                f"{self._base}/history/{prompt_id}", timeout=15
            ) as resp:
                entry = json.loads(resp.read()).get(prompt_id)
            if entry:
                for node_out in entry.get("outputs", {}).values():
                    for image in node_out.get("images", []):
                        query = urllib.parse.urlencode(image)
                        with urllib.request.urlopen(  # nosec B310
                            f"{self._base}/view?{query}", timeout=30
                        ) as image_resp:
                            return bytes(image_resp.read())
                raise EvalError(f"ComfyUI prompt {prompt_id} finished with no image")
            time.sleep(self._poll)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {self._timeout:.0f}s")


@dataclass
class StubStackSampler:
    """GPU-free sampler: a deterministic solid tile per cell (for tests + --dry-run)."""

    size: int = 64

    def sample(self, graph: dict) -> bytes:
        """Return a small PNG whose colour is a deterministic hash of the graph."""
        digest = abs(hash(json.dumps(graph, sort_keys=True)))
        colour = (digest % 256, (digest // 256) % 256, (digest // 65536) % 256)
        buffer = io.BytesIO()
        Image.new("RGB", (self.size, self.size), colour).save(buffer, format="PNG")
        return buffer.getvalue()


# --- Orchestration ----------------------------------------------------------


def resolve_char_lora(settings: Settings) -> str:
    """Char-LoRA filename for eval: ``eval_char_lora_path`` or the trained output stem."""
    if settings.eval_char_lora_path.strip():
        return settings.eval_char_lora_path.strip()
    name = settings.train_output_name.strip() or settings.trigger_token
    return f"{name}.safetensors"


def load_scenes(settings: Settings) -> list[str]:
    """Eval scene prompts from ``eval_prompts_file`` (one per line), else the defaults."""
    path = settings.eval_prompts_file.strip()
    if not path:
        return list(DEFAULT_SCENES)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    scenes = [line.strip() for line in lines if line.strip()]
    if not scenes:
        raise EvalError(f"eval prompts file {path} is empty")
    return scenes


def build_manifest(settings: Settings, *, scenes: Sequence[str], char_lora: str) -> dict:
    """Provenance for the eval run (what was loaded, which prompts/triggers)."""
    return {
        "stack": "Flux + cmcstyle + <char>_char",
        "char_trigger": settings.trigger_token,
        "style_trigger": settings.eval_style_trigger,
        "char_lora": char_lora,
        "char_lora_weight": settings.eval_char_lora_weight,
        "style_lora": Path(settings.style_lora_path).name if settings.style_lora_path else "",
        "style_lora_weight": settings.style_lora_weight,
        "modes": list(MODES),
        "scenes": list(scenes),
        "seed": settings.eval_seed,
        "guidance": settings.eval_guidance,
    }


def run_eval(*, sampler: StackSampler | None = None, dry_run: bool = False) -> Path:
    """Render the in-stack eval grid into ``07_eval/`` and return the grid path.

    Builds the 4-column trigger-isolation cells per prompt, renders each through
    ``sampler`` (the ComfyUI backend by default), assembles the contact sheet, and
    writes a provenance manifest. ``dry_run`` writes the per-cell graphs (and an
    empty-tile grid) without calling ComfyUI — so the harness validates with no GPU.
    """
    settings = get_settings()
    workspace = Workspace(settings.workspace)
    out = workspace.eval_dir
    out.mkdir(parents=True, exist_ok=True)

    style_lora = Path(settings.style_lora_path).name if settings.style_lora_path else ""
    char_lora = resolve_char_lora(settings)
    scenes = load_scenes(settings)
    prompts = [
        build_prompt(
            scene, style_trigger=settings.eval_style_trigger, char_trigger=settings.trigger_token
        )
        for scene in scenes
    ]
    runner = sampler if sampler is not None else ComfyStackSampler(settings.comfy_url)

    images: dict[tuple[int, str], Image.Image | None] = {}
    for cell in make_cells(prompts):
        graph = build_flux_workflow(
            prompt=cell.prompt,
            loras=loras_for_mode(
                cell.mode,
                style_lora=style_lora,
                style_weight=settings.style_lora_weight,
                char_lora=char_lora,
                char_weight=settings.eval_char_lora_weight,
            ),
            seed=settings.eval_seed,
            width=settings.eval_size,
            height=settings.eval_size,
            steps=settings.eval_steps,
            guidance=settings.eval_guidance,
            unet=settings.eval_unet,
            clip_l=settings.eval_clip_l,
            t5xxl=settings.eval_t5xxl,
            vae=settings.eval_vae,
        )
        if dry_run:
            (out / f"graph_r{cell.row}_{cell.mode}.json").write_text(
                json.dumps(graph, indent=2), encoding="utf-8"
            )
            images[(cell.row, cell.mode)] = None
            continue
        images[(cell.row, cell.mode)] = Image.open(io.BytesIO(runner.sample(graph)))

    grid = assemble_grid(images, scenes, settings.trigger_token)
    grid_path = out / GRID_FILENAME
    grid.save(grid_path)
    (out / MANIFEST_FILENAME).write_text(
        json.dumps(build_manifest(settings, scenes=scenes, char_lora=char_lora), indent=2) + "\n",
        encoding="utf-8",
    )
    return grid_path
